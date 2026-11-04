"""Docker helpers with wall-clock timeouts that match docker-py semantics.

docker-py 7.x notes (see /tmp/docker-py):
- ``ImageApiMixin.pull`` hardcodes ``timeout=None`` on the HTTP request and the
  streaming helper disables socket timeouts, so ``docker.from_env(timeout=…)``
  does **not** bound image pulls. We enforce a deadline with a worker thread.
- ``Container.stop(timeout=N)`` is the SIGTERM→SIGKILL grace (default 10). The
  HTTP timeout used is ``api.timeout + N``. With ``from_env(timeout=1800)`` a
  wedged daemon can block ~1810s; we temporarily lower ``api.timeout``.
- ``Container.wait(timeout=N)`` is the HTTP timeout for ``/wait`` and raises
  ``requests.exceptions.ReadTimeout`` when exceeded.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Large GPU/DB images; keep well under typical inspect task budgets.
DOCKER_PULL_TIMEOUT_SEC = 15 * 60
DOCKER_STOP_GRACE_SEC = 10
DOCKER_STOP_HTTP_TIMEOUT_SEC = 60
DOCKER_WAIT_TIMEOUT_SEC = 60
# version_command / probe containers should finish quickly
DOCKER_VERSION_TIMEOUT_SEC = 60


def pull_image(client, repository: str, *, timeout_sec: float = DOCKER_PULL_TIMEOUT_SEC, **kwargs):
    """Pull ``repository`` or raise ``TimeoutError`` after ``timeout_sec``.

    docker-py has no cancel for an in-flight pull; we run it on a daemon
    thread so a timed-out caller is not blocked waiting for the worker.
    """
    box: dict[str, Any] = {}

    def _worker() -> None:
        try:
            box["image"] = client.images.pull(repository, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — surface to caller thread
            box["exc"] = exc

    thread = threading.Thread(target=_worker, name=f"docker-pull:{repository}", daemon=True)
    thread.start()
    thread.join(timeout=timeout_sec)
    if thread.is_alive():
        raise TimeoutError(f"docker pull of {repository!r} exceeded {timeout_sec}s")
    if "exc" in box:
        raise box["exc"]
    return box["image"]


def stop_container(
    container,
    *,
    grace_sec: int = DOCKER_STOP_GRACE_SEC,
    http_timeout_sec: float = DOCKER_STOP_HTTP_TIMEOUT_SEC,
) -> None:
    """Stop a container with bounded daemon grace and HTTP wait."""
    api = container.client.api
    old_timeout = api.timeout
    # API uses conn_timeout = api.timeout + grace_sec
    api.timeout = max(1.0, float(http_timeout_sec) - float(grace_sec))
    try:
        container.stop(timeout=grace_sec)
    finally:
        api.timeout = old_timeout


def wait_container(container, *, timeout_sec: float = DOCKER_WAIT_TIMEOUT_SEC) -> dict[str, Any]:
    """Block until the container exits; ``timeout_sec`` is the HTTP wait timeout."""
    return container.wait(timeout=timeout_sec)


def run_attached(
    client,
    image: str,
    command=None,
    *,
    timeout_sec: float = DOCKER_VERSION_TIMEOUT_SEC,
    remove: bool = True,
    **docker_opts,
) -> tuple[bytes, bytes, int]:
    """Run a container to completion with a wall-clock wait timeout.

    Unlike ``containers.run`` in attach mode (no timeout), this always detaches,
    waits with ``timeout_sec``, then collects logs.
    """
    opts = dict(docker_opts)
    opts["detach"] = True
    container = client.containers.run(image, command, **opts)
    try:
        try:
            result = wait_container(container, timeout_sec=timeout_sec)
        except Exception:
            try:
                stop_container(container)
            except Exception:
                logger.exception("Failed to stop container after wait timeout")
            raise TimeoutError(
                f"docker run of {image!r} exceeded {timeout_sec}s"
            )
        stdout = container.logs(stdout=True, stderr=False)
        stderr = container.logs(stdout=False, stderr=True)
        return stdout, stderr, int(result.get("StatusCode", 1))
    finally:
        if remove:
            try:
                container.remove(force=True)
            except Exception:
                logger.exception("Failed to remove container %s", getattr(container, "id", "?"))
