import git
import logging
import os
import psutil
import random
import subprocess
import time
from urllib.parse import urlparse, urlunparse
from wrapt import synchronized


# pre-set in the ghcr.io/sparecores/sc-inspector:main image
REPO_URL = os.environ.get("REPO_URL")
REPO_PATH = os.environ.get("REPO_PATH")

# Sparse checkout prefix(es) for this inspector run (e.g. data/vultr/vhf-6c-24gb).
_sparse_paths: tuple[str, ...] | None = None

_LOW_MEMORY_GIT_CONFIG = [
    ["git", "config", "--global", "core.bigFileThreshold", "1"],
    ["git", "config", "--global", "core.deltaBaseCacheLimit", "0"],
    ["git", "config", "--global", "gc.auto", "0"],
    ["git", "config", "--global", "pack.deltaCacheLimit", "0"],
    ["git", "config", "--global", "pack.deltaCacheSize", "1"],
    ["git", "config", "--global", "pack.threads", "1"],
    ["git", "config", "--global", "pack.windowMemory", "10m"],
    ["git", "config", "--global", "pack.packSizeLimit", "20m"],
    ["git", "config", "--global", "core.packedGitWindowSize", "16m"],
    ["git", "config", "--global", "checkout.thresholdForParallelism", "99999999"],
    ["git", "config", "--global", "core.compression", "0"],
    ["git", "config", "--global", "index.threads", "1"],
]

_git_low_memory_configured = False

# Hard timeouts for git subprocesses. Without them a stalled network operation
# (seen on 2026-11-04: inspectors stuck after "Starting <task>" while ~25 VMs
# pushed concurrently) blocked forever while holding the push_paths lock, so no
# later result was ever pushed and cleanup eventually destroyed the VM.
GIT_NETWORK_COMMANDS = frozenset({"clone", "fetch", "pull", "push", "ls-remote"})
GIT_NETWORK_TIMEOUT_SEC = float(os.environ.get("GIT_NETWORK_TIMEOUT_SEC", 300))
GIT_LOCAL_TIMEOUT_SEC = float(os.environ.get("GIT_LOCAL_TIMEOUT_SEC", 600))
GIT_CLONE_TIMEOUT_SEC = float(os.environ.get("GIT_CLONE_TIMEOUT_SEC", 900))

# push_paths retry loop (unchanged values, module-level so tests can shrink them)
PUSH_DEADLINE_SEC = 10 * 60
PUSH_INITIAL_BACKOFF_SEC = 5
PUSH_MAX_BACKOFF_SEC = 2 * 60


def git_timeout(args: list[str]) -> float:
    """Timeout (seconds) for ``git <args>``: shorter for network commands."""
    subcommand = next((a for a in args if not a.startswith("-")), "")
    if subcommand == "clone":
        return GIT_CLONE_TIMEOUT_SEC
    if subcommand in GIT_NETWORK_COMMANDS:
        return GIT_NETWORK_TIMEOUT_SEC
    return GIT_LOCAL_TIMEOUT_SEC


# Transport-level timeouts: detect a dead/stalled connection within ~1 minute
# instead of waiting for the hard subprocess timeout.
GIT_SSH_TIMEOUT_OPTIONS = "-o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
GIT_HTTP_LOW_SPEED = {"GIT_HTTP_LOW_SPEED_LIMIT": "1000", "GIT_HTTP_LOW_SPEED_TIME": "60"}


def _git_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for git subprocesses with SSH keepalives and an HTTP stall limit.

    - SSH (inspector VMs push via the deploy key): connect timeout and
      ServerAlive probes, so a silently dropped connection fails in ~60s.
      A custom GIT_SSH is left alone; an existing GIT_SSH_COMMAND gets the
      options appended unless it already sets ServerAliveInterval.
    - HTTPS (GitHub Actions with GITHUB_TOKEN): abort transfers slower than
      1 KB/s for 60s, unless GIT_HTTP_LOW_SPEED_* are already set.
    """
    env = dict(os.environ if base is None else base)
    ssh_command = env.get("GIT_SSH_COMMAND")
    if ssh_command:
        if "ServerAliveInterval" not in ssh_command:
            env["GIT_SSH_COMMAND"] = f"{ssh_command} {GIT_SSH_TIMEOUT_OPTIONS}"
    elif not env.get("GIT_SSH"):
        env["GIT_SSH_COMMAND"] = f"ssh {GIT_SSH_TIMEOUT_OPTIONS}"
    for key, value in GIT_HTTP_LOW_SPEED.items():
        env.setdefault(key, value)
    return env


_TOP_LEVEL_GIT_LOCKS = ("index.lock", "shallow.lock", "packed-refs.lock", "HEAD.lock", "config.lock", "FETCH_HEAD.lock")


def _git_dir(cwd) -> str | None:
    """The .git directory of the repo containing ``cwd`` (None if not found)."""
    path = os.path.abspath(cwd or os.getcwd())
    while True:
        candidate = os.path.join(path, ".git")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(path)
        if parent == path:
            return None
        path = parent


def _remove_locks_left_by_killed_git(cwd, started: float) -> None:
    """Delete *.lock files a killed git left behind (only ones created after ``started``).

    A git killed mid-write keeps holding index.lock / shallow.lock / ref locks, and
    every later command then fails with "Unable to create '...lock': File exists",
    which would turn one timeout into a permanently wedged repo. Locks older than
    the killed command belong to someone else and are left alone.
    """
    git_dir = _git_dir(cwd)
    if git_dir is None:
        return
    candidates = [os.path.join(git_dir, name) for name in _TOP_LEVEL_GIT_LOCKS]
    for root, _dirs, files in os.walk(os.path.join(git_dir, "refs")):
        candidates.extend(os.path.join(root, f) for f in files if f.endswith(".lock"))
    for lock in candidates:
        try:
            if os.path.getmtime(lock) >= started - 1:
                os.remove(lock)
                logging.warning("Removed git lock left by killed git process: %s", lock)
        except FileNotFoundError:
            continue
        except OSError as e:
            logging.warning("Could not remove git lock %s: %s", lock, e)


def _run_git(args: list[str], *, cwd, timeout: float, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run(["git", *args]) that turns a timeout into CalledProcessError.

    subprocess.run kills the child on timeout; raising CalledProcessError (not
    TimeoutExpired) lets the existing ``except CalledProcessError`` retry paths
    handle a hang exactly like a failed command.
    """
    started = time.time()
    kwargs.setdefault("env", _git_env())
    try:
        return subprocess.run(["git", *args], cwd=cwd, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as e:
        _remove_locks_left_by_killed_git(cwd, started)
        raise subprocess.CalledProcessError(
            returncode=-9,
            cmd=e.cmd,
            output=e.output,
            stderr=f"git timed out after {timeout:.0f}s and was killed",
        ) from e


def _small_memory_machine() -> bool:
    # Use total RAM: "available" often stays high on Linux due to cache.
    return psutil.virtual_memory().total < 4 * 1024 ** 3


def configure_git_low_memory() -> None:
    global _git_low_memory_configured
    if _git_low_memory_configured:
        return
    for command in _LOW_MEMORY_GIT_CONFIG:
        try:
            _run_git(command[1:], cwd=None, timeout=GIT_LOCAL_TIMEOUT_SEC, check=True)
        except subprocess.CalledProcessError as e:
            logging.warning(f"git low-memory config failed: {command}: {e}")
    _git_low_memory_configured = True


if _small_memory_machine():
    configure_git_low_memory()

def is_ssh_url(url: str) -> bool:
    """Check if the URL is an SSH URL (starts with git@ or ssh://)."""
    if not url:
        return False
    return url.startswith("git@") or url.startswith("ssh://")


def should_push() -> bool:
    """Check if we should push to git (either GITHUB_TOKEN is present or REPO_URL is SSH)."""
    if os.environ.get("GITHUB_TOKEN"):
        return True
    repo_url = os.environ.get("REPO_URL", REPO_URL)
    if is_ssh_url(repo_url):
        return True
    return False


def add_token_auth(url: str, token: str) -> str:
    parsed = urlparse(url)
    # works no matter if the original url had a user/pass or not
    domain = parsed.netloc.split("@")[-1]
    domain = f"user:{token}@{domain}"
    unparsed = (parsed[0], domain, parsed[2], parsed[3], parsed[4], parsed[5])
    return urlunparse(unparsed)


def _authenticated_repo_url(repo_url: str) -> str:
    if not is_ssh_url(repo_url):
        if token := os.environ.get("GITHUB_TOKEN"):
            logging.debug("Using GITHUB_TOKEN for authentication")
            return add_token_auth(repo_url, token)
        logging.debug("No GITHUB_TOKEN found, using URL as-is")
    else:
        logging.debug("Using SSH URL, no token auth needed")
    return repo_url


def _configure_origin_auth(repo_root: str) -> None:
    """Set origin URL with token auth so git CLI push/pull work outside checkout's credential helper."""
    repo_url = os.environ.get("REPO_URL", REPO_URL)
    if not repo_url:
        try:
            repo_url = run_git_command(["remote", "get-url", "origin"], cwd=repo_root).strip()
        except subprocess.CalledProcessError:
            return
    if is_ssh_url(repo_url) or not os.environ.get("GITHUB_TOKEN"):
        return
    auth_url = _authenticated_repo_url(repo_url)
    logging.debug("Configuring origin remote with GITHUB_TOKEN authentication")
    run_git_command(["remote", "set-url", "origin", auth_url], cwd=repo_root)


def _clone_repo(repo_url: str, repo_path: str, sparse_paths: tuple[str, ...] | None) -> None:
    """Shallow clone via git CLI; optional sparse checkout limits working tree size."""
    configure_git_low_memory()
    parent = os.path.dirname(repo_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    logging.info(f"Cloning from {repo_url} to {repo_path} (sparse={sparse_paths})")
    _run_git(
        [
            "clone", "--depth", "1", "--single-branch", "--branch", "main",
            "--no-checkout", repo_url, repo_path,
        ],
        cwd=None,
        timeout=GIT_CLONE_TIMEOUT_SEC,
        check=True,
        capture_output=True,
        text=True,
    )
    if sparse_paths:
        run_git_command(["sparse-checkout", "init", "--cone"], cwd=repo_path)
        run_git_command(["sparse-checkout", "set", *sparse_paths], cwd=repo_path)
    run_git_command(["checkout", "-f", "main"], cwd=repo_path)
    logging.info(f"Successfully cloned repo to {repo_path}")


def _repo_root(path: str | os.PathLike | None = None) -> str:
    if REPO_PATH and os.path.isdir(os.path.join(REPO_PATH, ".git")):
        return REPO_PATH
    if path is None:
        raise ValueError("REPO_PATH not set and no path given to locate git root")
    cwd = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    return run_git_command(["rev-parse", "--show-toplevel"], cwd=cwd).strip()


def path_has_changes(repo_root: str, path: str | os.PathLike) -> bool:
    """Check for changes under path without loading the full index via GitPython."""
    rel_path = os.path.relpath(path, repo_root)
    out = run_git_command(["status", "--porcelain", "--", rel_path], cwd=repo_root)
    return bool(out.strip())


def _is_shallow_repo(repo_root: str) -> bool:
    return os.path.isfile(os.path.join(repo_root, ".git", "shallow"))


def _origin_main_sha(repo_root: str) -> str:
    return run_git_command(["rev-parse", "origin/main"], cwd=repo_root).strip()


def _remote_main_sha(repo_root: str) -> str:
    out = run_git_command(["ls-remote", "--heads", "origin", "main"], cwd=repo_root).strip()
    if not out:
        raise RuntimeError("origin has no main branch")
    return out.split()[0]


def _read_sparse_paths(repo_root: str) -> tuple[str, ...] | None:
    try:
        out = run_git_command(["sparse-checkout", "list"], cwd=repo_root).strip()
    except subprocess.CalledProcessError:
        return None
    paths = tuple(line.strip() for line in out.splitlines() if line.strip())
    return paths or None


def _path_under_prefix(rel_path: str, prefix: str) -> bool:
    return rel_path == prefix or rel_path.startswith(prefix + "/")


def _assert_path_in_sparse(rel_path: str) -> None:
    if not _sparse_paths:
        return
    if any(_path_under_prefix(rel_path, prefix) for prefix in _sparse_paths):
        return
    logging.warning(
        "push_path %s is outside sparse checkout %s; concurrent pushes may conflict",
        rel_path,
        _sparse_paths,
    )


def _fetch_origin_main(repo_root: str) -> None:
    """Update origin/main to the remote tip without altering full-clone history."""
    fetch_args = ["fetch", "origin", "+refs/heads/main:refs/remotes/origin/main"]
    if _is_shallow_repo(repo_root):
        # Shallow clone: fetch latest tip only. Do NOT --deepen; that walks backward
        # through history and can leave origin/main far behind HEAD.
        fetch_args.append("--depth=1")
    run_git_command(fetch_args, cwd=repo_root)

    remote_sha = _remote_main_sha(repo_root)
    local_sha = _origin_main_sha(repo_root)
    if remote_sha == local_sha:
        return

    logging.warning(
        "origin/main stale after fetch (%s != %s), retrying fetch",
        local_sha[:8],
        remote_sha[:8],
    )
    retry_args = ["fetch", "origin", "+refs/heads/main:refs/remotes/origin/main"]
    if _is_shallow_repo(repo_root):
        retry_args.append("--depth=1")
    run_git_command(retry_args, cwd=repo_root)
    local_sha = _origin_main_sha(repo_root)
    if remote_sha != local_sha:
        raise RuntimeError(
            f"Could not sync origin/main to {remote_sha[:8]} (still at {local_sha[:8]})"
        )


def _changed_files_under(repo_root: str, rel_path: str) -> list[str]:
    """Return list of file paths changed between origin/main and HEAD under rel_path.

    Must be called BEFORE fetching so origin/main still points to our base.
    The returned list is stable across retries and safe to pass to
    ``_squash_commit_and_push`` even after origin/main advances.

    Includes deletions (e.g. transform.raw dropping a stale empty stdout).
    """
    out = run_git_command(
        ["diff", "--name-only", "origin/main", "HEAD", "--", rel_path],
        cwd=repo_root,
    )
    return [f.strip() for f in out.splitlines() if f.strip()]


def _path_in_commit(repo_root: str, commit: str, path: str) -> bool:
    """True if ``path`` exists as a blob in ``commit`` (not a deletion)."""
    return (
        _run_git(
            ["cat-file", "-e", f"{commit}:{path}"],
            cwd=repo_root,
            timeout=GIT_LOCAL_TIMEOUT_SEC,
            capture_output=True,
        ).returncode
        == 0
    )


def _squash_commit_and_push(
    repo_root: str,
    rel_path: str,
    msg: str,
    changed_files: list[str],
    saved_head: str,
) -> None:
    """
    Publish *only* ``changed_files`` as a single commit on origin/main.

    The caller captures ``changed_files`` and ``saved_head`` once (before
    fetching) so retries still restore from the inspector's result commit even
    after a failed attempt's ``reset --hard origin/main``.  Present files are
    checked out from ``saved_head``; paths deleted there (stale stdout/stderr)
    are ``git rm``'d after the reset — ``git checkout saved_head -- deleted``
    would pathspec-fail and wipe the working tree for good.
    """
    if not changed_files:
        logging.info("No changed files under %s to push", rel_path)
        return

    origin = "origin/main"
    present = [f for f in changed_files if _path_in_commit(repo_root, saved_head, f)]
    deleted = [f for f in changed_files if f not in present]
    logging.info(
        "Squashing %d files under %s onto %s from %s (%d present, %d deleted)",
        len(changed_files),
        rel_path,
        origin,
        saved_head[:8],
        len(present),
        len(deleted),
    )

    run_git_command(["reset", "--hard", origin], cwd=repo_root)
    if present:
        run_git_command(["checkout", saved_head, "--"] + present, cwd=repo_root)
        run_git_command(["add", "--"] + present, cwd=repo_root)
    # After reset, origin may still have paths we intentionally dropped.
    to_remove = [
        f for f in deleted if os.path.lexists(os.path.join(repo_root, f))
    ]
    if to_remove:
        run_git_command(["rm", "-f", "--"] + to_remove, cwd=repo_root)

    no_staged_changes = _run_git(
        ["diff", "--cached", "--quiet"],
        cwd=repo_root,
        timeout=GIT_LOCAL_TIMEOUT_SEC,
        capture_output=True,
    ).returncode == 0
    if no_staged_changes:
        logging.info("Already up to date with %s for %s after squash", origin, rel_path)
        return

    run_git_command(["commit", "-m", msg], cwd=repo_root)
    _push_origin_main(repo_root)


def _push_origin_main(repo_root: str) -> None:
    run_git_command(["push", "origin", "HEAD:main"], cwd=repo_root)


@synchronized
def get_repo(repo_url=REPO_URL, repo_path=REPO_PATH, sparse_paths: tuple[str, ...] | None = None):
    """
    Return git.Repo.

    If there"s an already existing repo, use that, otherwise do a shallow clone.
    Pass sparse_paths (e.g. ("data/aws/t3.micro",)) to checkout only that subtree.
    """
    global _sparse_paths
    logging.info(f"Getting repo from {repo_path}, URL: {repo_url}, sparse={sparse_paths}")
    repo_url = _authenticated_repo_url(repo_url)
    try:
        logging.info(f"Attempting to open existing repo at {repo_path}")
        repo = git.Repo(repo_path)
        _configure_origin_auth(repo_path)
        if sparse_paths:
            _sparse_paths = sparse_paths
        elif _sparse_paths is None:
            _sparse_paths = _read_sparse_paths(repo_path)
        logging.info(f"Successfully opened existing repo at {repo_path}")
        return repo
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        logging.info(f"Repo not found at {repo_path}, cloning from {repo_url}")
        try:
            _clone_repo(repo_url, repo_path, sparse_paths)
            _sparse_paths = sparse_paths
            return git.Repo(repo_path)
        except Exception as clone_error:
            logging.error(f"Failed to clone repo from {repo_url} to {repo_path}: {clone_error}")
            raise


def run_git_command(args, cwd, timeout: float | None = None):
    """Helper function to run git commands using subprocess.

    Every call has a hard timeout (see ``git_timeout``): a git process that hangs
    (e.g. a stalled fetch/push to GitHub) is killed and reported as a failed
    command, so callers' retry logic runs instead of blocking forever.
    """
    command_str = ' '.join(['git'] + args)
    timeout = git_timeout(args) if timeout is None else timeout
    logging.info(f"Running git command: {command_str} (cwd: {cwd})")
    try:
        result = _run_git(args, cwd=cwd, timeout=timeout, capture_output=True, text=True, check=True)
        if result.stdout:
            logging.debug(f"Git command stdout: {result.stdout}")
        logging.info(f"Git command succeeded: {command_str}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Git command failed: {command_str}\n"
        error_msg += f"Exit code: {e.returncode}\n"
        if e.stdout:
            error_msg += f"Stdout: {e.stdout}\n"
        if e.stderr:
            error_msg += f"Stderr: {e.stderr}\n"
        logging.error(error_msg)
        raise

@synchronized
def push_paths(paths: list[str | os.PathLike], msg: str):
    """Commit and squash-push every dirty path in one round.

    Callers that update both a variant root and ``schedule/`` must use this (or
    a single ``push_path`` covering both) rather than sequential ``push_path``
    calls: each squash does ``reset --hard origin/main``, which would wipe
    sibling dirty trees that were not part of the first commit.
    """
    path_list = [os.path.abspath(str(p)) for p in paths]
    logging.info(f"push_paths called for paths: {path_list}, message: {msg}")
    if not path_list:
        return
    if not should_push():
        has_token = bool(os.environ.get("GITHUB_TOKEN"))
        repo_url = os.environ.get("REPO_URL", REPO_URL)
        is_ssh = is_ssh_url(repo_url)
        logging.warning(
            f"Skipping push: no authentication available "
            f"(GITHUB_TOKEN={has_token}, SSH_URL={is_ssh}, REPO_URL={repo_url})"
        )
        return
    repo_root = _repo_root(path_list[0])
    logging.info(f"Using repo root: {repo_root}")
    try:
        # Pass repo_path explicitly: get_repo's default is bound at import time.
        get_repo(repo_path=repo_root)
        dirty = [p for p in dict.fromkeys(path_list) if path_has_changes(repo_root, p)]
        if not dirty:
            logging.info("No changes detected, skipping git operations")
            return
        rel_paths = [os.path.relpath(p, repo_root) for p in dirty]
        for rel_path in rel_paths:
            _assert_path_in_sparse(rel_path)
        logging.info("Staging changes under %s...", rel_paths)
        run_git_command(["add", "--"] + rel_paths, cwd=repo_root)
        logging.info("Committing changes...")
        run_git_command(["commit", "-m", msg], cwd=repo_root)
        # Snapshot the exact files + result commit BEFORE fetching/reset.
        # origin/main still points to our base, so this diff is stable.
        # saved_head must outlive retries: a failed squash does reset --hard
        # to origin/main, and re-reading HEAD then would lose the results.
        changed_files: list[str] = []
        for rel_path in rel_paths:
            changed_files.extend(_changed_files_under(repo_root, rel_path))
        changed_files = list(dict.fromkeys(changed_files))
        if not changed_files:
            logging.info("Commit created but no file-level diff vs origin/main")
            return
        saved_head = run_git_command(["rev-parse", "HEAD"], cwd=repo_root).strip()
        rel_label = ",".join(rel_paths)
        # Retry push with exponential backoff (many inspectors push concurrently).
        # With git timeouts every attempt is bounded, so the deadline is reached
        # even when git hangs instead of failing.
        deadline = time.monotonic() + PUSH_DEADLINE_SEC
        wait_sec = PUSH_INITIAL_BACKOFF_SEC
        max_wait_per_round = PUSH_MAX_BACKOFF_SEC
        attempt = 0
        while True:
            try:
                if attempt > 0:
                    sleep_sec = min(wait_sec, max_wait_per_round) + random.uniform(0, 2)
                    logging.info(
                        f"Retry attempt {attempt}: waiting {sleep_sec:.1f}s then fetch/squash/push"
                    )
                    time.sleep(sleep_sec)
                    wait_sec = min(wait_sec * 2, max_wait_per_round)
                logging.info("Fetching origin/main and squashing push...")
                _fetch_origin_main(repo_root)
                _squash_commit_and_push(
                    repo_root, rel_label, msg, changed_files, saved_head
                )
                logging.info(f"Successfully pushed changes to git: {msg}")
                break
            except subprocess.CalledProcessError as e:
                attempt += 1
                if time.monotonic() >= deadline:
                    logging.error(f"Push failed after retries ({PUSH_DEADLINE_SEC:.0f}s deadline)")
                    raise
                logging.warning(f"Push failed (attempt {attempt}), will retry with backoff: {e}")
    except Exception as e:
        logging.error(f"Failed to push paths {path_list}: {e}", exc_info=True)
        raise


def push_path(path: str | os.PathLike, msg: str):
    """Push a single path; see ``push_paths`` for multi-root updates."""
    push_paths([path], msg)


@synchronized
def pull():
    logging.info("Pulling latest changes from remote...")
    repo_root = _repo_root()
    try:
        get_repo()
        run_git_command(["fetch", "origin"], cwd=repo_root)
        logging.info("Merging origin/main with 'theirs' strategy...")
        run_git_command(["merge", "-X", "theirs", "origin/main"], cwd=repo_root)
        logging.info("Pull completed successfully")
    except Exception as e:
        logging.error(f"Failed to pull from remote: {e}", exc_info=True)
        raise


def gha_url():
    """Return GHA run URL."""
    url = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    return f"{url}/{repo}/actions/runs/{run_id}"
