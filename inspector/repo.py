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


def _small_memory_machine() -> bool:
    # Use total RAM: "available" often stays high on Linux due to cache.
    return psutil.virtual_memory().total < 4 * 1024 ** 3


def configure_git_low_memory() -> None:
    global _git_low_memory_configured
    if _git_low_memory_configured:
        return
    for command in _LOW_MEMORY_GIT_CONFIG:
        try:
            subprocess.run(command, check=True)
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
    subprocess.run(
        [
            "git", "clone", "--depth", "1", "--single-branch", "--branch", "main",
            "--no-checkout", repo_url, repo_path,
        ],
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
    """
    out = run_git_command(
        ["diff", "--name-only", "origin/main", "HEAD", "--", rel_path],
        cwd=repo_root,
    )
    return [f.strip() for f in out.splitlines() if f.strip()]


def _squash_commit_and_push(
    repo_root: str, rel_path: str, msg: str, changed_files: list[str],
) -> None:
    """
    Publish *only* ``changed_files`` as a single commit on origin/main.

    The caller captures ``changed_files`` once (before fetching) so the list
    reflects exactly the files this inspector modified, regardless of how
    origin/main moves between retries.  Only those files are checked out from
    ``saved_head`` — the rest of the tree (including paths written by other
    concurrent inspectors) is left at origin/main's version.
    """
    if not changed_files:
        logging.info("No changed files under %s to push", rel_path)
        return

    origin = "origin/main"
    saved_head = run_git_command(["rev-parse", "HEAD"], cwd=repo_root).strip()
    logging.info(
        "Squashing %d files under %s onto %s from %s",
        len(changed_files),
        rel_path,
        origin,
        saved_head[:8],
    )

    run_git_command(["reset", "--hard", origin], cwd=repo_root)
    run_git_command(["checkout", saved_head, "--"] + changed_files, cwd=repo_root)
    run_git_command(["add", "--"] + changed_files, cwd=repo_root)

    no_staged_changes = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo_root,
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


def run_git_command(args, cwd):
    """Helper function to run git commands using subprocess."""
    command_str = ' '.join(['git'] + args)
    logging.info(f"Running git command: {command_str} (cwd: {cwd})")
    try:
        result = subprocess.run(
            ['git'] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
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
def push_path(path: str | os.PathLike, msg: str):
    logging.info(f"push_path called for path: {path}, message: {msg}")
    # Only push if we have authentication (GITHUB_TOKEN or SSH URL)
    if not should_push():
        has_token = bool(os.environ.get("GITHUB_TOKEN"))
        repo_url = os.environ.get("REPO_URL", REPO_URL)
        is_ssh = is_ssh_url(repo_url)
        logging.warning(f"Skipping push: no authentication available (GITHUB_TOKEN={has_token}, SSH_URL={is_ssh}, REPO_URL={repo_url})")
        return
    repo_root = _repo_root(path)
    logging.info(f"Using repo root: {repo_root}")
    try:
        get_repo()
        if path_has_changes(repo_root, path):
            rel_path = os.path.relpath(path, repo_root)
            _assert_path_in_sparse(rel_path)
            logging.info("Staging changes...")
            run_git_command(["add", rel_path], cwd=repo_root)
            logging.info("Committing changes...")
            run_git_command(["commit", "-m", msg], cwd=repo_root)
            # Snapshot the exact files we changed BEFORE fetching.
            # origin/main still points to our base, so this diff is stable.
            changed_files = _changed_files_under(repo_root, rel_path)
            if not changed_files:
                logging.info("Commit created but no file-level diff vs origin/main")
                return
            # Retry push with exponential backoff (many inspectors push concurrently)
            deadline = time.monotonic() + 10 * 60  # 10 minutes total
            wait_sec = 5  # initial backoff in seconds
            max_wait_per_round = 2 * 60  # cap 2 minutes per round
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
                    _squash_commit_and_push(repo_root, rel_path, msg, changed_files)
                    logging.info(f"Successfully pushed changes to git: {msg}")
                    break
                except subprocess.CalledProcessError as e:
                    attempt += 1
                    if time.monotonic() >= deadline:
                        logging.error("Push failed after retries (10 min deadline)")
                        raise
                    logging.warning(f"Push failed (attempt {attempt}), will retry with backoff: {e}")
        else:
            logging.info("No changes detected, skipping git operations")
    except Exception as e:
        logging.error(f"Failed to push path {path}: {e}", exc_info=True)
        raise


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
