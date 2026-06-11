import functools
import git
import logging
import os
import psutil
import subprocess
import time
from urllib.parse import urlparse, urlunparse
from wrapt import synchronized


# pre-set in the ghcr.io/sparecores/sc-inspector:main image
REPO_URL = os.environ.get("REPO_URL")
REPO_PATH = os.environ.get("REPO_PATH")

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
    out = run_git_command(["status", "--porcelain", "--", path], cwd=repo_root)
    return bool(out.strip())


@functools.cache
@synchronized
def get_repo(repo_url=REPO_URL, repo_path=REPO_PATH, sparse_paths: tuple[str, ...] | None = None):
    """
    Return git.Repo.

    If there"s an already existing repo, use that, otherwise do a shallow clone.
    Pass sparse_paths (e.g. ("data/aws/t3.micro",)) to checkout only that subtree.
    """
    logging.info(f"Getting repo from {repo_path}, URL: {repo_url}, sparse={sparse_paths}")
    repo_url = _authenticated_repo_url(repo_url)
    try:
        logging.info(f"Attempting to open existing repo at {repo_path}")
        repo = git.Repo(repo_path)
        _configure_origin_auth(repo_path)
        logging.info(f"Successfully opened existing repo at {repo_path}")
        return repo
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        logging.info(f"Repo not found at {repo_path}, cloning from {repo_url}")
        try:
            _clone_repo(repo_url, repo_path, sparse_paths)
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
            logging.info("Staging changes...")
            run_git_command(['add', path], cwd=repo_root)
            logging.info("Committing changes...")
            run_git_command(['commit', '-m', msg], cwd=repo_root)
            # Retry push with exponential backoff (remote ref may have moved; pull --rebase and retry)
            deadline = time.monotonic() + 10 * 60  # 10 minutes total
            wait_sec = 5  # initial backoff in seconds
            max_wait_per_round = 2 * 60  # cap 2 minutes per round
            attempt = 0
            while True:
                try:
                    if attempt > 0:
                        sleep_sec = min(wait_sec, max_wait_per_round)
                        logging.info(f"Retry attempt {attempt}: waiting {sleep_sec}s then pull --rebase and push")
                        time.sleep(sleep_sec)
                        wait_sec = min(wait_sec * 2, max_wait_per_round)
                    logging.info("Pulling with rebase...")
                    run_git_command(['pull', '--rebase', 'origin'], cwd=repo_root)
                    logging.info("Pushing to origin...")
                    run_git_command(['push', 'origin'], cwd=repo_root)
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
