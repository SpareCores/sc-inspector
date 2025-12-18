import functools
import git
import logging
import os
import psutil
import subprocess
from urllib.parse import urlparse, urlunparse
from wrapt import synchronized


# pre-set in the ghcr.io/sparecores/sc-inspector:main image
REPO_URL = os.environ.get("REPO_URL")
REPO_PATH = os.environ.get("REPO_PATH")

commands = [
    ["git", "config", "--global", "core.bigFileThreshold", "1"],
    ["git", "config", "--global", "core.deltaBaseCacheLimit", "0"],
    ["git", "config", "--global", "gc.auto", "0"],
    ["git", "config", "--global", "pack.deltaCacheLimit", "0"],
    ["git", "config", "--global", "pack.deltaCacheSize", "1"],
    ["git", "config", "--global", "pack.threads", "1"],
    ["git", "config", "--global", "pack.windowMemory", "10m"],
    ["git", "config", "--global", "checkout.thresholdForParallelism", "99999999"],
    ["git", "config", "--global", "core.compression", "0"],
    ["git", "config", "--global", "index.threads", "1"]
]

if psutil.virtual_memory().available < 1024 ** 3:
    # try to reduce git's memory usage on small-mem machines
    for command in commands:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {e}")

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


@functools.cache
@synchronized
def get_repo(repo_url=REPO_URL, repo_path=REPO_PATH):
    """
    Return git.Repo.

    If there"s an already existing repo, use that, otherwise do a clone.
    """
    logging.info(f"Getting repo from {repo_path}, URL: {repo_url}")
    # Only use token auth for HTTPS URLs, not SSH URLs
    if not is_ssh_url(repo_url):
        if token := os.environ.get("GITHUB_TOKEN"):
            logging.debug("Using GITHUB_TOKEN for authentication")
            repo_url = add_token_auth(repo_url, token)
        else:
            logging.debug("No GITHUB_TOKEN found, using URL as-is")
    else:
        logging.debug("Using SSH URL, no token auth needed")
    try:
        logging.info(f"Attempting to open existing repo at {repo_path}")
        repo = git.Repo(repo_path)
        logging.info(f"Successfully opened existing repo at {repo_path}")
        return repo
    except (git.InvalidGitRepositoryError, git.NoSuchPathError) as e:
        logging.info(f"Repo not found at {repo_path}, cloning from {repo_url}")
        try:
            repo = git.Repo.clone_from(repo_url, repo_path, depth=1, branch="main", single_branch=True)
            logging.info(f"Successfully cloned repo to {repo_path}")
            return repo
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
    repo_path = os.path.dirname(os.path.abspath(path))
    logging.info(f"Using repo_path: {repo_path}")
    try:
        repo = get_repo()
        changes = repo.untracked_files + repo.index.diff(None)
        logging.info(f"Found {len(changes)} changes: {len(repo.untracked_files)} untracked files, {len(repo.index.diff(None))} modified files")
        if changes:
            logging.info(f"Untracked files: {repo.untracked_files}")
            logging.info(f"Modified files: {[d.a_path for d in repo.index.diff(None)]}")
            # use git command instead of gitpython as the latter requires a lot of
            # memory
            logging.info("Staging changes...")
            run_git_command(['add', path], cwd=repo_path)
            logging.info("Committing changes...")
            run_git_command(['commit', '-m', msg], cwd=repo_path)
            logging.info("Pulling with rebase...")
            run_git_command(['pull', '--rebase', 'origin'], cwd=repo_path)
            logging.info("Pushing to origin...")
            run_git_command(['push', 'origin'], cwd=repo_path)
            logging.info(f"Successfully pushed changes to git: {msg}")
        else:
            logging.info("No changes detected, skipping git operations")
    except Exception as e:
        logging.error(f"Failed to push path {path}: {e}", exc_info=True)
        raise


@synchronized
def pull():
    logging.info("Pulling latest changes from remote...")
    try:
        repo = get_repo()
        origin = repo.remotes.origin
        logging.info(f"Fetching from origin: {origin.url}")
        origin.fetch()
        logging.info("Fetch completed successfully")
        logging.info("Merging origin/main with 'theirs' strategy...")
        repo.git.merge("-X", "theirs", "origin/main")
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
