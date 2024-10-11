import functools
import git
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

def add_token_auth(url: str, token: str) -> str:
    parsed = urlparse(url)
    # works no matter if the original url had a user/pass or not
    domain = parsed.netloc.split("@")[-1]
    domain = f"{token}@{domain}"
    unparsed = (parsed[0], domain, parsed[2], parsed[3], parsed[4], parsed[5])
    return urlunparse(unparsed)


@functools.cache
@synchronized
def get_repo(repo_url=REPO_URL, repo_path=REPO_PATH):
    """
    Return git.Repo.

    If there"s an already existing repo, use that, otherwise do a clone.
    """
    if token := os.environ.get("GITHUB_TOKEN"):
        repo_url = add_token_auth(repo_url, token)
    try:
        return git.Repo(repo_path)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return git.Repo.clone_from(repo_url, repo_path, depth=1, branch="main", single_branch=True)


def run_git_command(args, cwd):
    """Helper function to run git commands using subprocess."""
    result = subprocess.run(
        ['git'] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout

@synchronized
def push_path(path: str | os.PathLike, msg: str):
    repo_path = os.path.dirname(os.path.abspath(path))
    repo = get_repo()
    changes = repo.untracked_files + repo.index.diff(None)
    if changes:
        # use git command instead of gitpython as the latter requires a lot of
        # memory
        run_git_command(['add', path], cwd=repo_path)
        run_git_command(['commit', '-m', msg], cwd=repo_path)
        run_git_command(['pull', '--rebase', 'origin'], cwd=repo_path)
        run_git_command(['push', 'origin'], cwd=repo_path)


@synchronized
def pull():
    repo = get_repo()
    origin = repo.remotes.origin
    origin.fetch()
    repo.git.merge("-X", "theirs", "origin/main")


def gha_url():
    """Return GHA run URL."""
    url = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    return f"{url}/{repo}/actions/runs/{run_id}"
