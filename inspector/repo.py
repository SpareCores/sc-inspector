import os
import git
import functools
from urllib.parse import urlparse, urlunparse
from wrapt import synchronized


# pre-set in the ghcr.io/sparecores/sc-inspector:main image
REPO_URL = os.environ.get("REPO_URL")
REPO_PATH = os.environ.get("REPO_PATH")


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

    If there's an already existing repo, use that, otherwise do a clone.
    """
    if token := os.environ.get("GITHUB_TOKEN"):
        repo_url = add_token_auth(repo_url, token)
    try:
        return git.Repo(repo_path)
    except (git.InvalidGitRepositoryError, git.NoSuchPathError):
        return git.Repo.clone_from(repo_url, repo_path)


@synchronized
def push_path(path: str | os.PathLike, msg: str):
    repo = get_repo()
    changes = repo.untracked_files + repo.index.diff(None)
    if changes:
        repo.index.add(path)
        repo.index.commit(msg)
        origin = repo.remote(name="origin")
        origin.pull(strategy_option="ours")
        origin.push()


def gha_url():
    """Return GHA run URL."""
    url = os.environ.get("GITHUB_SERVER_URL")
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    return f"{url}/{repo}/actions/runs/{run_id}"
