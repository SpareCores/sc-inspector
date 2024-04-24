import os
import git
import functools

REPO_URL = os.environ.get("REPO_URL")
REPO_PATH = os.environ.get("REPO_PATH")


@functools.cache
def get_repo():
    """
    Return git.Repo.

    If there's an already existing repo, use that, otherwise do a clone.
    """
    try:
        repo = git.Repo(REPO_PATH)
    except git.InvalidGitRepositoryError:
        repo = git.Repo.clone_from(REPO_URL, REPO_PATH)
    return repo


def push_path(path: str | os.PathLike, msg: str):
    repo = get_repo()
    repo.index.add(path)
    repo.index.commit(msg)
    origin = repo.remote(name="origin")
    origin.pull()
    origin.push()
