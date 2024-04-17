import git
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

REPO_PATH = "/repo/sc-inspector-data"
repo = git.Repo(REPO_PATH)
with open(f"{REPO_PATH}/data/test", "w") as f:
    f.write("test")
repo.index.add("data/test")
repo.index.commit("test commit")
origin = repo.remote(name="origin")
origin.push()
