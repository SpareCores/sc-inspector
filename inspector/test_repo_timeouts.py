"""Git helpers pass subprocess timeout and classify network vs local cmds."""
from __future__ import annotations

import subprocess
from unittest import mock

import pytest

import repo


def test_git_timeout_for_network_vs_local():
    assert repo._git_timeout_for(["fetch", "origin"]) == repo.GIT_NETWORK_TIMEOUT_SEC
    assert repo._git_timeout_for(["push", "origin", "HEAD:main"]) == repo.GIT_NETWORK_TIMEOUT_SEC
    assert repo._git_timeout_for(["status", "--porcelain"]) == repo.GIT_CMD_TIMEOUT_SEC
    assert repo._git_timeout_for(["rev-parse", "HEAD"]) == repo.GIT_CMD_TIMEOUT_SEC


def test_run_git_command_passes_timeout():
    completed = subprocess.CompletedProcess(
        args=["git", "status"], returncode=0, stdout="", stderr=""
    )
    with mock.patch("repo.subprocess.run", return_value=completed) as run:
        repo.run_git_command(["status", "--porcelain"], cwd="/tmp")
    assert run.call_args.kwargs["timeout"] == repo.GIT_CMD_TIMEOUT_SEC


def test_run_git_command_network_timeout():
    completed = subprocess.CompletedProcess(
        args=["git", "fetch"], returncode=0, stdout="", stderr=""
    )
    with mock.patch("repo.subprocess.run", return_value=completed) as run:
        repo.run_git_command(["fetch", "origin"], cwd="/tmp")
    assert run.call_args.kwargs["timeout"] == repo.GIT_NETWORK_TIMEOUT_SEC


def test_run_git_command_timeout_expired_propagates():
    with mock.patch(
        "repo.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["git", "fetch"], timeout=1),
    ):
        with pytest.raises(subprocess.TimeoutExpired):
            repo.run_git_command(["fetch", "origin"], cwd="/tmp")
