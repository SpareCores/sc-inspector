"""Empty-stack pending CREATE must be clearable so cleanup can finish the batch.

Regression for azure.spaincentral...d2sv3-pg18: destroy emptied confirmed
resources but left an interrupted ResourceGroup CREATE; older sc-runner then
refused remove_stack and absolute-timeout cleanup failed the whole job.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import inspector as insp
import pytest


def test_clear_pending_ops_removes_empty_stack():
    deployment = {
        "resources": [{"urn": "urn:pulumi:x::y::pulumi:pulumi:Stack::s", "type": "pulumi:pulumi:Stack"}],
        "pending_operations": [
            {
                "type": "creating",
                "resource": {
                    "urn": "urn:pulumi:x::y::azure-native:resources:ResourceGroup::rg",
                    "type": "azure-native:resources:ResourceGroup",
                    "custom": True,
                    "inputs": {"resourceGroupName": "spaincentrald2sv3-pg18"},
                },
            }
        ],
    }
    imported: dict = {}
    removed: list[tuple] = []

    stack = SimpleNamespace(
        name="azure.spaincentral.None.Standard_D4lds_v6.d2sv3-pg18",
        export_stack=lambda: SimpleNamespace(version=3, deployment=deployment),
        import_stack=lambda dep: imported.update(version=dep.version, deployment=dep.deployment),
        workspace=SimpleNamespace(
            remove_stack=lambda name, force=False: removed.append((name, force))
        ),
    )

    insp._clear_pending_ops_and_remove_stack(stack, stack.name)

    assert "pending_operations" not in imported["deployment"]
    assert "pendingOperations" not in imported["deployment"]
    assert imported["version"] == 3
    assert removed == [(stack.name, True)]


def test_clear_pending_ops_refuses_when_custom_resources_remain():
    deployment = {
        "resources": [
            {"urn": "urn:pulumi:x::y::pulumi:pulumi:Stack::s", "type": "pulumi:pulumi:Stack"},
            {
                "urn": "urn:pulumi:x::y::azure-native:compute:VirtualMachine::vm",
                "type": "azure-native:compute:VirtualMachine",
                "custom": True,
            },
        ],
        "pending_operations": [{"type": "creating", "resource": {}}],
    }
    stack = SimpleNamespace(
        name="stack",
        export_stack=lambda: SimpleNamespace(version=1, deployment=deployment),
        import_stack=mock.Mock(),
        workspace=SimpleNamespace(remove_stack=mock.Mock()),
    )

    with pytest.raises(RuntimeError, match="confirmed custom resource"):
        insp._clear_pending_ops_and_remove_stack(stack, stack.name)

    stack.import_stack.assert_not_called()
    stack.workspace.remove_stack.assert_not_called()


def test_clear_pending_ops_still_removes_when_already_clean():
    deployment = {
        "resources": [{"urn": "urn:pulumi:x::y::pulumi:pulumi:Stack::s", "type": "pulumi:pulumi:Stack"}],
    }
    removed: list[tuple] = []
    stack = SimpleNamespace(
        name="stack",
        export_stack=lambda: SimpleNamespace(version=1, deployment=deployment),
        import_stack=mock.Mock(),
        workspace=SimpleNamespace(
            remove_stack=lambda name, force=False: removed.append((name, force))
        ),
    )

    insp._clear_pending_ops_and_remove_stack(stack, stack.name)

    stack.import_stack.assert_not_called()
    assert removed == [("stack", True)]
