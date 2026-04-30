from __future__ import annotations

import asyncio
import importlib
import io
import re
import shlex
import sys
import tarfile
import time
import types
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from agents.sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
)
from agents.sandbox.manifest import Manifest
from agents.sandbox.snapshot import NoopSnapshot
from tests._fake_workspace_paths import resolve_fake_workspace_path

# ---------------------------------------------------------------------------
# Fake azure SDK stubs (injected into sys.modules so the lazy import helpers work)
# ---------------------------------------------------------------------------


class _FakeResourceNotFoundError(Exception):
    """Stand-in for ``azure.core.exceptions.ResourceNotFoundError``."""


_ENV_PREFIX_RE = re.compile(r"^env\s+(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+sh\s+-c\s+(.*)$", re.DOTALL)


def _maybe_unwrap_env(command: str) -> str:
    """Return the inner command if ``command`` is wrapped by ``env K=V ... sh -c '<inner>'``."""
    match = _ENV_PREFIX_RE.match(command)
    if not match:
        return command
    quoted_inner = match.group(1)
    try:
        tokens = shlex.split(quoted_inner)
    except ValueError:
        return command
    if not tokens:
        return command
    return tokens[0]


class _FakeSandboxClient:
    """Stand-in for ``azure.sandbox.SandboxClient`` (data plane)."""

    def __init__(
        self,
        resource_group: str | None = None,
        subscription_id: str | None = None,
        *,
        credential: Any = None,
    ) -> None:
        self.resource_group = resource_group
        self.subscription_id = subscription_id
        self.credential = credential
        self.closed = False

        self.exec_calls: list[dict[str, Any]] = []
        self.read_file_calls: list[tuple[str, str, str]] = []
        self.write_file_calls: list[tuple[str, str, str, bytes, bool]] = []
        self.mkdir_calls: list[tuple[str, str, str]] = []
        self.get_sandbox_calls: list[tuple[str, str]] = []
        self.stop_sandbox_calls: list[tuple[str, str]] = []
        self.delete_sandbox_calls: list[tuple[str, str]] = []
        self.resume_sandbox_calls: list[tuple[str, str]] = []
        self.create_sandbox_calls: list[dict[str, Any]] = []

        self.next_exec_result: dict[str, Any] = {"exitCode": 0, "stdout": "", "stderr": ""}
        self.next_get_sandbox: dict[str, Any] = {
            "id": "sb-123",
            "state": "Running",
            "ports": [],
        }
        self.next_read_file: bytes = b"file-content"
        self.next_create_sandbox: dict[str, Any] = {"id": "sb-new-456"}

        self.exec_delay_s: float = 0.0
        self.exec_error: BaseException | None = None
        self.get_sandbox_error: BaseException | None = None
        self.read_file_error: BaseException | None = None
        self.write_file_error: BaseException | None = None
        self.mkdir_error: BaseException | None = None
        self.stop_sandbox_error: BaseException | None = None

    def exec(
        self,
        sandbox_id: str,
        sandbox_group: str,
        command: str,
        working_directory: str | None = None,
    ) -> dict[str, Any]:
        self.exec_calls.append(
            {
                "sandbox_id": sandbox_id,
                "sandbox_group": sandbox_group,
                "command": command,
                "working_directory": working_directory,
            }
        )

        inner = _maybe_unwrap_env(command)
        try:
            parts = shlex.split(inner)
        except ValueError:
            parts = []

        # Helper install/probe commands all live under ``/tmp/openai-agents/``.
        if parts and parts[0] == "mkdir" and "-p" in parts:
            target = parts[-1]
            if target.startswith("/tmp/openai-agents/"):
                return {"exitCode": 0, "stdout": "", "stderr": ""}
        if parts and parts[0] == "test" and "-x" in parts:
            return {"exitCode": 0, "stdout": "", "stderr": ""}
        # The runtime helper install command is a multi-line shell script that contains the
        # ``INSTALL_RUNTIME_HELPER_V1`` marker; treat all install scripts as successful.
        if "INSTALL_RUNTIME_HELPER_V1" in inner:
            return {"exitCode": 0, "stdout": "", "stderr": ""}

        resolved = resolve_fake_workspace_path(
            inner,
            symlinks={},
            home_dir="/home/user/workspace",
        )
        if resolved is not None:
            return {
                "exitCode": resolved.exit_code,
                "stdout": resolved.stdout,
                "stderr": resolved.stderr,
            }

        if self.exec_delay_s > 0:
            time.sleep(self.exec_delay_s)
        if self.exec_error is not None:
            raise self.exec_error
        result = self.next_exec_result
        self.next_exec_result = {"exitCode": 0, "stdout": "", "stderr": ""}
        return result

    def read_file(
        self,
        sandbox_id: str,
        sandbox_group: str,
        path: str,
    ) -> bytes:
        self.read_file_calls.append((sandbox_id, sandbox_group, path))
        if self.read_file_error is not None:
            raise self.read_file_error
        return self.next_read_file

    def write_file(
        self,
        sandbox_id: str,
        sandbox_group: str,
        path: str,
        content: bytes,
        create_dirs: bool = True,
    ) -> None:
        self.write_file_calls.append((sandbox_id, sandbox_group, path, content, create_dirs))
        if self.write_file_error is not None:
            raise self.write_file_error

    def mkdir(self, sandbox_id: str, sandbox_group: str, path: str) -> None:
        self.mkdir_calls.append((sandbox_id, sandbox_group, path))
        if self.mkdir_error is not None:
            raise self.mkdir_error

    def get_sandbox(self, sandbox_id: str, sandbox_group: str) -> dict[str, Any]:
        self.get_sandbox_calls.append((sandbox_id, sandbox_group))
        if self.get_sandbox_error is not None:
            raise self.get_sandbox_error
        return self.next_get_sandbox

    def stop_sandbox(self, sandbox_id: str, sandbox_group: str) -> None:
        self.stop_sandbox_calls.append((sandbox_id, sandbox_group))
        if self.stop_sandbox_error is not None:
            raise self.stop_sandbox_error

    def delete_sandbox(self, sandbox_id: str, sandbox_group: str) -> None:
        self.delete_sandbox_calls.append((sandbox_id, sandbox_group))

    def resume_sandbox(self, sandbox_id: str, sandbox_group: str) -> None:
        self.resume_sandbox_calls.append((sandbox_id, sandbox_group))

    def create_sandbox(self, sandbox_group: str, **kwargs: Any) -> dict[str, Any]:
        recorded = dict(kwargs)
        recorded["sandbox_group"] = sandbox_group
        self.create_sandbox_calls.append(recorded)
        return self.next_create_sandbox

    def close(self) -> None:
        self.closed = True


class _FakeMgmtClient:
    """Stand-in for ``azure.mgmt.sandbox.SandboxGroupManagementClient`` (control plane)."""

    def __init__(
        self,
        resource_group: str | None = None,
        subscription_id: str | None = None,
        *,
        credential: Any = None,
    ) -> None:
        self.resource_group = resource_group
        self.subscription_id = subscription_id
        self.credential = credential
        self.closed = False

        self.list_groups_calls: list[Any] = []
        self.get_group_calls: list[str] = []
        self.create_group_calls: list[dict[str, Any]] = []
        self.delete_group_calls: list[str] = []

        self.next_list_groups: list[dict[str, Any]] = []
        self.next_get_group: dict[str, Any] = {"name": "sg-test", "location": "westus2"}
        self.next_create_group: dict[str, Any] = {"name": "sg-created", "location": "westus2"}
        self.get_group_error: BaseException | None = None

    def list_groups(self) -> list[dict[str, Any]]:
        self.list_groups_calls.append(None)
        return list(self.next_list_groups)

    def get_group(self, name: str) -> dict[str, Any]:
        self.get_group_calls.append(name)
        if self.get_group_error is not None:
            raise self.get_group_error
        return self.next_get_group

    def create_group(
        self,
        name: str,
        location: str,
        identity: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        **properties: Any,
    ) -> dict[str, Any]:
        self.create_group_calls.append(
            {
                "name": name,
                "location": location,
                "identity": identity,
                "tags": tags,
                "properties": properties,
            }
        )
        return self.next_create_group

    def delete_group(self, name: str) -> None:
        self.delete_group_calls.append(name)

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Module-level fixtures — inject fake SDK into sys.modules
# ---------------------------------------------------------------------------


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Insert fake ``azure.sandbox``, ``azure.mgmt.sandbox`` and ``azure.core.exceptions``."""

    fake_azure: Any = types.ModuleType("azure")
    fake_azure_sandbox: Any = types.ModuleType("azure.sandbox")
    fake_azure_sandbox.SandboxClient = _FakeSandboxClient

    fake_azure_mgmt: Any = types.ModuleType("azure.mgmt")
    fake_azure_mgmt_sandbox: Any = types.ModuleType("azure.mgmt.sandbox")
    fake_azure_mgmt_sandbox.SandboxGroupManagementClient = _FakeMgmtClient

    fake_azure_core: Any = types.ModuleType("azure.core")
    fake_azure_core_exceptions: Any = types.ModuleType("azure.core.exceptions")
    fake_azure_core_exceptions.ResourceNotFoundError = _FakeResourceNotFoundError

    fake_azure_identity: Any = types.ModuleType("azure.identity")

    fake_azure.sandbox = fake_azure_sandbox
    fake_azure.mgmt = fake_azure_mgmt
    fake_azure.core = fake_azure_core
    fake_azure.identity = fake_azure_identity
    fake_azure_mgmt.sandbox = fake_azure_mgmt_sandbox
    fake_azure_core.exceptions = fake_azure_core_exceptions

    monkeypatch.setitem(sys.modules, "azure", fake_azure)
    monkeypatch.setitem(sys.modules, "azure.sandbox", fake_azure_sandbox)
    monkeypatch.setitem(sys.modules, "azure.mgmt", fake_azure_mgmt)
    monkeypatch.setitem(sys.modules, "azure.mgmt.sandbox", fake_azure_mgmt_sandbox)
    monkeypatch.setitem(sys.modules, "azure.core", fake_azure_core)
    monkeypatch.setitem(sys.modules, "azure.core.exceptions", fake_azure_core_exceptions)
    monkeypatch.setitem(sys.modules, "azure.identity", fake_azure_identity)

    sys.modules.pop("agents.extensions.sandbox.azuresandboxes.sandbox", None)
    sys.modules.pop("agents.extensions.sandbox.azuresandboxes", None)

    return importlib.import_module("agents.extensions.sandbox.azuresandboxes.sandbox")


@pytest.fixture()
def azure_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return the reloaded azuresandboxes.sandbox module with fake SDKs installed."""
    return _install_fake_sdk(monkeypatch)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    mod: Any,
    *,
    sandbox_id: str = "sb-123",
    sandbox_group_id: str = "sg-test",
    root: str = "/home/user/workspace",
    env_vars: dict[str, str] | None = None,
    pause_on_exit: bool = False,
    exposed_ports: tuple[int, ...] = (),
    disk: str | None = None,
    snapshot_id: str | None = None,
    preset: str | None = None,
) -> Any:
    """Build an ``AzureSandboxSessionState`` for tests."""
    sid = uuid.uuid4()
    return mod.AzureSandboxSessionState(
        session_id=sid,
        manifest=Manifest(root=root),
        snapshot=NoopSnapshot(id=str(sid)),
        sandbox_id=sandbox_id,
        sandbox_group_id=sandbox_group_id,
        subscription_id="sub-test",
        resource_group_name="rg-test",
        disk=disk,
        snapshot_id=snapshot_id,
        preset=preset,
        base_env_vars=env_vars or {},
        pause_on_exit=pause_on_exit,
        exposed_ports=exposed_ports,
    )


def _make_session(
    mod: Any,
    client: _FakeSandboxClient | None = None,
    **state_kw: Any,
) -> tuple[Any, _FakeSandboxClient]:
    """Build an ``AzureSandboxSession`` plus its fake data-plane client."""
    if client is None:
        client = _FakeSandboxClient()
    state = _make_state(mod, **state_kw)
    session = mod.AzureSandboxSession.from_state(
        state,
        client=client,
        sandbox_group_id=state.sandbox_group_id,
    )
    return session, client


def _make_client(mod: Any) -> Any:
    """Construct an ``AzureSandboxClient`` with default fake test parameters."""
    return mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )


def _valid_tar_bytes() -> bytes:
    """Return a minimal valid tar archive."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="hello.txt")
        data = b"hello"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


# ===========================================================================
# Package re-export tests
# ===========================================================================


def test_package_re_exports_backend_symbols(azure_mod: Any) -> None:
    package = importlib.import_module("agents.extensions.sandbox.azuresandboxes")
    assert package.AzureSandboxClient is azure_mod.AzureSandboxClient
    assert package.AzureSandboxSession is azure_mod.AzureSandboxSession
    assert package.AzureSandboxClientOptions is azure_mod.AzureSandboxClientOptions
    assert package.AzureSandboxSessionState is azure_mod.AzureSandboxSessionState
    assert package.AzureSandboxTimeouts is azure_mod.AzureSandboxTimeouts
    assert package.AzureSandboxResources is azure_mod.AzureSandboxResources


# ===========================================================================
# Options / state model tests
# ===========================================================================


def test_options_type_literal(azure_mod: Any) -> None:
    opts = azure_mod.AzureSandboxClientOptions(disk="ubuntu")
    assert opts.type == "azuresandboxes"


def test_options_defaults(azure_mod: Any) -> None:
    opts = azure_mod.AzureSandboxClientOptions()
    assert opts.cpu == "1000m"
    assert opts.memory == "2048Mi"
    assert opts.pause_on_exit is False
    assert opts.exposed_ports == ()
    assert opts.auto_suspend_seconds == 300
    assert opts.disk is None
    assert opts.snapshot_id is None
    assert opts.preset is None


def test_options_with_disk(azure_mod: Any) -> None:
    opts = azure_mod.AzureSandboxClientOptions(disk="ubuntu")
    assert opts.disk == "ubuntu"


def test_options_with_snapshot_id(azure_mod: Any) -> None:
    opts = azure_mod.AzureSandboxClientOptions(snapshot_id="snap-1")
    assert opts.snapshot_id == "snap-1"


def test_options_with_preset(azure_mod: Any) -> None:
    opts = azure_mod.AzureSandboxClientOptions(preset="copilot")
    assert opts.preset == "copilot"


def test_session_state_serialization_round_trip(azure_mod: Any) -> None:
    state = _make_state(
        azure_mod,
        sandbox_id="sb-rt-001",
        env_vars={"FOO": "bar"},
        disk="ubuntu",
        preset="copilot",
    )
    dumped = state.model_dump(mode="json")
    restored = azure_mod.AzureSandboxSessionState.model_validate(dumped)
    assert restored.sandbox_id == "sb-rt-001"
    assert restored.base_env_vars == {"FOO": "bar"}
    assert restored.disk == "ubuntu"
    assert restored.preset == "copilot"
    assert restored.auto_suspend_seconds == 300
    assert restored.type == "azuresandboxes"


def test_timeouts_defaults(azure_mod: Any) -> None:
    t = azure_mod.AzureSandboxTimeouts()
    assert t.exec_timeout_unbounded_s == 300
    assert t.keepalive_s == 10
    assert t.cleanup_s == 30
    assert t.fast_op_s == 30
    assert t.file_upload_s == 300
    assert t.file_download_s == 300
    assert t.workspace_tar_s == 300


def test_resources_defaults(azure_mod: Any) -> None:
    r = azure_mod.AzureSandboxResources()
    assert r.cpu == "1000m"
    # AzureSandboxResources uses ``2048Mi`` as the default in the new implementation.
    assert r.memory == "2048Mi"


# ===========================================================================
# Helper function tests
# ===========================================================================


@pytest.mark.parametrize("name", ["FOO", "_X", "FOO_BAR_1", "_", "a", "abc123"])
def test_validate_env_var_name_valid(azure_mod: Any, name: str) -> None:
    azure_mod._validate_env_var_name(name)


@pytest.mark.parametrize("name", ["", "1FOO", "FOO BAR", "FOO-BAR", "FOO=BAR", "FOO.BAR"])
def test_validate_env_var_name_invalid(azure_mod: Any, name: str) -> None:
    with pytest.raises(ValueError):
        azure_mod._validate_env_var_name(name)


def test_build_command_with_env_no_envs(azure_mod: Any) -> None:
    assert azure_mod._build_command_with_env("echo hi", {}) == "echo hi"


def test_build_command_with_env_with_envs(azure_mod: Any) -> None:
    cmd = azure_mod._build_command_with_env("echo hi", {"FOO": "with space", "BAR": "x;y"})
    assert cmd.startswith("env ")
    assert "FOO='with space'" in cmd
    assert "BAR='x;y'" in cmd
    assert " sh -c " in cmd
    # The inner command should be shell-quoted.
    assert "'echo hi'" in cmd


def test_build_command_with_env_invalid_name_raises(azure_mod: Any) -> None:
    with pytest.raises(ValueError):
        azure_mod._build_command_with_env("echo", {"BAD-NAME": "x"})


# ===========================================================================
# Session: exec
# ===========================================================================


@pytest.mark.asyncio
async def test_exec_returns_dict_mapped_to_exec_result(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_exec_result = {"exitCode": 0, "stdout": "hello world", "stderr": ""}

    result = await session._exec_internal("echo", "hello world")

    assert result.exit_code == 0
    assert result.stdout == b"hello world"
    assert result.stderr == b""
    assert len(client.exec_calls) == 1
    call = client.exec_calls[0]
    assert call["sandbox_id"] == "sb-123"
    assert call["sandbox_group"] == "sg-test"
    # The adapter prefixes commands with `cd <root> && ...` instead of passing
    # `working_directory` to the SDK so backends that pre-validate the cwd do
    # not reject early calls before the workspace root is created.
    assert call["working_directory"] is None
    assert call["command"].startswith("cd /home/user/workspace && ")


@pytest.mark.asyncio
async def test_exec_with_env_vars_prefixes_command(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, env_vars={"MY_VAR": "hello"})
    client.next_exec_result = {"exitCode": 0, "stdout": "ok", "stderr": ""}

    await session._exec_internal("env")

    call = client.exec_calls[0]
    assert call["command"].startswith("cd /home/user/workspace && env MY_VAR=hello sh -c ")


@pytest.mark.asyncio
async def test_exec_with_no_env_vars_no_prefix(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_exec_result = {"exitCode": 0, "stdout": "ok", "stderr": ""}

    await session._exec_internal("echo", "hi")

    call = client.exec_calls[0]
    assert not call["command"].startswith("env ")


@pytest.mark.asyncio
async def test_exec_invalid_env_var_name_raises(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, env_vars={"BAD-NAME": "x"})

    # The ``ValueError`` from ``_build_command_with_env`` is raised before the try/except
    # in ``_exec_internal``, so it propagates unchanged.
    with pytest.raises(ValueError, match="invalid environment variable name"):
        await session._exec_internal("echo", "hi")


@pytest.mark.asyncio
async def test_exec_timeout_raises_exec_timeout_error(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.exec_delay_s = 0.5

    with pytest.raises(ExecTimeoutError):
        await session._exec_internal("sleep", "100", timeout=0.05)


@pytest.mark.asyncio
async def test_exec_transport_error(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.exec_error = ConnectionError("connection refused")

    with pytest.raises(ExecTransportError):
        await session._exec_internal("echo", "hi")


@pytest.mark.asyncio
async def test_exec_nonzero_exit_code(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_exec_result = {"exitCode": 1, "stdout": "", "stderr": "err msg"}

    result = await session._exec_internal("false")

    assert result.exit_code == 1
    assert result.stderr == b"err msg"


# ===========================================================================
# Session: supports_pty
# ===========================================================================


def test_supports_pty_false(azure_mod: Any) -> None:
    session, _ = _make_session(azure_mod)
    assert session.supports_pty() is False


# ===========================================================================
# Session: read
# ===========================================================================


@pytest.mark.asyncio
async def test_read_returns_bytes_io(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_read_file = b"file content here"

    result = await session.read("/home/user/workspace/test.txt")

    assert isinstance(result, io.BytesIO)
    assert result.read() == b"file content here"
    assert client.read_file_calls[0] == (
        "sb-123",
        "sg-test",
        "/home/user/workspace/test.txt",
    )


@pytest.mark.asyncio
async def test_read_not_found_via_resource_not_found(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.read_file_error = _FakeResourceNotFoundError("missing")

    with pytest.raises(WorkspaceReadNotFoundError):
        await session.read("/home/user/workspace/missing.txt")


@pytest.mark.asyncio
async def test_read_not_found_via_string_match(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.read_file_error = Exception("not found 404")

    with pytest.raises(WorkspaceReadNotFoundError):
        await session.read("/home/user/workspace/missing.txt")


@pytest.mark.asyncio
async def test_read_other_error_raises_archive_read_error(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.read_file_error = RuntimeError("transport failure")

    with pytest.raises(WorkspaceArchiveReadError):
        await session.read("/home/user/workspace/some.txt")


# ===========================================================================
# Session: write
# ===========================================================================


@pytest.mark.asyncio
async def test_write_sends_bytes(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    data = io.BytesIO(b"new content")

    await session.write("/home/user/workspace/output.txt", data)

    assert len(client.write_file_calls) == 1
    sid, group, path, content, create_dirs = client.write_file_calls[0]
    assert sid == "sb-123"
    assert group == "sg-test"
    assert path == "/home/user/workspace/output.txt"
    assert content == b"new content"
    assert create_dirs is True


@pytest.mark.asyncio
async def test_write_string_io_encodes_utf8(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    data = io.StringIO("unicode text")

    await session.write("/home/user/workspace/text.txt", data)

    _, _, _, content, _ = client.write_file_calls[0]
    assert content == b"unicode text"


@pytest.mark.asyncio
async def test_write_error_raises_archive_write_error(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.write_file_error = RuntimeError("write failed")

    with pytest.raises(WorkspaceArchiveWriteError):
        await session.write("/home/user/workspace/f.txt", io.BytesIO(b"x"))


# ===========================================================================
# Session: mkdir
# ===========================================================================


@pytest.mark.asyncio
async def test_mkdir_no_parents_calls_sdk(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)

    await session.mkdir("/home/user/workspace/subdir", parents=False)

    assert len(client.mkdir_calls) == 1
    sid, group, path = client.mkdir_calls[0]
    assert sid == "sb-123"
    assert group == "sg-test"
    assert path == "/home/user/workspace/subdir"


@pytest.mark.asyncio
async def test_mkdir_with_parents_uses_exec(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_exec_result = {"exitCode": 0, "stdout": "", "stderr": ""}

    await session.mkdir("/home/user/workspace/deep/nested", parents=True)

    # ``parents=True`` should not invoke the SDK ``mkdir`` directly.
    assert client.mkdir_calls == []
    # The workspace mkdir should be present in the recorded exec calls.
    workspace_mkdirs = [
        c["command"]
        for c in client.exec_calls
        if "mkdir" in c["command"] and "/home/user/workspace/deep/nested" in c["command"]
    ]
    assert workspace_mkdirs, client.exec_calls


@pytest.mark.asyncio
async def test_mkdir_parents_exec_failure_raises(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_exec_result = {"exitCode": 1, "stdout": "", "stderr": "denied"}

    with pytest.raises(WorkspaceArchiveWriteError):
        await session.mkdir("/home/user/workspace/bad", parents=True)


@pytest.mark.asyncio
async def test_mkdir_no_parents_error_raises_archive_write_error(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.mkdir_error = RuntimeError("mkdir failed")

    with pytest.raises(WorkspaceArchiveWriteError):
        await session.mkdir("/home/user/workspace/bad", parents=False)


# ===========================================================================
# Session: running
# ===========================================================================


@pytest.mark.asyncio
async def test_running_returns_true_when_running(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_get_sandbox = {"id": "sb-123", "state": "Running"}

    assert await session.running() is True


@pytest.mark.asyncio
async def test_running_returns_false_when_stopped(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_get_sandbox = {"id": "sb-123", "state": "Stopped"}

    assert await session.running() is False


@pytest.mark.asyncio
async def test_running_returns_false_when_other_state(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_get_sandbox = {"id": "sb-123", "state": "Pending"}

    assert await session.running() is False


@pytest.mark.asyncio
async def test_running_returns_false_on_error(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.get_sandbox_error = ConnectionError("unavailable")

    assert await session.running() is False


@pytest.mark.asyncio
async def test_running_returns_false_when_not_dict(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_get_sandbox = None  # type: ignore[assignment]

    assert await session.running() is False


# ===========================================================================
# Session: shutdown
# ===========================================================================


@pytest.mark.asyncio
async def test_shutdown_stops_when_pause_on_exit(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, pause_on_exit=True)

    await session._shutdown_backend()

    assert client.stop_sandbox_calls == [("sb-123", "sg-test")]
    assert client.delete_sandbox_calls == []


@pytest.mark.asyncio
async def test_shutdown_deletes_when_not_pause(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, pause_on_exit=False)

    await session._shutdown_backend()

    assert client.delete_sandbox_calls == [("sb-123", "sg-test")]
    assert client.stop_sandbox_calls == []


@pytest.mark.asyncio
async def test_shutdown_swallows_errors(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, pause_on_exit=True)
    client.stop_sandbox_error = RuntimeError("stop failed")

    await session._shutdown_backend()


# ===========================================================================
# Session: resolve_exposed_port
# ===========================================================================


@pytest.mark.asyncio
async def test_resolve_exposed_port_parses_https_url(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(8080,))
    client.next_get_sandbox = {
        "id": "sb-123",
        "state": "Running",
        "ports": [{"port": 8080, "url": "https://sb-123-8080.westus2.example.com"}],
    }

    endpoint = await session._resolve_exposed_port(8080)

    assert endpoint.host == "sb-123-8080.westus2.example.com"
    assert endpoint.port == 443
    assert endpoint.tls is True


@pytest.mark.asyncio
async def test_resolve_exposed_port_parses_http_url(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(3000,))
    client.next_get_sandbox = {
        "id": "sb-123",
        "state": "Running",
        "ports": [{"port": 3000, "url": "http://sb-123-3000.local:3000"}],
    }

    endpoint = await session._resolve_exposed_port(3000)

    assert endpoint.host == "sb-123-3000.local"
    assert endpoint.port == 3000
    assert endpoint.tls is False


@pytest.mark.asyncio
async def test_resolve_exposed_port_not_found_in_list(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(8080,))
    client.next_get_sandbox = {
        "id": "sb-123",
        "ports": [{"port": 9999, "url": "https://other.example.com"}],
    }

    with pytest.raises(ExposedPortUnavailableError) as ei:
        await session._resolve_exposed_port(8080)
    assert ei.value.context["reason"] == "port_not_found"


@pytest.mark.asyncio
async def test_resolve_exposed_port_missing_ports_field(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(8080,))
    client.next_get_sandbox = {"id": "sb-123"}

    with pytest.raises(ExposedPortUnavailableError) as ei:
        await session._resolve_exposed_port(8080)
    assert ei.value.context["reason"] == "port_not_found"


@pytest.mark.asyncio
async def test_resolve_exposed_port_ports_is_none(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(8080,))
    client.next_get_sandbox = {"id": "sb-123", "ports": None}

    with pytest.raises(ExposedPortUnavailableError) as ei:
        await session._resolve_exposed_port(8080)
    assert ei.value.context["reason"] == "port_not_found"


@pytest.mark.asyncio
async def test_resolve_exposed_port_malformed_entry(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(8080,))
    client.next_get_sandbox = {
        "id": "sb-123",
        "ports": ["not-a-dict", {"port": "bad-int"}],
    }

    with pytest.raises(ExposedPortUnavailableError) as ei:
        await session._resolve_exposed_port(8080)
    assert ei.value.context["reason"] == "port_not_found"


@pytest.mark.asyncio
async def test_resolve_exposed_port_invalid_url(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(8080,))
    client.next_get_sandbox = {
        "id": "sb-123",
        "ports": [{"port": 8080, "url": ""}],
    }

    with pytest.raises(ExposedPortUnavailableError) as ei:
        await session._resolve_exposed_port(8080)
    assert ei.value.context["reason"] == "backend_unavailable"


@pytest.mark.asyncio
async def test_resolve_exposed_port_backend_error(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod, exposed_ports=(8080,))
    client.get_sandbox_error = ConnectionError("unavailable")

    with pytest.raises(ExposedPortUnavailableError) as ei:
        await session._resolve_exposed_port(8080)
    assert ei.value.context["reason"] == "backend_unavailable"


# ===========================================================================
# Session: persist_workspace / hydrate_workspace
# ===========================================================================


@pytest.mark.asyncio
async def test_persist_workspace_creates_tar_and_cleans_up(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    tar_bytes = _valid_tar_bytes()
    client.next_exec_result = {"exitCode": 0, "stdout": "", "stderr": ""}
    client.next_read_file = tar_bytes

    result = await session.persist_workspace()

    assert isinstance(result, io.BytesIO)
    assert result.read() == tar_bytes

    workspace_execs = [
        c for c in client.exec_calls if c["sandbox_group"] == "sg-test" and "tar" in c["command"]
    ]
    assert workspace_execs, client.exec_calls
    assert any("rm -f" in c["command"] for c in client.exec_calls)
    # All workspace ops should pass the configured sandbox_group.
    assert all(c["sandbox_group"] == "sg-test" for c in client.exec_calls)


@pytest.mark.asyncio
async def test_hydrate_workspace_uploads_and_extracts(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    tar_data = _valid_tar_bytes()
    client.next_exec_result = {"exitCode": 0, "stdout": "", "stderr": ""}

    await session.hydrate_workspace(io.BytesIO(tar_data))

    tar_writes = [call for call in client.write_file_calls if call[2].endswith(".tar")]
    assert tar_writes, client.write_file_calls
    tar_extracts = [c for c in client.exec_calls if "tar" in c["command"] and "-xf" in c["command"]]
    assert tar_extracts, client.exec_calls


@pytest.mark.asyncio
async def test_hydrate_workspace_rejects_unsafe_tar(azure_mod: Any) -> None:
    session, _ = _make_session(azure_mod)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        info = tarfile.TarInfo(name="/etc/passwd")
        data = b"root:x:0:0:::"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    unsafe_tar = buf.getvalue()

    with pytest.raises(WorkspaceArchiveWriteError):
        await session.hydrate_workspace(io.BytesIO(unsafe_tar))


# ===========================================================================
# Session: prepare workspace root
# ===========================================================================


@pytest.mark.asyncio
async def test_prepare_workspace_root_uses_exec_mkdir_p(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_exec_result = {"exitCode": 0, "stdout": "", "stderr": ""}

    await session._prepare_workspace_root()

    # _prepare_workspace_root must bypass _exec_internal (which prefixes with
    # `cd <root> && ...`) because the workspace root may not exist yet. It must
    # also skip working_directory for the same reason.
    matching = [
        c
        for c in client.exec_calls
        if c["command"] == "mkdir -p -- /home/user/workspace"
        and c["sandbox_group"] == "sg-test"
        and c["working_directory"] is None
    ]
    assert matching, client.exec_calls


@pytest.mark.asyncio
async def test_prepare_workspace_root_exec_failure_raises(azure_mod: Any) -> None:
    session, client = _make_session(azure_mod)
    client.next_exec_result = {"exitCode": 1, "stdout": "", "stderr": "permission denied"}

    with pytest.raises(WorkspaceStartError):
        await session._prepare_workspace_root()


# ===========================================================================
# Client: create
# ===========================================================================


@pytest.mark.asyncio
async def test_client_create_with_disk(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {"id": "sb-new-1"}

    options = azure_mod.AzureSandboxClientOptions(disk="ubuntu")
    await client.create(options=options)

    assert len(fake.create_sandbox_calls) == 1
    call = fake.create_sandbox_calls[0]
    assert call["sandbox_group"] == "sg-1"
    assert call["disk"] == "ubuntu"
    assert call["cpu"] == "1000m"
    assert call["memory"] == "2048Mi"
    assert call["auto_suspend_seconds"] == 300


@pytest.mark.asyncio
async def test_client_create_with_snapshot_id(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {"id": "sb-snap-1"}

    options = azure_mod.AzureSandboxClientOptions(snapshot_id="snap-1")
    await client.create(options=options)

    call = fake.create_sandbox_calls[0]
    assert call["snapshot_id"] == "snap-1"


@pytest.mark.asyncio
async def test_client_create_with_preset(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {"id": "sb-preset-1"}

    options = azure_mod.AzureSandboxClientOptions(preset="copilot")
    await client.create(options=options)

    call = fake.create_sandbox_calls[0]
    assert call["preset"] == "copilot"


@pytest.mark.asyncio
async def test_client_create_passes_env_vars(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {"id": "sb-env-1"}

    options = azure_mod.AzureSandboxClientOptions(disk="ubuntu", env_vars={"KEY": "val"})
    await client.create(options=options)

    call = fake.create_sandbox_calls[0]
    assert call["environment"] == {"KEY": "val"}


@pytest.mark.asyncio
async def test_client_create_passes_exposed_ports(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {"id": "sb-port-1"}

    options = azure_mod.AzureSandboxClientOptions(disk="ubuntu", exposed_ports=(8080, 9000))
    await client.create(options=options)

    call = fake.create_sandbox_calls[0]
    assert call["ports"] == [{"port": 8080}, {"port": 9000}]


@pytest.mark.asyncio
async def test_client_create_passes_labels_entrypoint_cmd(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {"id": "sb-meta-1"}

    options = azure_mod.AzureSandboxClientOptions(
        disk="ubuntu",
        labels={"team": "infra"},
        entrypoint=["/bin/sh"],
        cmd=["-c", "echo hi"],
    )
    await client.create(options=options)

    call = fake.create_sandbox_calls[0]
    assert call["labels"] == {"team": "infra"}
    assert call["entrypoint"] == ["/bin/sh"]
    assert call["cmd"] == ["-c", "echo hi"]


@pytest.mark.asyncio
async def test_client_create_raises_without_source(azure_mod: Any) -> None:
    client = _make_client(azure_mod)

    options = azure_mod.AzureSandboxClientOptions()
    with pytest.raises(ValueError, match="disk, snapshot_id, or preset"):
        await client.create(options=options)


@pytest.mark.asyncio
async def test_client_create_raises_when_id_missing(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {}

    options = azure_mod.AzureSandboxClientOptions(disk="ubuntu")
    with pytest.raises(ValueError, match="missing 'id'"):
        await client.create(options=options)


# ===========================================================================
# Client: resume
# ===========================================================================


@pytest.mark.asyncio
async def test_client_resume_reconnects_running(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_get_sandbox = {"id": "sb-123", "state": "Running"}

    state = _make_state(azure_mod, sandbox_id="sb-123", sandbox_group_id="sg-1", disk="ubuntu")
    await client.resume(state)

    assert fake.get_sandbox_calls == [("sb-123", "sg-1")]
    assert fake.resume_sandbox_calls == []
    assert fake.create_sandbox_calls == []


@pytest.mark.asyncio
async def test_client_resume_resumes_stopped(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_get_sandbox = {"id": "sb-123", "state": "Stopped"}

    state = _make_state(azure_mod, sandbox_id="sb-123", sandbox_group_id="sg-1", disk="ubuntu")
    await client.resume(state)

    assert fake.resume_sandbox_calls == [("sb-123", "sg-1")]


@pytest.mark.asyncio
async def test_client_resume_recreates_on_not_found(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.get_sandbox_error = _FakeResourceNotFoundError("missing")
    fake.next_create_sandbox = {"id": "sb-recreated"}

    state = _make_state(azure_mod, sandbox_id="sb-gone", sandbox_group_id="sg-1", disk="ubuntu")
    await client.resume(state)

    assert len(fake.create_sandbox_calls) == 1
    assert fake.create_sandbox_calls[0]["disk"] == "ubuntu"
    assert state.sandbox_id == "sb-recreated"


@pytest.mark.asyncio
async def test_client_resume_recreates_with_snapshot(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.get_sandbox_error = _FakeResourceNotFoundError("missing")
    fake.next_create_sandbox = {"id": "sb-recreated"}

    state = _make_state(
        azure_mod, sandbox_id="sb-gone", sandbox_group_id="sg-1", snapshot_id="snap-1"
    )
    await client.resume(state)

    assert fake.create_sandbox_calls[0]["snapshot_id"] == "snap-1"


@pytest.mark.asyncio
async def test_client_resume_recreates_with_preset(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.get_sandbox_error = _FakeResourceNotFoundError("missing")
    fake.next_create_sandbox = {"id": "sb-recreated"}

    state = _make_state(azure_mod, sandbox_id="sb-gone", sandbox_group_id="sg-1", preset="copilot")
    await client.resume(state)

    assert fake.create_sandbox_calls[0]["preset"] == "copilot"


@pytest.mark.asyncio
async def test_client_resume_raises_when_no_source_to_recreate(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.get_sandbox_error = _FakeResourceNotFoundError("missing")

    state = _make_state(azure_mod, sandbox_id="sb-gone", sandbox_group_id="sg-1")
    with pytest.raises(ValueError, match="no disk, snapshot_id, or preset"):
        await client.resume(state)


# ===========================================================================
# Client: delete
# ===========================================================================


@pytest.mark.asyncio
async def test_client_delete(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    fake = client._client
    fake.next_create_sandbox = {"id": "sb-del"}

    options = azure_mod.AzureSandboxClientOptions(disk="ubuntu")
    session = await client.create(options=options)
    await client.delete(session)

    assert len(fake.delete_sandbox_calls) == 1


# ===========================================================================
# Client: close / context manager
# ===========================================================================


@pytest.mark.asyncio
async def test_client_close_closes_both_clients(azure_mod: Any) -> None:
    client = _make_client(azure_mod)

    await client.close()

    assert client._client.closed is True
    assert client._mgmt_client.closed is True


@pytest.mark.asyncio
async def test_client_context_manager(azure_mod: Any) -> None:
    async with _make_client(azure_mod) as client:
        pass

    assert client._client.closed is True
    assert client._mgmt_client.closed is True


# ===========================================================================
# Client: deserialize_session_state
# ===========================================================================


def test_deserialize_session_state(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    state = _make_state(azure_mod, sandbox_id="sb-deser")
    payload = state.model_dump(mode="json")

    restored = client.deserialize_session_state(payload)

    assert restored.sandbox_id == "sb-deser"
    assert restored.type == "azuresandboxes"


# ===========================================================================
# _is_not_found_error helper
# ===========================================================================


def test_is_not_found_error_resource_not_found(azure_mod: Any) -> None:
    assert azure_mod._is_not_found_error(_FakeResourceNotFoundError("gone")) is True


def test_is_not_found_error_string_match(azure_mod: Any) -> None:
    assert azure_mod._is_not_found_error(Exception("resource not found")) is True
    assert azure_mod._is_not_found_error(Exception("404 Not Found")) is True
    assert azure_mod._is_not_found_error(Exception("something else")) is False


# ===========================================================================
# Group management helpers (NEW)
# ===========================================================================


@pytest.mark.asyncio
async def test_list_sandbox_groups(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    mgmt = client._mgmt_client
    mgmt.next_list_groups = [{"name": "sg-a"}, {"name": "sg-b"}]

    result = await client.list_sandbox_groups()

    assert result == [{"name": "sg-a"}, {"name": "sg-b"}]
    assert len(mgmt.list_groups_calls) == 1


@pytest.mark.asyncio
async def test_get_sandbox_group_default_name(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    mgmt = client._mgmt_client

    await client.get_sandbox_group()

    assert mgmt.get_group_calls == ["sg-1"]


@pytest.mark.asyncio
async def test_get_sandbox_group_custom_name(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    mgmt = client._mgmt_client

    await client.get_sandbox_group("other-group")

    assert mgmt.get_group_calls == ["other-group"]


@pytest.mark.asyncio
async def test_create_sandbox_group(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    mgmt = client._mgmt_client

    await client.create_sandbox_group(
        "westus2",
        identity={"type": "SystemAssigned"},
        tags={"env": "test"},
        foo="bar",
    )

    assert len(mgmt.create_group_calls) == 1
    call = mgmt.create_group_calls[0]
    assert call["name"] == "sg-1"
    assert call["location"] == "westus2"
    assert call["identity"] == {"type": "SystemAssigned"}
    assert call["tags"] == {"env": "test"}
    assert call["properties"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_delete_sandbox_group(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    mgmt = client._mgmt_client

    await client.delete_sandbox_group()

    assert mgmt.delete_group_calls == ["sg-1"]


@pytest.mark.asyncio
async def test_ensure_sandbox_group_existing(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    mgmt = client._mgmt_client
    mgmt.next_get_group = {"name": "sg-1", "location": "westus2"}

    result = await client.ensure_sandbox_group("westus2")

    assert result == {"name": "sg-1", "location": "westus2"}
    assert mgmt.create_group_calls == []


@pytest.mark.asyncio
async def test_ensure_sandbox_group_creates_when_missing(azure_mod: Any) -> None:
    client = _make_client(azure_mod)
    mgmt = client._mgmt_client
    mgmt.get_group_error = _FakeResourceNotFoundError("missing")
    mgmt.next_create_group = {"name": "sg-1", "location": "westus2"}

    result = await client.ensure_sandbox_group(
        "westus2", identity={"type": "SystemAssigned"}, tags={"env": "test"}
    )

    assert result == {"name": "sg-1", "location": "westus2"}
    assert len(mgmt.create_group_calls) == 1
    call = mgmt.create_group_calls[0]
    assert call["name"] == "sg-1"
    assert call["location"] == "westus2"


# Sanity: ensure ``asyncio`` import is referenced; ruff would otherwise flag it.
_ = asyncio
