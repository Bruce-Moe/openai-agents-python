from __future__ import annotations

import asyncio
import importlib
import io
import shlex
import sys
import tarfile
import types
import uuid
from typing import Any
from unittest.mock import MagicMock

import pytest

from agents.sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
)
from agents.sandbox.manifest import Manifest
from agents.sandbox.snapshot import NoopSnapshot
from tests._fake_workspace_paths import resolve_fake_workspace_path

# ---------------------------------------------------------------------------
# Fake ADC SDK stubs (injected into sys.modules so the lazy import helpers work)
# ---------------------------------------------------------------------------


class _FakeSandboxData:
    """Mimics ``adc_core.models.sandbox.SandboxData``."""

    def __init__(self, *, id: str = "sb-123", state: str = "Running") -> None:
        self.id = id
        self.state = state


class _FakeCommandResult:
    """Mimics ``adc_core.models.sandbox.CommandExecutionResult``."""

    def __init__(self, *, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class _FakePort:
    """Mimics ``adc_core.models.sandbox.SandboxPort``."""

    def __init__(self, *, port: int, url: str) -> None:
        self.port = port
        self.url = url


class _FakeDiskImage:
    def __init__(self, *, id: str = "di-new-123") -> None:
        self.id = id


class _FakeSandboxSourceDiskImageById:
    def __init__(self, *, id: str) -> None:
        self.id = id


class _FakeSandboxSourceSnapshot:
    def __init__(self, *, id: str) -> None:
        self.id = id


class _FakeSandboxOperations:
    """Fake for ``SandboxGroupScope.sandboxes``."""

    def __init__(self) -> None:
        self.exec_calls: list[dict[str, Any]] = []
        self.read_file_calls: list[tuple[str, str]] = []
        self.write_file_calls: list[tuple[str, str, bytes, bool]] = []
        self.mkdir_calls: list[tuple[str, str, bool]] = []
        self.get_calls: list[str] = []
        self.get_ports_calls: list[str] = []
        self.stop_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.resume_calls: list[str] = []
        self.create_from_disk_image_calls: list[dict[str, Any]] = []
        self.create_from_snapshot_calls: list[dict[str, Any]] = []

        # Configurable return values
        self.next_exec_result = _FakeCommandResult()
        self.next_get_result = _FakeSandboxData()
        self.next_read_file_result: bytes = b"file-content"
        self.next_ports: list[_FakePort] = []
        self.next_create_result = _FakeSandboxData(id="sb-new-456")
        self.exec_delay_s: float = 0.0
        self.get_error: BaseException | None = None
        self.read_file_error: BaseException | None = None

    async def execute_shell_command(
        self,
        sandbox_id: str,
        command: str,
        shell: str = "/bin/sh",
        environment: dict[str, str] | None = None,
        working_directory: str | None = None,
    ) -> _FakeCommandResult:
        self.exec_calls.append(
            {
                "sandbox_id": sandbox_id,
                "command": command,
                "shell": shell,
                "environment": environment,
                "working_directory": working_directory,
            }
        )

        # Handle runtime helper install/probe commands (mkdir -p, test -x, sh -c cat)
        parts = shlex.split(command)
        if parts and parts[0] == "mkdir" and "-p" in parts:
            return _FakeCommandResult(exit_code=0)
        if parts and parts[0] == "test" and "-x" in parts:
            return _FakeCommandResult(exit_code=0)
        if command.startswith("sh -c cat"):
            return _FakeCommandResult(exit_code=0)

        # Handle resolve-workspace-path helper
        resolved = resolve_fake_workspace_path(
            command,
            symlinks={},
            home_dir="/home/user/workspace",
        )
        if resolved is not None:
            return _FakeCommandResult(
                exit_code=resolved.exit_code,
                stdout=resolved.stdout,
                stderr=resolved.stderr,
            )

        if self.exec_delay_s > 0:
            await asyncio.sleep(self.exec_delay_s)
        result = self.next_exec_result
        self.next_exec_result = _FakeCommandResult()
        return result

    async def read_file(self, sandbox_id: str, path: str) -> bytes:
        self.read_file_calls.append((sandbox_id, path))
        if self.read_file_error is not None:
            raise self.read_file_error
        return self.next_read_file_result

    async def write_file(
        self,
        sandbox_id: str,
        path: str,
        content: bytes,
        create_dirs: bool = False,
    ) -> None:
        self.write_file_calls.append((sandbox_id, path, content, create_dirs))

    async def mkdir(
        self,
        sandbox_id: str,
        path: str,
        create_parents: bool = False,
    ) -> None:
        self.mkdir_calls.append((sandbox_id, path, create_parents))

    async def get(self, sandbox_id: str) -> _FakeSandboxData:
        self.get_calls.append(sandbox_id)
        if self.get_error is not None:
            raise self.get_error
        return self.next_get_result

    async def get_ports(self, sandbox_id: str) -> list[_FakePort]:
        self.get_ports_calls.append(sandbox_id)
        return self.next_ports

    async def stop(self, sandbox_id: str) -> None:
        self.stop_calls.append(sandbox_id)

    async def delete(self, sandbox_id: str) -> None:
        self.delete_calls.append(sandbox_id)

    async def resume(self, sandbox_id: str) -> None:
        self.resume_calls.append(sandbox_id)

    async def create_from_disk_image(self, **kwargs: Any) -> _FakeSandboxData:
        self.create_from_disk_image_calls.append(kwargs)
        return self.next_create_result

    async def create_from_snapshot(self, **kwargs: Any) -> _FakeSandboxData:
        self.create_from_snapshot_calls.append(kwargs)
        return self.next_create_result


class _FakeDiskImageOperations:
    """Fake for ``SandboxGroupScope.disk_images``."""

    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeDiskImage:
        self.create_calls.append(kwargs)
        return _FakeDiskImage()


class _FakeScope:
    """Fake for ``SandboxGroupScope``."""

    def __init__(self) -> None:
        self.sandboxes = _FakeSandboxOperations()
        self.disk_images = _FakeDiskImageOperations()


class _FakeArmAdcClient:
    def __init__(self, options: Any) -> None:
        self.options = options
        self._scope = _FakeScope()
        self.closed = False

    def for_sandbox_group(self, group_id: str) -> _FakeScope:
        return self._scope

    async def close(self) -> None:
        self.closed = True


class _FakeArmAdcClientOptions:
    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Module-level fixtures — inject fake SDK into sys.modules
# ---------------------------------------------------------------------------


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Insert fake ``adc_arm`` and ``adc_core`` into sys.modules and return the loaded module."""

    # Build fake adc_arm module hierarchy
    fake_adc_arm: Any = types.ModuleType("adc_arm")
    fake_adc_arm.ArmAdcClient = _FakeArmAdcClient
    fake_adc_arm.ArmAdcClientOptions = _FakeArmAdcClientOptions

    # Build fake adc_core module hierarchy
    fake_adc_core: Any = types.ModuleType("adc_core")
    fake_adc_core_models: Any = types.ModuleType("adc_core.models")
    fake_adc_core_models_sandbox: Any = types.ModuleType("adc_core.models.sandbox")
    fake_adc_core_models_sandbox.SandboxSourceDiskImageById = _FakeSandboxSourceDiskImageById
    fake_adc_core_models_sandbox.SandboxSourceSnapshot = _FakeSandboxSourceSnapshot
    fake_adc_core.models = fake_adc_core_models
    fake_adc_core_models.sandbox = fake_adc_core_models_sandbox

    monkeypatch.setitem(sys.modules, "adc_arm", fake_adc_arm)
    monkeypatch.setitem(sys.modules, "adc_core", fake_adc_core)
    monkeypatch.setitem(sys.modules, "adc_core.models", fake_adc_core_models)
    monkeypatch.setitem(sys.modules, "adc_core.models.sandbox", fake_adc_core_models_sandbox)

    # Force reimport so lazy imports pick up our fakes
    sys.modules.pop("agents.extensions.sandbox.azuresandboxes.sandbox", None)
    sys.modules.pop("agents.extensions.sandbox.azuresandboxes", None)

    mod = importlib.import_module("agents.extensions.sandbox.azuresandboxes.sandbox")
    return {
        "mod": mod,
        "scope": _FakeScope,
        "arm_client": _FakeArmAdcClient,
    }


@pytest.fixture()
def azure_mod(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Return the reloaded azuresandboxes.sandbox module with fake SDK."""
    info = _install_fake_sdk(monkeypatch)
    return info["mod"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(
    mod: Any,
    *,
    sandbox_id: str = "sb-123",
    root: str = "/home/user/workspace",
    env_vars: dict[str, str] | None = None,
    pause_on_exit: bool = False,
    exposed_ports: tuple[int, ...] = (),
) -> Any:
    """Build an ``AzureSandboxSessionState`` for tests."""
    sid = uuid.uuid4()
    return mod.AzureSandboxSessionState(
        session_id=sid,
        manifest=Manifest(root=root),
        snapshot=NoopSnapshot(id=str(sid)),
        sandbox_id=sandbox_id,
        sandbox_group_id="sg-test",
        subscription_id="sub-test",
        resource_group_name="rg-test",
        base_env_vars=env_vars or {},
        pause_on_exit=pause_on_exit,
        exposed_ports=exposed_ports,
    )


def _make_session(mod: Any, scope: _FakeScope | None = None, **state_kw: Any) -> Any:
    """Build an ``AzureSandboxSession`` with a fake scope."""
    if scope is None:
        scope = _FakeScope()
    state = _make_state(mod, **state_kw)
    return mod.AzureSandboxSession.from_state(state, scope=scope), scope


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
    opts = azure_mod.AzureSandboxClientOptions(disk_image_id="di-1")
    assert opts.type == "azuresandboxes"


def test_options_defaults(azure_mod: Any) -> None:
    opts = azure_mod.AzureSandboxClientOptions()
    assert opts.cpu == "1000m"
    assert opts.memory == "1024Mi"
    assert opts.pause_on_exit is False
    assert opts.exposed_ports == ()


def test_session_state_serialization_round_trip(azure_mod: Any) -> None:
    state = _make_state(azure_mod, sandbox_id="sb-rt-001", env_vars={"FOO": "bar"})
    dumped = state.model_dump(mode="json")
    restored = azure_mod.AzureSandboxSessionState.model_validate(dumped)
    assert restored.sandbox_id == "sb-rt-001"
    assert restored.base_env_vars == {"FOO": "bar"}
    assert restored.type == "azuresandboxes"


def test_timeouts_defaults(azure_mod: Any) -> None:
    t = azure_mod.AzureSandboxTimeouts()
    assert t.exec_timeout_unbounded_s == 300
    assert t.keepalive_s == 10
    assert t.cleanup_s == 30


def test_resources_defaults(azure_mod: Any) -> None:
    r = azure_mod.AzureSandboxResources()
    assert r.cpu == "1000m"
    assert r.memory == "1024Mi"


# ===========================================================================
# Session: exec tests
# ===========================================================================


@pytest.mark.asyncio
async def test_exec_maps_command_result(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    scope.sandboxes.next_exec_result = _FakeCommandResult(
        exit_code=0, stdout="hello world", stderr=""
    )

    result = await session._exec_internal("echo", "hello world")

    assert result.exit_code == 0
    assert result.stdout == b"hello world"
    assert result.stderr == b""
    assert len(scope.sandboxes.exec_calls) == 1
    call = scope.sandboxes.exec_calls[0]
    assert call["shell"] == "/bin/sh"
    assert call["sandbox_id"] == "sb-123"


@pytest.mark.asyncio
async def test_exec_with_env_vars(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod, env_vars={"MY_VAR": "hello"})
    scope.sandboxes.next_exec_result = _FakeCommandResult(exit_code=0, stdout="ok", stderr="")

    await session._exec_internal("env")

    call = scope.sandboxes.exec_calls[0]
    assert call["environment"]["MY_VAR"] == "hello"


@pytest.mark.asyncio
async def test_exec_timeout_raises(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    scope.sandboxes.exec_delay_s = 5.0

    with pytest.raises(ExecTimeoutError):
        await session._exec_internal("sleep", "100", timeout=0.05)


@pytest.mark.asyncio
async def test_exec_transport_error(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)

    async def _raise(*a: Any, **kw: Any) -> None:
        raise ConnectionError("connection refused")

    scope.sandboxes.execute_shell_command = _raise  # type: ignore[assignment]

    with pytest.raises(ExecTransportError):
        await session._exec_internal("echo", "hi")


@pytest.mark.asyncio
async def test_exec_nonzero_exit_code(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    scope.sandboxes.next_exec_result = _FakeCommandResult(
        exit_code=1, stdout="", stderr="error msg"
    )

    result = await session._exec_internal("false")

    assert result.exit_code == 1
    assert result.stderr == b"error msg"


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
    session, scope = _make_session(azure_mod)
    scope.sandboxes.next_read_file_result = b"file content here"

    result = await session.read("/home/user/workspace/test.txt")

    assert isinstance(result, io.BytesIO)
    assert result.read() == b"file content here"
    assert scope.sandboxes.read_file_calls[0] == ("sb-123", "/home/user/workspace/test.txt")


@pytest.mark.asyncio
async def test_read_not_found(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    scope.sandboxes.read_file_error = Exception("not found 404")

    with pytest.raises(WorkspaceReadNotFoundError):
        await session.read("/home/user/workspace/missing.txt")


# ===========================================================================
# Session: write
# ===========================================================================


@pytest.mark.asyncio
async def test_write_sends_bytes(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    data = io.BytesIO(b"new content")

    await session.write("/home/user/workspace/output.txt", data)

    assert len(scope.sandboxes.write_file_calls) == 1
    sid, path, content, create_dirs = scope.sandboxes.write_file_calls[0]
    assert sid == "sb-123"
    assert path == "/home/user/workspace/output.txt"
    assert content == b"new content"
    assert create_dirs is True


@pytest.mark.asyncio
async def test_write_string_io_encodes_utf8(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    data = io.StringIO("unicode text")

    await session.write("/home/user/workspace/text.txt", data)

    _, _, content, _ = scope.sandboxes.write_file_calls[0]
    assert content == b"unicode text"


@pytest.mark.asyncio
async def test_write_error_raises(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)

    async def _raise(*a: Any, **kw: Any) -> None:
        raise RuntimeError("write failed")

    scope.sandboxes.write_file = _raise  # type: ignore[assignment]

    with pytest.raises(WorkspaceArchiveWriteError):
        await session.write("/home/user/workspace/f.txt", io.BytesIO(b"x"))


# ===========================================================================
# Session: mkdir
# ===========================================================================


@pytest.mark.asyncio
async def test_mkdir_calls_backend(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)

    await session.mkdir("/home/user/workspace/subdir", parents=True)

    assert len(scope.sandboxes.mkdir_calls) == 1
    sid, path, parents = scope.sandboxes.mkdir_calls[0]
    assert sid == "sb-123"
    assert path == "/home/user/workspace/subdir"
    assert parents is True


@pytest.mark.asyncio
async def test_mkdir_error_raises(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)

    async def _raise(*a: Any, **kw: Any) -> None:
        raise RuntimeError("mkdir failed")

    scope.sandboxes.mkdir = _raise  # type: ignore[assignment]

    with pytest.raises(WorkspaceArchiveWriteError):
        await session.mkdir("/home/user/workspace/bad", parents=False)


# ===========================================================================
# Session: running
# ===========================================================================


@pytest.mark.asyncio
async def test_running_returns_true_when_running(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    scope.sandboxes.next_get_result = _FakeSandboxData(state="Running")

    assert await session.running() is True


@pytest.mark.asyncio
async def test_running_returns_false_when_stopped(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    scope.sandboxes.next_get_result = _FakeSandboxData(state="Stopped")

    assert await session.running() is False


@pytest.mark.asyncio
async def test_running_returns_false_on_error(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    scope.sandboxes.get_error = ConnectionError("unavailable")

    assert await session.running() is False


# ===========================================================================
# Session: shutdown
# ===========================================================================


@pytest.mark.asyncio
async def test_shutdown_stops_when_pause_on_exit(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod, pause_on_exit=True)

    await session._shutdown_backend()

    assert scope.sandboxes.stop_calls == ["sb-123"]
    assert scope.sandboxes.delete_calls == []


@pytest.mark.asyncio
async def test_shutdown_deletes_when_not_pause(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod, pause_on_exit=False)

    await session._shutdown_backend()

    assert scope.sandboxes.delete_calls == ["sb-123"]
    assert scope.sandboxes.stop_calls == []


# ===========================================================================
# Session: resolve_exposed_port
# ===========================================================================


@pytest.mark.asyncio
async def test_resolve_exposed_port_parses_https_url(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod, exposed_ports=(8080,))
    scope.sandboxes.next_ports = [
        _FakePort(port=8080, url="https://sb-123-8080.westus2.azuredevcompute.io"),
    ]

    endpoint = await session._resolve_exposed_port(8080)

    assert endpoint.host == "sb-123-8080.westus2.azuredevcompute.io"
    assert endpoint.port == 443
    assert endpoint.tls is True


@pytest.mark.asyncio
async def test_resolve_exposed_port_parses_http_url(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod, exposed_ports=(3000,))
    scope.sandboxes.next_ports = [
        _FakePort(port=3000, url="http://sb-123-3000.local:3000"),
    ]

    endpoint = await session._resolve_exposed_port(3000)

    assert endpoint.host == "sb-123-3000.local"
    assert endpoint.port == 3000
    assert endpoint.tls is False


@pytest.mark.asyncio
async def test_resolve_exposed_port_not_found(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod, exposed_ports=(8080,))
    scope.sandboxes.next_ports = [_FakePort(port=9999, url="https://other.example.com")]

    with pytest.raises(ExposedPortUnavailableError):
        await session._resolve_exposed_port(8080)


@pytest.mark.asyncio
async def test_resolve_exposed_port_backend_error(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod, exposed_ports=(8080,))

    async def _raise(*a: Any, **kw: Any) -> None:
        raise ConnectionError("unavailable")

    scope.sandboxes.get_ports = _raise  # type: ignore[assignment]

    with pytest.raises(ExposedPortUnavailableError):
        await session._resolve_exposed_port(8080)


# ===========================================================================
# Session: persist_workspace / hydrate_workspace
# ===========================================================================


@pytest.mark.asyncio
async def test_persist_workspace_creates_tar_and_cleans_up(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    tar_bytes = _valid_tar_bytes()
    scope.sandboxes.next_exec_result = _FakeCommandResult(exit_code=0)
    scope.sandboxes.next_read_file_result = tar_bytes

    result = await session.persist_workspace()

    assert isinstance(result, io.BytesIO)
    assert result.read() == tar_bytes
    # Should have: (1) tar create, (2) rm cleanup
    assert len(scope.sandboxes.exec_calls) >= 2
    assert "tar" in scope.sandboxes.exec_calls[0]["command"]
    assert "rm -f" in scope.sandboxes.exec_calls[1]["command"]


@pytest.mark.asyncio
async def test_hydrate_workspace_uploads_and_extracts(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)
    tar_data = _valid_tar_bytes()
    scope.sandboxes.next_exec_result = _FakeCommandResult(exit_code=0)

    await session.hydrate_workspace(io.BytesIO(tar_data))

    # Should have written the tar file
    assert len(scope.sandboxes.write_file_calls) >= 1
    # Should have executed tar extract
    tar_extract_calls = [
        c for c in scope.sandboxes.exec_calls if "tar" in c["command"] and "-xf" in c["command"]
    ]
    assert len(tar_extract_calls) >= 1


@pytest.mark.asyncio
async def test_hydrate_workspace_rejects_unsafe_tar(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)

    # Create a tar with an absolute path member (unsafe)
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
async def test_prepare_workspace_root_calls_mkdir(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)

    await session._prepare_workspace_root()

    assert len(scope.sandboxes.mkdir_calls) == 1
    sid, path, parents = scope.sandboxes.mkdir_calls[0]
    assert sid == "sb-123"
    assert path == "/home/user/workspace"
    assert parents is True


@pytest.mark.asyncio
async def test_prepare_workspace_root_error(azure_mod: Any) -> None:
    session, scope = _make_session(azure_mod)

    async def _raise(*a: Any, **kw: Any) -> None:
        raise RuntimeError("permission denied")

    scope.sandboxes.mkdir = _raise  # type: ignore[assignment]

    with pytest.raises(WorkspaceStartError):
        await session._prepare_workspace_root()


# ===========================================================================
# Client: create
# ===========================================================================


@pytest.mark.asyncio
async def test_client_create_from_disk_image(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    scope = client._scope
    scope.sandboxes.next_create_result = _FakeSandboxData(id="sb-new-1")

    options = azure_mod.AzureSandboxClientOptions(disk_image_id="di-1")
    await client.create(options=options)

    assert len(scope.sandboxes.create_from_disk_image_calls) == 1
    call = scope.sandboxes.create_from_disk_image_calls[0]
    assert call["source_disk_image"].id == "di-1"
    assert call["cpu"] == "1000m"
    assert call["memory"] == "1024Mi"


@pytest.mark.asyncio
async def test_client_create_from_snapshot(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    scope = client._scope
    scope.sandboxes.next_create_result = _FakeSandboxData(id="sb-snap-1")

    options = azure_mod.AzureSandboxClientOptions(snapshot_id="snap-1")
    await client.create(options=options)

    assert len(scope.sandboxes.create_from_snapshot_calls) == 1
    call = scope.sandboxes.create_from_snapshot_calls[0]
    assert call["snapshot_id"] == "snap-1"


@pytest.mark.asyncio
async def test_client_create_from_image(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    scope = client._scope
    scope.sandboxes.next_create_result = _FakeSandboxData(id="sb-img-1")

    options = azure_mod.AzureSandboxClientOptions(image="ubuntu:22.04")
    await client.create(options=options)

    # Should first create a disk image, then create sandbox from it
    assert len(scope.disk_images.create_calls) == 1
    assert scope.disk_images.create_calls[0]["base_image"] == "ubuntu:22.04"
    assert len(scope.sandboxes.create_from_disk_image_calls) == 1


@pytest.mark.asyncio
async def test_client_create_raises_without_source(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )

    options = azure_mod.AzureSandboxClientOptions()
    with pytest.raises(ValueError, match="disk_image_id, snapshot_id, or image"):
        await client.create(options=options)


# ===========================================================================
# Client: resume
# ===========================================================================


@pytest.mark.asyncio
async def test_client_resume_reconnects_running(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    scope = client._scope
    scope.sandboxes.next_get_result = _FakeSandboxData(id="sb-123", state="Running")

    state = _make_state(azure_mod, sandbox_id="sb-123")
    await client.resume(state)

    assert scope.sandboxes.get_calls == ["sb-123"]
    # No create calls — it reconnected
    assert len(scope.sandboxes.create_from_disk_image_calls) == 0


@pytest.mark.asyncio
async def test_client_resume_resumes_stopped(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    scope = client._scope
    scope.sandboxes.next_get_result = _FakeSandboxData(id="sb-123", state="Stopped")

    state = _make_state(azure_mod, sandbox_id="sb-123")
    await client.resume(state)

    assert scope.sandboxes.resume_calls == ["sb-123"]


@pytest.mark.asyncio
async def test_client_resume_recreates_on_not_found(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    scope = client._scope
    scope.sandboxes.get_error = Exception("not found")
    scope.sandboxes.next_create_result = _FakeSandboxData(id="sb-recreated")

    state = _make_state(azure_mod, sandbox_id="sb-gone")
    state.disk_image_id = "di-orig"
    await client.resume(state)

    assert len(scope.sandboxes.create_from_disk_image_calls) == 1


# ===========================================================================
# Client: delete
# ===========================================================================


@pytest.mark.asyncio
async def test_client_delete(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    scope = client._scope
    scope.sandboxes.next_create_result = _FakeSandboxData(id="sb-del")

    options = azure_mod.AzureSandboxClientOptions(disk_image_id="di-1")
    session = await client.create(options=options)
    await client.delete(session)

    # delete calls shutdown which calls delete on the backend
    assert len(scope.sandboxes.delete_calls) == 1


# ===========================================================================
# Client: close
# ===========================================================================


@pytest.mark.asyncio
async def test_client_close(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )

    await client.close()

    assert client._arm_client.closed is True


@pytest.mark.asyncio
async def test_client_context_manager(azure_mod: Any) -> None:
    async with azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    ) as client:
        pass

    assert client._arm_client.closed is True


# ===========================================================================
# Client: deserialize_session_state
# ===========================================================================


def test_deserialize_session_state(azure_mod: Any) -> None:
    client = azure_mod.AzureSandboxClient(
        credential=MagicMock(),
        subscription_id="sub-1",
        resource_group_name="rg-1",
        sandbox_group_id="sg-1",
    )
    state = _make_state(azure_mod, sandbox_id="sb-deser")
    payload = state.model_dump(mode="json")

    restored = client.deserialize_session_state(payload)

    assert restored.sandbox_id == "sb-deser"
    assert restored.type == "azuresandboxes"


# ===========================================================================
# _is_not_found_error helper
# ===========================================================================


def test_is_not_found_error_string_match(azure_mod: Any) -> None:
    assert azure_mod._is_not_found_error(Exception("resource not found")) is True
    assert azure_mod._is_not_found_error(Exception("404 Not Found")) is True
    assert azure_mod._is_not_found_error(Exception("something else")) is False
