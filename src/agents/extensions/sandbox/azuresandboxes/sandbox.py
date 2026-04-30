"""Azure Sandboxes (Azure Dev Compute) implementation.

This module provides an Azure-backed sandbox client/session implementation using the new
``azure-sandbox`` (data plane) and ``azure-mgmt-sandbox`` (control plane) packages.

Both SDKs are synchronous; their calls are wrapped in ``asyncio.to_thread`` to keep the
extension's public surface async. Both packages are optional, so package-level exports
should guard imports of this module. Within this module, SDK imports are lazy so users
without the extra can still import the package.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import shlex
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from ....sandbox.entries import Mount
from ....sandbox.errors import (
    ExecTimeoutError,
    ExecTransportError,
    ExposedPortUnavailableError,
    WorkspaceArchiveReadError,
    WorkspaceArchiveWriteError,
    WorkspaceReadNotFoundError,
    WorkspaceStartError,
    WorkspaceWriteTypeError,
)
from ....sandbox.manifest import Manifest
from ....sandbox.session import SandboxSession, SandboxSessionState
from ....sandbox.session.base_sandbox_session import BaseSandboxSession
from ....sandbox.session.dependencies import Dependencies
from ....sandbox.session.manager import Instrumentation
from ....sandbox.session.runtime_helpers import RESOLVE_WORKSPACE_PATH_HELPER, RuntimeHelperScript
from ....sandbox.session.sandbox_client import BaseSandboxClient, BaseSandboxClientOptions
from ....sandbox.session.tar_workspace import shell_tar_exclude_args
from ....sandbox.snapshot import SnapshotBase, SnapshotSpec, resolve_snapshot
from ....sandbox.types import ExecResult, ExposedPortEndpoint, User
from ....sandbox.util.retry import (
    TRANSIENT_HTTP_STATUS_CODES,
    exception_chain_has_status_code,
    retry_async,
)
from ....sandbox.util.tar_utils import UnsafeTarMemberError, validate_tar_bytes
from ....sandbox.workspace_paths import (
    coerce_posix_path,
    posix_path_as_path,
    posix_path_for_error,
    sandbox_path_str,
)

DEFAULT_AZURE_SANDBOX_WORKSPACE_ROOT = "/home/user/workspace"
logger = logging.getLogger(__name__)

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _import_azure_sandbox_sdk() -> Any:
    """Lazily import azure.sandbox.SandboxClient, raising a clear error if missing."""
    try:
        from azure.sandbox import SandboxClient

        return SandboxClient
    except ImportError as e:
        raise ImportError(
            "AzureSandboxClient requires the optional `azure-sandbox` dependency.\n"
            "Install the azuresandboxes extra before using this sandbox backend:\n"
            "  pip install openai-agents[azuresandboxes]"
        ) from e


def _import_azure_mgmt_sandbox_sdk() -> Any:
    """Lazily import azure.mgmt.sandbox.SandboxGroupManagementClient."""
    try:
        from azure.mgmt.sandbox import SandboxGroupManagementClient

        return SandboxGroupManagementClient
    except ImportError as e:
        raise ImportError(
            "AzureSandboxClient requires the optional `azure-mgmt-sandbox` dependency.\n"
            "Install the azuresandboxes extra before using this sandbox backend:\n"
            "  pip install openai-agents[azuresandboxes]"
        ) from e


def _import_azure_core_not_found() -> type[BaseException]:
    """Lazily import azure.core.exceptions.ResourceNotFoundError; falls back to BaseException."""
    try:
        from azure.core.exceptions import ResourceNotFoundError

        return ResourceNotFoundError
    except ImportError:
        return type("_MissingResourceNotFoundError", (BaseException,), {})


def _validate_env_var_name(name: str) -> None:
    """Validate that an environment variable name is a safe POSIX shell identifier."""
    if not _ENV_VAR_NAME_RE.match(name):
        raise ValueError(
            f"invalid environment variable name {name!r}; must match ^[A-Za-z_][A-Za-z0-9_]*$"
        )


def _build_command_with_env(cmd_str: str, envs: dict[str, str]) -> str:
    """Wrap a shell command with env var assignments via ``env KEY=value ... sh -c <cmd>``.

    Validates env var names; uses ``shlex.quote`` on values and the inner command.
    """
    if not envs:
        return cmd_str
    for name in envs:
        _validate_env_var_name(name)
    env_assignments = " ".join(f"{name}={shlex.quote(value)}" for name, value in envs.items())
    return f"env {env_assignments} sh -c {shlex.quote(cmd_str)}"


class AzureSandboxResources(BaseModel):
    """Resource configuration for an Azure sandbox."""

    model_config = {"frozen": True}

    cpu: str = "1000m"
    memory: str = "2048Mi"


class AzureSandboxTimeouts(BaseModel):
    """Timeout configuration for Azure sandbox operations."""

    exec_timeout_unbounded_s: int = Field(default=300, ge=1)
    keepalive_s: int = Field(default=10, ge=1)
    cleanup_s: int = Field(default=30, ge=1)
    fast_op_s: int = Field(default=30, ge=1)
    file_upload_s: int = Field(default=300, ge=1)
    file_download_s: int = Field(default=300, ge=1)
    workspace_tar_s: int = Field(default=300, ge=1)


class AzureSandboxClientOptions(BaseSandboxClientOptions):
    """Client options for the Azure sandbox."""

    type: Literal["azuresandboxes"] = "azuresandboxes"
    disk: str | None = None
    snapshot_id: str | None = None
    preset: str | None = None
    env_vars: dict[str, str] | None = None
    pause_on_exit: bool = False
    cpu: str = "1000m"
    memory: str = "2048Mi"
    auto_suspend_seconds: int = Field(default=300, ge=1)
    timeouts: AzureSandboxTimeouts | dict[str, object] | None = None
    exposed_ports: tuple[int, ...] = ()
    labels: dict[str, str] | None = None
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None

    def __init__(
        self,
        disk: str | None = None,
        snapshot_id: str | None = None,
        preset: str | None = None,
        env_vars: dict[str, str] | None = None,
        pause_on_exit: bool = False,
        cpu: str = "1000m",
        memory: str = "2048Mi",
        auto_suspend_seconds: int = 300,
        timeouts: AzureSandboxTimeouts | dict[str, object] | None = None,
        exposed_ports: tuple[int, ...] = (),
        labels: dict[str, str] | None = None,
        entrypoint: list[str] | None = None,
        cmd: list[str] | None = None,
        *,
        type: Literal["azuresandboxes"] = "azuresandboxes",
    ) -> None:
        super().__init__(
            type=type,
            disk=disk,
            snapshot_id=snapshot_id,
            preset=preset,
            env_vars=env_vars,
            pause_on_exit=pause_on_exit,
            cpu=cpu,
            memory=memory,
            auto_suspend_seconds=auto_suspend_seconds,
            timeouts=timeouts,
            exposed_ports=exposed_ports,
            labels=labels,
            entrypoint=entrypoint,
            cmd=cmd,
        )


class AzureSandboxSessionState(SandboxSessionState):
    """Serializable state for an Azure-backed session."""

    type: Literal["azuresandboxes"] = "azuresandboxes"
    sandbox_id: str
    sandbox_group_id: str
    subscription_id: str
    resource_group_name: str
    disk: str | None = None
    snapshot_id: str | None = None
    preset: str | None = None
    base_env_vars: dict[str, str] = Field(default_factory=dict)
    pause_on_exit: bool = False
    cpu: str = "1000m"
    memory: str = "2048Mi"
    auto_suspend_seconds: int = 300
    timeouts: AzureSandboxTimeouts = Field(default_factory=AzureSandboxTimeouts)
    labels: dict[str, str] = Field(default_factory=dict)
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None


class AzureSandboxSession(BaseSandboxSession):
    """Azure Dev Compute sandbox session implementation."""

    state: AzureSandboxSessionState
    _client: Any  # azure.sandbox.SandboxClient
    _sandbox_group_id: str

    def __init__(
        self,
        *,
        state: AzureSandboxSessionState,
        client: Any,
        sandbox_group_id: str,
    ) -> None:
        self.state = state
        self._client = client
        self._sandbox_group_id = sandbox_group_id

    @classmethod
    def from_state(
        cls,
        state: AzureSandboxSessionState,
        *,
        client: Any,
        sandbox_group_id: str,
    ) -> AzureSandboxSession:
        return cls(state=state, client=client, sandbox_group_id=sandbox_group_id)

    @property
    def sandbox_id(self) -> str:
        return self.state.sandbox_id

    def _coerce_exec_timeout(self, timeout_s: float | None) -> float:
        if timeout_s is None:
            return float(self.state.timeouts.exec_timeout_unbounded_s)
        if timeout_s <= 0:
            return 0.001
        return float(timeout_s)

    async def _resolved_envs(self) -> dict[str, str]:
        manifest_envs = await self.state.manifest.environment.resolve()
        return {**self.state.base_env_vars, **manifest_envs}

    async def _exec_internal(
        self,
        *command: str | Path,
        timeout: float | None = None,
    ) -> ExecResult:
        cmd_str = shlex.join(str(c) for c in command)
        envs = await self._resolved_envs()
        cwd = sandbox_path_str(self.state.manifest.root)
        # The new SDK accepts a `working_directory` kwarg, but the Azure runC
        # backend rejects exec calls whose cwd does not yet exist (e.g. before
        # the workspace root is created). Prefix with `cd <root> && ...` instead
        # so the chdir happens inside the shell and the SDK doesn't pre-validate.
        env_inner = _build_command_with_env(cmd_str, envs)
        final_cmd = f"cd {shlex.quote(cwd)} && {env_inner}"
        caller_timeout = self._coerce_exec_timeout(timeout)

        try:
            result_dict = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.exec,
                    self.state.sandbox_id,
                    self._sandbox_group_id,
                    final_cmd,
                ),
                timeout=caller_timeout,
            )
            return ExecResult(
                stdout=str(result_dict.get("stdout", "")).encode("utf-8", errors="replace"),
                stderr=str(result_dict.get("stderr", "")).encode("utf-8", errors="replace"),
                exit_code=int(result_dict.get("exitCode", 0)),
            )
        except asyncio.TimeoutError as e:
            raise ExecTimeoutError(command=command, timeout_s=timeout, cause=e) from e
        except Exception as e:
            if isinstance(e, ExecTimeoutError):
                raise
            raise ExecTransportError(command=command, cause=e) from e

    def supports_pty(self) -> bool:
        return False

    async def _resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        try:
            sandbox_data = await asyncio.to_thread(
                self._client.get_sandbox,
                self.state.sandbox_id,
                self._sandbox_group_id,
            )
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "azuresandboxes", "detail": "get_sandbox_failed"},
                cause=e,
            ) from e

        raw_ports = sandbox_data.get("ports") if isinstance(sandbox_data, dict) else None
        if not isinstance(raw_ports, list):
            raw_ports = []

        available_ports: list[int] = []
        matching: dict[str, Any] | None = None
        for entry in raw_ports:
            if not isinstance(entry, dict):
                continue
            entry_port = entry.get("port")
            try:
                entry_port_int = int(entry_port) if entry_port is not None else None
            except (TypeError, ValueError):
                entry_port_int = None
            if entry_port_int is None:
                continue
            available_ports.append(entry_port_int)
            if entry_port_int == port and matching is None:
                matching = entry

        if matching is None:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="port_not_found",
                context={"backend": "azuresandboxes", "available_ports": available_ports},
            )

        url = matching.get("url")
        if not isinstance(url, str) or not url:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "azuresandboxes", "detail": "invalid_port_url", "url": url},
            )

        try:
            split = urlsplit(url)
            host = split.hostname
            if host is None:
                raise ValueError("missing hostname")
            port_value = split.port or (443 if split.scheme == "https" else 80)
            return ExposedPortEndpoint(host=host, port=port_value, tls=split.scheme == "https")
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={
                    "backend": "azuresandboxes",
                    "detail": "invalid_port_url",
                    "url": url,
                },
                cause=e,
            ) from e

    async def _shutdown_backend(self) -> None:
        try:
            if self.state.pause_on_exit:
                await asyncio.to_thread(
                    self._client.stop_sandbox,
                    self.state.sandbox_id,
                    self._sandbox_group_id,
                )
            else:
                await asyncio.to_thread(
                    self._client.delete_sandbox,
                    self.state.sandbox_id,
                    self._sandbox_group_id,
                )
        except Exception:
            pass

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    async def _prepare_workspace_root(self) -> None:
        """Create the workspace root directory.

        Bypasses ``_exec_internal`` (which prefixes commands with
        ``cd <root> && ...``) because the workspace root may not exist yet.
        Calls the SDK directly with no working directory so the shell can
        ``mkdir -p`` the root from scratch.
        """
        root = sandbox_path_str(self.state.manifest.root)
        error_root = posix_path_for_error(root)
        try:
            result_dict = await asyncio.to_thread(
                self._client.exec,
                self.state.sandbox_id,
                self._sandbox_group_id,
                f"mkdir -p -- {shlex.quote(root)}",
            )
            exit_code = int(result_dict.get("exitCode", 0)) if isinstance(result_dict, dict) else 0
            if exit_code != 0:
                stderr_value = ""
                if isinstance(result_dict, dict):
                    stderr_value = str(result_dict.get("stderr", ""))
                raise WorkspaceStartError(
                    path=error_root,
                    context={"reason": "workspace_root_nonzero_exit", "stderr": stderr_value},
                )
        except WorkspaceStartError:
            raise
        except Exception as e:
            raise WorkspaceStartError(path=error_root, cause=e) from e

    async def _prepare_backend_workspace(self) -> None:
        await self._prepare_workspace_root()

    async def mkdir(
        self,
        path: Path | str,
        *,
        parents: bool = False,
        user: str | User | None = None,
    ) -> None:
        if user is not None:
            path = await self._check_mkdir_with_exec(path, parents=parents, user=user)
        else:
            path = await self._validate_path_access(path, for_write=True)
        if path == Path("/"):
            return
        try:
            if parents:
                # SDK ``mkdir`` does not accept ``create_parents``; emulate via shell.
                result = await self._exec_internal("mkdir", "-p", "--", sandbox_path_str(path))
                if result.exit_code != 0:
                    raise WorkspaceArchiveWriteError(
                        path=path,
                        context={
                            "reason": "mkdir_failed",
                            "stderr": result.stderr.decode("utf-8", errors="replace"),
                        },
                    )
            else:
                await asyncio.to_thread(
                    self._client.mkdir,
                    self.state.sandbox_id,
                    self._sandbox_group_id,
                    sandbox_path_str(path),
                )
        except WorkspaceArchiveWriteError:
            raise
        except Exception as e:
            raise WorkspaceArchiveWriteError(
                path=path,
                context={"reason": "mkdir_failed"},
                cause=e,
            ) from e

    async def read(self, path: Path | str, *, user: str | User | None = None) -> io.IOBase:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            workspace_path = await self._check_read_with_exec(path, user=user)
        else:
            workspace_path = await self._validate_path_access(path)

        try:
            data: bytes = await asyncio.to_thread(
                self._client.read_file,
                self.state.sandbox_id,
                self._sandbox_group_id,
                sandbox_path_str(workspace_path),
            )
            return io.BytesIO(data)
        except Exception as e:
            if _is_not_found_error(e):
                raise WorkspaceReadNotFoundError(path=error_path, cause=e) from e
            raise WorkspaceArchiveReadError(path=error_path, cause=e) from e

    async def write(
        self,
        path: Path | str,
        data: io.IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        error_path = posix_path_as_path(coerce_posix_path(path))
        if user is not None:
            await self._check_write_with_exec(path, user=user)

        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=error_path, actual_type=type(payload).__name__)

        workspace_path = await self._validate_path_access(path, for_write=True)
        try:
            await asyncio.to_thread(
                self._client.write_file,
                self.state.sandbox_id,
                self._sandbox_group_id,
                sandbox_path_str(workspace_path),
                bytes(payload),
                create_dirs=True,
            )
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=workspace_path, cause=e) from e

    async def running(self) -> bool:
        try:
            sandbox_data = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.get_sandbox,
                    self.state.sandbox_id,
                    self._sandbox_group_id,
                ),
                timeout=self.state.timeouts.keepalive_s,
            )
            if isinstance(sandbox_data, dict):
                return sandbox_data.get("state") == "Running"
            return False
        except Exception:
            return False

    def _tar_exclude_args(self) -> list[str]:
        return shell_tar_exclude_args(self._persist_workspace_skip_relpaths())

    @retry_async(
        retry_if=lambda exc, self, tar_cmd, tar_path: (
            exception_chain_has_status_code(exc, TRANSIENT_HTTP_STATUS_CODES)
        )
    )
    async def _run_persist_workspace_command(self, tar_cmd: str, tar_path: str) -> bytes:
        try:
            result_dict = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.exec,
                    self.state.sandbox_id,
                    self._sandbox_group_id,
                    tar_cmd,
                ),
                timeout=self.state.timeouts.workspace_tar_s,
            )
            exit_code = int(result_dict.get("exitCode", 0))
            stdout = str(result_dict.get("stdout", ""))
            stderr = str(result_dict.get("stderr", ""))
            if exit_code != 0:
                raise WorkspaceArchiveReadError(
                    path=self._workspace_root_path(),
                    context={"reason": "tar_failed", "output": stderr or stdout},
                )
            return await asyncio.to_thread(
                self._client.read_file,
                self.state.sandbox_id,
                self._sandbox_group_id,
                tar_path,
            )
        except WorkspaceArchiveReadError:
            raise
        except Exception as e:
            raise WorkspaceArchiveReadError(path=self._workspace_root_path(), cause=e) from e

    async def persist_workspace(self) -> io.IOBase:
        def _error_context_summary(error: WorkspaceArchiveReadError) -> dict[str, str]:
            summary = {"message": error.message}
            if error.cause is not None:
                summary["cause_type"] = type(error.cause).__name__
                summary["cause"] = str(error.cause)
            return summary

        root = self._workspace_root_path()
        tar_path = f"/tmp/sandbox-persist-{self.state.session_id.hex}.tar"
        excludes = " ".join(self._tar_exclude_args())
        tar_cmd = (
            f"tar {excludes} -C {shlex.quote(root.as_posix())} -cf {shlex.quote(tar_path)} ."
        ).strip()

        unmounted_mounts: list[tuple[Mount, Path]] = []
        unmount_error: WorkspaceArchiveReadError | None = None
        for mount_entry, mount_path in self.state.manifest.ephemeral_mount_targets():
            try:
                await mount_entry.mount_strategy.teardown_for_snapshot(
                    mount_entry, self, mount_path
                )
            except Exception as e:
                unmount_error = WorkspaceArchiveReadError(path=root, cause=e)
                break
            unmounted_mounts.append((mount_entry, mount_path))

        snapshot_error: WorkspaceArchiveReadError | None = None
        raw: bytes | None = None
        if unmount_error is None:
            try:
                raw = await self._run_persist_workspace_command(tar_cmd, tar_path)
            except WorkspaceArchiveReadError as e:
                snapshot_error = e
            finally:
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self._client.exec,
                            self.state.sandbox_id,
                            self._sandbox_group_id,
                            f"rm -f -- {shlex.quote(tar_path)}",
                        ),
                        timeout=self.state.timeouts.cleanup_s,
                    )
                except Exception:
                    pass

        remount_error: WorkspaceArchiveReadError | None = None
        for mount_entry, mount_path in reversed(unmounted_mounts):
            try:
                await mount_entry.mount_strategy.restore_after_snapshot(
                    mount_entry, self, mount_path
                )
            except Exception as e:
                current_error = WorkspaceArchiveReadError(path=root, cause=e)
                if remount_error is None:
                    remount_error = current_error
                    if unmount_error is not None:
                        remount_error.context["earlier_unmount_error"] = _error_context_summary(
                            unmount_error
                        )
                else:
                    additional_remount_errors = remount_error.context.setdefault(
                        "additional_remount_errors",
                        [],
                    )
                    assert isinstance(additional_remount_errors, list)
                    additional_remount_errors.append(_error_context_summary(current_error))

        if remount_error is not None:
            if snapshot_error is not None:
                remount_error.context["snapshot_error_before_remount_corruption"] = (
                    _error_context_summary(snapshot_error)
                )
            raise remount_error
        if unmount_error is not None:
            raise unmount_error
        if snapshot_error is not None:
            raise snapshot_error

        assert raw is not None
        return io.BytesIO(raw)

    async def hydrate_workspace(self, data: io.IOBase) -> None:
        root = self._workspace_root_path()
        tar_path = f"/tmp/sandbox-hydrate-{self.state.session_id.hex}.tar"
        payload = data.read()
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        if not isinstance(payload, bytes | bytearray):
            raise WorkspaceWriteTypeError(path=Path(tar_path), actual_type=type(payload).__name__)

        try:
            validate_tar_bytes(bytes(payload))
        except UnsafeTarMemberError as e:
            raise WorkspaceArchiveWriteError(
                path=root,
                context={
                    "reason": "unsafe_or_invalid_tar",
                    "member": e.member,
                    "detail": str(e),
                },
                cause=e,
            ) from e

        try:
            await self.mkdir(root, parents=True)
            await asyncio.to_thread(
                self._client.write_file,
                self.state.sandbox_id,
                self._sandbox_group_id,
                tar_path,
                bytes(payload),
                create_dirs=True,
            )
            result_dict = await asyncio.wait_for(
                asyncio.to_thread(
                    self._client.exec,
                    self.state.sandbox_id,
                    self._sandbox_group_id,
                    f"tar -C {shlex.quote(root.as_posix())} -xf {shlex.quote(tar_path)}",
                ),
                timeout=self.state.timeouts.workspace_tar_s,
            )
            exit_code = int(result_dict.get("exitCode", 0))
            stdout = str(result_dict.get("stdout", ""))
            stderr = str(result_dict.get("stderr", ""))
            if exit_code != 0:
                raise WorkspaceArchiveWriteError(
                    path=root,
                    context={
                        "reason": "tar_extract_failed",
                        "output": stderr or stdout,
                    },
                )
        except WorkspaceArchiveWriteError:
            raise
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=root, cause=e) from e
        finally:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self._client.exec,
                        self.state.sandbox_id,
                        self._sandbox_group_id,
                        f"rm -f -- {shlex.quote(tar_path)}",
                    ),
                    timeout=self.state.timeouts.cleanup_s,
                )
            except Exception:
                pass


def _is_not_found_error(exc: BaseException) -> bool:
    """Check if an exception represents a 404 / not-found error."""
    not_found_cls = _import_azure_core_not_found()
    if isinstance(exc, not_found_cls):
        return True
    try:
        from httpx import HTTPStatusError

        if isinstance(exc, HTTPStatusError) and exc.response.status_code == 404:
            return True
    except ImportError:
        pass
    text = str(exc).lower()
    return "not found" in text or "404" in text


class AzureSandboxClient(BaseSandboxClient[AzureSandboxClientOptions]):
    """Azure Dev Compute sandbox client managing sandbox lifecycle via the new SDKs."""

    backend_id = "azuresandboxes"
    _instrumentation: Instrumentation

    def __init__(
        self,
        *,
        credential: Any = None,
        subscription_id: str,
        resource_group_name: str,
        sandbox_group_id: str,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        SandboxClient = _import_azure_sandbox_sdk()
        SandboxGroupManagementClient = _import_azure_mgmt_sandbox_sdk()

        self._client = SandboxClient(
            resource_group=resource_group_name,
            subscription_id=subscription_id,
            credential=credential,
        )
        self._mgmt_client = SandboxGroupManagementClient(
            resource_group=resource_group_name,
            subscription_id=subscription_id,
            credential=credential,
        )
        self._sandbox_group_id = sandbox_group_id
        self._subscription_id = subscription_id
        self._resource_group_name = resource_group_name
        self._instrumentation = instrumentation or Instrumentation()
        self._dependencies = dependencies

    async def create(
        self,
        *,
        snapshot: SnapshotSpec | SnapshotBase | None = None,
        manifest: Manifest | None = None,
        options: AzureSandboxClientOptions,
    ) -> SandboxSession:
        if manifest is None:
            manifest = Manifest(root=DEFAULT_AZURE_SANDBOX_WORKSPACE_ROOT)

        timeouts_in = options.timeouts
        if isinstance(timeouts_in, AzureSandboxTimeouts):
            timeouts = timeouts_in
        elif timeouts_in is None:
            timeouts = AzureSandboxTimeouts()
        else:
            timeouts = AzureSandboxTimeouts.model_validate(timeouts_in)

        if not (options.disk or options.snapshot_id or options.preset):
            raise ValueError(
                "AzureSandboxClientOptions requires one of: disk, snapshot_id, or preset"
            )

        session_id = uuid.uuid4()
        create_kwargs: dict[str, Any] = {
            "cpu": options.cpu,
            "memory": options.memory,
            "auto_suspend_seconds": options.auto_suspend_seconds,
        }
        if options.disk is not None:
            create_kwargs["disk"] = options.disk
        if options.snapshot_id is not None:
            create_kwargs["snapshot_id"] = options.snapshot_id
        if options.preset is not None:
            create_kwargs["preset"] = options.preset
        if options.labels:
            create_kwargs["labels"] = options.labels
        if options.env_vars:
            create_kwargs["environment"] = options.env_vars
        if options.exposed_ports:
            create_kwargs["ports"] = [{"port": p} for p in options.exposed_ports]
        if options.entrypoint is not None:
            create_kwargs["entrypoint"] = options.entrypoint
        if options.cmd is not None:
            create_kwargs["cmd"] = options.cmd

        sandbox_data = await asyncio.to_thread(
            self._client.create_sandbox,
            self._sandbox_group_id,
            **create_kwargs,
        )
        sandbox_id = sandbox_data.get("id") if isinstance(sandbox_data, dict) else None
        if not isinstance(sandbox_id, str):
            raise ValueError(f"Azure sandbox create response missing 'id': {sandbox_data!r}")

        snapshot_instance = resolve_snapshot(snapshot, str(session_id))
        state = AzureSandboxSessionState(
            session_id=session_id,
            manifest=manifest,
            snapshot=snapshot_instance,
            sandbox_id=sandbox_id,
            sandbox_group_id=self._sandbox_group_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            disk=options.disk,
            snapshot_id=options.snapshot_id,
            preset=options.preset,
            base_env_vars=dict(options.env_vars or {}),
            pause_on_exit=options.pause_on_exit,
            cpu=options.cpu,
            memory=options.memory,
            auto_suspend_seconds=options.auto_suspend_seconds,
            timeouts=timeouts,
            exposed_ports=options.exposed_ports,
            labels=dict(options.labels or {}),
            entrypoint=options.entrypoint,
            cmd=options.cmd,
        )
        inner = AzureSandboxSession.from_state(
            state,
            client=self._client,
            sandbox_group_id=self._sandbox_group_id,
        )
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def close(self) -> None:
        """Close both the data plane and control plane clients."""
        try:
            await asyncio.to_thread(self._client.close)
        finally:
            await asyncio.to_thread(self._mgmt_client.close)

    async def __aenter__(self) -> AzureSandboxClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def delete(self, session: SandboxSession) -> SandboxSession:
        inner = session._inner
        if not isinstance(inner, AzureSandboxSession):
            raise TypeError("AzureSandboxClient.delete expects an AzureSandboxSession")
        try:
            await inner.shutdown()
        except Exception:
            pass
        return session

    async def resume(
        self,
        state: SandboxSessionState,
    ) -> SandboxSession:
        if not isinstance(state, AzureSandboxSessionState):
            raise TypeError("AzureSandboxClient.resume expects an AzureSandboxSessionState")

        reconnected = False
        try:
            sandbox_data = await asyncio.to_thread(
                self._client.get_sandbox,
                state.sandbox_id,
                self._sandbox_group_id,
            )
            sandbox_state = sandbox_data.get("state") if isinstance(sandbox_data, dict) else None
            if sandbox_state == "Running":
                reconnected = True
            elif sandbox_state == "Stopped":
                await asyncio.to_thread(
                    self._client.resume_sandbox,
                    state.sandbox_id,
                    self._sandbox_group_id,
                )
                reconnected = True
        except Exception as e:
            logger.debug("Azure sandbox get_sandbox() failed, will recreate: %s", e)

        if not reconnected:
            if not (state.disk or state.snapshot_id or state.preset):
                raise ValueError(
                    "Cannot recreate Azure sandbox: no disk, snapshot_id, or preset in state"
                )
            recreate_kwargs: dict[str, Any] = {
                "cpu": state.cpu,
                "memory": state.memory,
                "auto_suspend_seconds": state.auto_suspend_seconds,
            }
            if state.disk is not None:
                recreate_kwargs["disk"] = state.disk
            if state.snapshot_id is not None:
                recreate_kwargs["snapshot_id"] = state.snapshot_id
            if state.preset is not None:
                recreate_kwargs["preset"] = state.preset
            if state.labels:
                recreate_kwargs["labels"] = state.labels
            if state.base_env_vars:
                recreate_kwargs["environment"] = state.base_env_vars
            if state.exposed_ports:
                recreate_kwargs["ports"] = [{"port": p} for p in state.exposed_ports]
            if state.entrypoint is not None:
                recreate_kwargs["entrypoint"] = state.entrypoint
            if state.cmd is not None:
                recreate_kwargs["cmd"] = state.cmd

            new_sandbox = await asyncio.to_thread(
                self._client.create_sandbox,
                self._sandbox_group_id,
                **recreate_kwargs,
            )
            new_id = new_sandbox.get("id") if isinstance(new_sandbox, dict) else None
            if not isinstance(new_id, str):
                raise ValueError(f"Azure sandbox create response missing 'id': {new_sandbox!r}")
            state.sandbox_id = new_id
            state.workspace_root_ready = False

        inner = AzureSandboxSession.from_state(
            state,
            client=self._client,
            sandbox_group_id=self._sandbox_group_id,
        )
        inner._set_start_state_preserved(reconnected, system=reconnected)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return AzureSandboxSessionState.model_validate(payload)

    async def list_sandbox_groups(self) -> list[dict[str, Any]]:
        """List sandbox groups in the configured resource group."""
        result = await asyncio.to_thread(self._mgmt_client.list_groups)
        return list(result) if result else []

    async def get_sandbox_group(self, name: str | None = None) -> dict[str, Any]:
        """Get a sandbox group by name (defaults to the client's configured sandbox group)."""
        target = name if name is not None else self._sandbox_group_id
        return await asyncio.to_thread(self._mgmt_client.get_group, target)

    async def create_sandbox_group(
        self,
        location: str,
        *,
        name: str | None = None,
        identity: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        **properties: Any,
    ) -> dict[str, Any]:
        """Create or update a sandbox group in the configured resource group."""
        target = name if name is not None else self._sandbox_group_id
        create_kwargs: dict[str, Any] = {}
        if identity is not None:
            create_kwargs["identity"] = identity
        if tags is not None:
            create_kwargs["tags"] = tags
        create_kwargs.update(properties)
        return await asyncio.to_thread(
            self._mgmt_client.create_group,
            target,
            location,
            **create_kwargs,
        )

    async def delete_sandbox_group(self, name: str | None = None) -> None:
        """Delete a sandbox group by name (defaults to the client's configured sandbox group)."""
        target = name if name is not None else self._sandbox_group_id
        await asyncio.to_thread(self._mgmt_client.delete_group, target)

    async def ensure_sandbox_group(
        self,
        location: str,
        *,
        name: str | None = None,
        identity: dict[str, Any] | None = None,
        tags: dict[str, str] | None = None,
        **properties: Any,
    ) -> dict[str, Any]:
        """Idempotently create the sandbox group if missing; return its current data."""
        target = name if name is not None else self._sandbox_group_id
        not_found_cls = _import_azure_core_not_found()
        try:
            return await asyncio.to_thread(self._mgmt_client.get_group, target)
        except not_found_cls:
            return await self.create_sandbox_group(
                location,
                name=target,
                identity=identity,
                tags=tags,
                **properties,
            )


__all__ = [
    "DEFAULT_AZURE_SANDBOX_WORKSPACE_ROOT",
    "AzureSandboxResources",
    "AzureSandboxClient",
    "AzureSandboxClientOptions",
    "AzureSandboxSession",
    "AzureSandboxSessionState",
    "AzureSandboxTimeouts",
]
