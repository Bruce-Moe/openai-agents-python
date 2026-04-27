"""Azure Sandboxes (Azure Dev Compute) implementation.

This module provides an Azure-backed sandbox client/session implementation using the
``adc-sdk-arm`` package for ARM-authenticated access to Azure Dev Compute microVMs.

The ``adc-sdk-arm`` and ``adc-sdk-core`` dependencies are optional, so package-level exports
should guard imports of this module. Within this module, ADC SDK imports are lazy so users
without the extra can still import the package.
"""

from __future__ import annotations

import asyncio
import io
import logging
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


def _import_adc_arm_sdk() -> tuple[Any, Any]:
    """Lazily import ADC ARM SDK classes, raising a clear error if missing."""
    try:
        from adc_arm import ArmAdcClient, ArmAdcClientOptions

        return ArmAdcClient, ArmAdcClientOptions
    except ImportError as e:
        raise ImportError(
            "AzureSandboxClient requires the optional `adc-sdk-arm` dependency.\n"
            "Install the azuresandboxes extra before using this sandbox backend:\n"
            "  pip install openai-agents[azuresandboxes]"
        ) from e


def _import_adc_core_models() -> tuple[Any, Any]:
    """Lazily import ADC core models for sandbox creation."""
    try:
        from adc_core.models.sandbox import SandboxSourceDiskImageById, SandboxSourceSnapshot

        return SandboxSourceDiskImageById, SandboxSourceSnapshot
    except ImportError as e:
        raise ImportError(
            "AzureSandboxClient requires the optional `adc-sdk-core` dependency.\n"
            "Install the azuresandboxes extra before using this sandbox backend:\n"
            "  pip install openai-agents[azuresandboxes]"
        ) from e


class AzureSandboxResources(BaseModel):
    """Resource configuration for an Azure sandbox."""

    model_config = {"frozen": True}

    cpu: str = "1000m"
    memory: str = "1024Mi"


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
    disk_image_id: str | None = None
    snapshot_id: str | None = None
    image: str | None = None
    env_vars: dict[str, str] | None = None
    pause_on_exit: bool = False
    cpu: str = "1000m"
    memory: str = "1024Mi"
    timeouts: AzureSandboxTimeouts | dict[str, object] | None = None
    exposed_ports: tuple[int, ...] = ()
    labels: dict[str, str] | None = None
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None

    def __init__(
        self,
        disk_image_id: str | None = None,
        snapshot_id: str | None = None,
        image: str | None = None,
        env_vars: dict[str, str] | None = None,
        pause_on_exit: bool = False,
        cpu: str = "1000m",
        memory: str = "1024Mi",
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
            disk_image_id=disk_image_id,
            snapshot_id=snapshot_id,
            image=image,
            env_vars=env_vars,
            pause_on_exit=pause_on_exit,
            cpu=cpu,
            memory=memory,
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
    disk_image_id: str | None = None
    snapshot_id: str | None = None
    image: str | None = None
    base_env_vars: dict[str, str] = Field(default_factory=dict)
    pause_on_exit: bool = False
    cpu: str = "1000m"
    memory: str = "1024Mi"
    timeouts: AzureSandboxTimeouts = Field(default_factory=AzureSandboxTimeouts)
    labels: dict[str, str] = Field(default_factory=dict)
    entrypoint: list[str] | None = None
    cmd: list[str] | None = None


class AzureSandboxSession(BaseSandboxSession):
    """Azure Dev Compute sandbox session implementation."""

    state: AzureSandboxSessionState
    _scope: Any  # SandboxGroupScope from adc_arm

    def __init__(self, *, state: AzureSandboxSessionState, scope: Any) -> None:
        self.state = state
        self._scope = scope

    @classmethod
    def from_state(
        cls,
        state: AzureSandboxSessionState,
        *,
        scope: Any,
    ) -> AzureSandboxSession:
        return cls(state=state, scope=scope)

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
        caller_timeout = self._coerce_exec_timeout(timeout)

        try:
            result = await asyncio.wait_for(
                self._scope.sandboxes.execute_shell_command(
                    self.state.sandbox_id,
                    cmd_str,
                    shell="/bin/sh",
                    environment=envs or None,
                    working_directory=cwd,
                ),
                timeout=caller_timeout,
            )
            return ExecResult(
                stdout=result.stdout.encode("utf-8", errors="replace"),
                stderr=result.stderr.encode("utf-8", errors="replace"),
                exit_code=result.exit_code,
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
            ports = await self._scope.sandboxes.get_ports(self.state.sandbox_id)
        except Exception as e:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="backend_unavailable",
                context={"backend": "azuresandboxes", "detail": "get_ports_failed"},
                cause=e,
            ) from e

        matching = [p for p in ports if p.port == port]
        if not matching:
            raise ExposedPortUnavailableError(
                port=port,
                exposed_ports=self.state.exposed_ports,
                reason="port_not_found",
                context={"backend": "azuresandboxes", "available_ports": [p.port for p in ports]},
            )

        url = matching[0].url
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
                await self._scope.sandboxes.stop(self.state.sandbox_id)
            else:
                await self._scope.sandboxes.delete(self.state.sandbox_id)
        except Exception:
            pass

    async def _validate_path_access(self, path: Path | str, *, for_write: bool = False) -> Path:
        return await self._validate_remote_path_access(path, for_write=for_write)

    def _runtime_helpers(self) -> tuple[RuntimeHelperScript, ...]:
        return (RESOLVE_WORKSPACE_PATH_HELPER,)

    async def _prepare_workspace_root(self) -> None:
        """Create the workspace root directory."""
        root = sandbox_path_str(self.state.manifest.root)
        error_root = posix_path_for_error(root)
        try:
            await self._scope.sandboxes.mkdir(
                self.state.sandbox_id,
                root,
                create_parents=True,
            )
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
            await self._scope.sandboxes.mkdir(
                self.state.sandbox_id,
                sandbox_path_str(path),
                create_parents=parents,
            )
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
            data: bytes = await self._scope.sandboxes.read_file(
                self.state.sandbox_id,
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
            await self._scope.sandboxes.write_file(
                self.state.sandbox_id,
                sandbox_path_str(workspace_path),
                bytes(payload),
                create_dirs=True,
            )
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=workspace_path, cause=e) from e

    async def running(self) -> bool:
        try:
            sandbox_data = await asyncio.wait_for(
                self._scope.sandboxes.get(self.state.sandbox_id),
                timeout=self.state.timeouts.keepalive_s,
            )
            return sandbox_data.state == "Running"
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
            result = await asyncio.wait_for(
                self._scope.sandboxes.execute_shell_command(
                    self.state.sandbox_id,
                    tar_cmd,
                    shell="/bin/sh",
                ),
                timeout=self.state.timeouts.workspace_tar_s,
            )
            if result.exit_code != 0:
                raise WorkspaceArchiveReadError(
                    path=self._workspace_root_path(),
                    context={"reason": "tar_failed", "output": result.stderr or result.stdout},
                )
            return await self._scope.sandboxes.read_file(
                self.state.sandbox_id,
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
                        self._scope.sandboxes.execute_shell_command(
                            self.state.sandbox_id,
                            f"rm -f -- {shlex.quote(tar_path)}",
                            shell="/bin/sh",
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
            await self._scope.sandboxes.write_file(
                self.state.sandbox_id,
                tar_path,
                bytes(payload),
                create_dirs=True,
            )
            result = await asyncio.wait_for(
                self._scope.sandboxes.execute_shell_command(
                    self.state.sandbox_id,
                    f"tar -C {shlex.quote(root.as_posix())} -xf {shlex.quote(tar_path)}",
                    shell="/bin/sh",
                ),
                timeout=self.state.timeouts.workspace_tar_s,
            )
            if result.exit_code != 0:
                raise WorkspaceArchiveWriteError(
                    path=root,
                    context={
                        "reason": "tar_extract_failed",
                        "output": result.stderr or result.stdout,
                    },
                )
        except WorkspaceArchiveWriteError:
            raise
        except Exception as e:
            raise WorkspaceArchiveWriteError(path=root, cause=e) from e
        finally:
            try:
                await asyncio.wait_for(
                    self._scope.sandboxes.execute_shell_command(
                        self.state.sandbox_id,
                        f"rm -f -- {shlex.quote(tar_path)}",
                        shell="/bin/sh",
                    ),
                    timeout=self.state.timeouts.cleanup_s,
                )
            except Exception:
                pass


def _is_not_found_error(exc: BaseException) -> bool:
    """Check if an exception represents a 404 / not-found error from the ADC SDK."""
    try:
        from httpx import HTTPStatusError

        if isinstance(exc, HTTPStatusError) and exc.response.status_code == 404:
            return True
    except ImportError:
        pass
    return "not found" in str(exc).lower() or "404" in str(exc)


class AzureSandboxClient(BaseSandboxClient[AzureSandboxClientOptions]):
    """Azure Dev Compute sandbox client managing sandbox lifecycle via ARM SDK."""

    backend_id = "azuresandboxes"
    _instrumentation: Instrumentation

    def __init__(
        self,
        *,
        credential: Any,
        subscription_id: str,
        resource_group_name: str,
        sandbox_group_id: str,
        data_plane_endpoint: str | None = None,
        instrumentation: Instrumentation | None = None,
        dependencies: Dependencies | None = None,
    ) -> None:
        ArmAdcClient, ArmAdcClientOptions = _import_adc_arm_sdk()

        options_kwargs: dict[str, Any] = {
            "subscription_id": subscription_id,
            "resource_group_name": resource_group_name,
            "credential": credential,
        }
        if data_plane_endpoint is not None:
            options_kwargs["data_plane_endpoint"] = data_plane_endpoint

        self._arm_client = ArmAdcClient(ArmAdcClientOptions(**options_kwargs))
        self._sandbox_group_id = sandbox_group_id
        self._subscription_id = subscription_id
        self._resource_group_name = resource_group_name
        self._scope = self._arm_client.for_sandbox_group(sandbox_group_id)
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

        session_id = uuid.uuid4()
        SandboxSourceDiskImageById, _ = _import_adc_core_models()

        sandbox_data: Any
        if options.snapshot_id:
            sandbox_data = await self._scope.sandboxes.create_from_snapshot(
                snapshot_id=options.snapshot_id,
                labels=options.labels,
                entrypoint=options.entrypoint,
                cmd=options.cmd,
            )
        elif options.disk_image_id:
            sandbox_data = await self._scope.sandboxes.create_from_disk_image(
                source_disk_image=SandboxSourceDiskImageById(id=options.disk_image_id),
                cpu=options.cpu,
                memory=options.memory,
                labels=options.labels,
                entrypoint=options.entrypoint,
                cmd=options.cmd,
                environment=options.env_vars,
            )
        elif options.image:
            disk_image = await self._scope.disk_images.create(
                labels=options.labels or {},
                base_image=options.image,
            )
            sandbox_data = await self._scope.sandboxes.create_from_disk_image(
                source_disk_image=SandboxSourceDiskImageById(id=disk_image.id),
                cpu=options.cpu,
                memory=options.memory,
                labels=options.labels,
                entrypoint=options.entrypoint,
                cmd=options.cmd,
                environment=options.env_vars,
            )
        else:
            raise ValueError(
                "AzureSandboxClientOptions requires one of: disk_image_id, snapshot_id, or image"
            )

        snapshot_instance = resolve_snapshot(snapshot, str(session_id))
        state = AzureSandboxSessionState(
            session_id=session_id,
            manifest=manifest,
            snapshot=snapshot_instance,
            sandbox_id=sandbox_data.id,
            sandbox_group_id=self._sandbox_group_id,
            subscription_id=self._subscription_id,
            resource_group_name=self._resource_group_name,
            disk_image_id=options.disk_image_id,
            snapshot_id=options.snapshot_id,
            image=options.image,
            base_env_vars=dict(options.env_vars or {}),
            pause_on_exit=options.pause_on_exit,
            cpu=options.cpu,
            memory=options.memory,
            timeouts=timeouts,
            exposed_ports=options.exposed_ports,
            labels=dict(options.labels or {}),
            entrypoint=options.entrypoint,
            cmd=options.cmd,
        )
        inner = AzureSandboxSession.from_state(state, scope=self._scope)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    async def close(self) -> None:
        """Close the underlying ARM client and release resources."""
        await self._arm_client.close()

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
            sandbox_data = await self._scope.sandboxes.get(state.sandbox_id)
            if sandbox_data.state == "Running":
                reconnected = True
            elif sandbox_data.state == "Stopped":
                await self._scope.sandboxes.resume(state.sandbox_id)
                reconnected = True
        except Exception as e:
            logger.debug("Azure sandbox get() failed, will recreate: %s", e)

        if not reconnected:
            SandboxSourceDiskImageById, _ = _import_adc_core_models()
            if state.snapshot_id:
                new_sandbox = await self._scope.sandboxes.create_from_snapshot(
                    snapshot_id=state.snapshot_id,
                    labels=state.labels or None,
                    entrypoint=state.entrypoint,
                    cmd=state.cmd,
                )
            elif state.disk_image_id:
                new_sandbox = await self._scope.sandboxes.create_from_disk_image(
                    source_disk_image=SandboxSourceDiskImageById(id=state.disk_image_id),
                    cpu=state.cpu,
                    memory=state.memory,
                    labels=state.labels or None,
                    entrypoint=state.entrypoint,
                    cmd=state.cmd,
                    environment=state.base_env_vars or None,
                )
            elif state.image:
                disk_image = await self._scope.disk_images.create(
                    labels=state.labels or {},
                    base_image=state.image,
                )
                new_sandbox = await self._scope.sandboxes.create_from_disk_image(
                    source_disk_image=SandboxSourceDiskImageById(id=disk_image.id),
                    cpu=state.cpu,
                    memory=state.memory,
                    labels=state.labels or None,
                    entrypoint=state.entrypoint,
                    cmd=state.cmd,
                    environment=state.base_env_vars or None,
                )
            else:
                raise ValueError(
                    "Cannot recreate Azure sandbox: "
                    "no disk_image_id, snapshot_id, or image in state"
                )
            state.sandbox_id = new_sandbox.id
            state.workspace_root_ready = False

        inner = AzureSandboxSession.from_state(state, scope=self._scope)
        inner._set_start_state_preserved(reconnected, system=reconnected)
        return self._wrap_session(inner, instrumentation=self._instrumentation)

    def deserialize_session_state(self, payload: dict[str, object]) -> SandboxSessionState:
        return AzureSandboxSessionState.model_validate(payload)


__all__ = [
    "DEFAULT_AZURE_SANDBOX_WORKSPACE_ROOT",
    "AzureSandboxResources",
    "AzureSandboxClient",
    "AzureSandboxClientOptions",
    "AzureSandboxSession",
    "AzureSandboxSessionState",
    "AzureSandboxTimeouts",
]
