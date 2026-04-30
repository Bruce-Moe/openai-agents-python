"""
Minimal Azure Sandboxes (Azure Dev Compute) example for manual validation.

Boots an Azure sandbox via the new ``azure-sandbox`` (data plane) and ``azure-mgmt-sandbox`` (control plane)
SDKs, optionally ensures the sandbox group exists via the management client, then
either runs a tiny agent loop or a no-OpenAI smoke test that just exercises the
SDK round trip (create, exec, read, write, delete).

Authentication uses ``azure-identity``'s ``DefaultAzureCredential`` so the script
picks up ``az login``, managed identity, environment variables, and so on without
extra wiring. The Azure-side parameters (subscription, resource group, sandbox
group) are stubbed so the script can be inspected before real values are wired
in; pass them as CLI flags or via ``AZURE_SUBSCRIPTION_ID`` / repository defaults.

Prerequisites:

    pip install azure-sandbox azure-mgmt-sandbox      # or install from source
    pip install openai-agents                          # this repo / branch
    az login                                           # for DefaultAzureCredential
    export OPENAI_API_KEY=...                          # only needed in agent mode

Usage examples::

    # Smoke test only (no OpenAI traffic): create -> exec -> read/write -> delete
    python -m examples.sandbox.extensions.azuresandboxes.azuresandboxes_runner \\
        --smoke-test \\
        --subscription-id <sub-id> \\
        --resource-group <rg> \\
        --sandbox-group <sg>

    # Agent demo (calls OpenAI): mounts a tiny manifest and asks one question.
    python -m examples.sandbox.extensions.azuresandboxes.azuresandboxes_runner \\
        --subscription-id <sub-id> \\
        --resource-group <rg> \\
        --sandbox-group <sg>

    # Ensure the sandbox group exists first (idempotent control-plane create).
    python -m examples.sandbox.extensions.azuresandboxes.azuresandboxes_runner \\
        --smoke-test --ensure-group --location westus2 \\
        --subscription-id <sub-id> --resource-group <rg> --sandbox-group <sg>
"""

from __future__ import annotations

import argparse
import asyncio
import io
import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

try:
    from agents.extensions.sandbox import (
        DEFAULT_AZURE_SANDBOX_WORKSPACE_ROOT,
        AzureSandboxClient,
        AzureSandboxClientOptions,
    )
except Exception as exc:  # pragma: no cover - optional extras path
    raise SystemExit(
        "Azure Sandboxes examples require the optional `azure-sandbox` and "
        "`azure-mgmt-sandbox` packages.\n"
        "Install them (both are still beta and not yet on PyPI) and retry."
    ) from exc

try:
    from azure.identity import DefaultAzureCredential
except Exception as exc:  # pragma: no cover - optional extras path
    raise SystemExit(
        "Azure Sandboxes examples require `azure-identity` for "
        "DefaultAzureCredential.\n"
        "Install it with: pip install azure-identity"
    ) from exc


# Stub placeholders. Replace with real values via CLI or env.
DEFAULT_SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
DEFAULT_RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "<your-resource-group>")
DEFAULT_SANDBOX_GROUP = os.environ.get("AZURE_SANDBOX_GROUP", "<your-sandbox-group>")
DEFAULT_DISK = "ubuntu"
DEFAULT_QUESTION = "Summarize this Azure sandbox workspace in 2 sentences."


def _build_manifest():
    """Build a tiny workspace manifest the agent can inspect (lazy-imports `agents`)."""
    from agents.sandbox import Manifest
    from examples.sandbox.misc.example_support import text_manifest

    manifest = text_manifest(
        {
            "README.md": (
                "# Azure Sandboxes Demo Workspace\n\n"
                "This workspace exists to validate the Azure Dev Compute sandbox backend "
                "end to end via the new `azure-sandbox` and `azure-mgmt-sandbox` SDKs.\n"
            ),
            "launch.md": (
                "# Launch\n\n"
                "- Customer: Contoso Logistics.\n"
                "- Goal: validate the Azure sandbox provider after the SDK migration.\n"
                "- Current status: data plane and control plane wired through "
                "`AzureSandboxClient`.\n"
            ),
            "tasks.md": (
                "# Tasks\n\n"
                "1. Inspect the workspace files.\n"
                "2. Summarize the setup and any notable status in two sentences.\n"
            ),
        }
    )
    return Manifest(root=DEFAULT_AZURE_SANDBOX_WORKSPACE_ROOT, entries=manifest.entries)


def _require_env(name: str) -> None:
    if os.environ.get(name):
        return
    raise SystemExit(f"{name} must be set before running this example.")


def _check_real_value(label: str, value: str) -> None:
    if not value or value.startswith("<your-"):
        raise SystemExit(
            f"{label} is unset or still a stub ({value!r}). Pass the real value via "
            f"the matching CLI flag or environment variable."
        )


async def _smoke_test(
    *,
    client: AzureSandboxClient,
    options: AzureSandboxClientOptions,
) -> None:
    """Exercise a minimal create -> exec -> file IO -> delete round trip."""
    print("[smoke] creating sandbox ...", flush=True)
    session = await client.create(options=options)
    try:
        sandbox_id = getattr(session._inner, "sandbox_id", "<unknown>")
        print(f"[smoke] sandbox id: {sandbox_id}", flush=True)

        # session.start() prepares the workspace root, installs runtime helpers,
        # and lays down the (empty) manifest. Without it, exec/read/write target
        # a workspace root that does not yet exist on the disk image.
        print("[smoke] starting session (workspace prep + runtime helpers) ...", flush=True)
        await session.start()

        print("[smoke] exec: uname -a", flush=True)
        result = await session.exec("uname -a", shell=True)
        print(
            f"[smoke] exit_code={result.exit_code}\n"
            f"[smoke] stdout={result.stdout.decode('utf-8', errors='replace').strip()!r}\n"
            f"[smoke] stderr={result.stderr.decode('utf-8', errors='replace').strip()!r}",
            flush=True,
        )

        target_file = f"{DEFAULT_AZURE_SANDBOX_WORKSPACE_ROOT}/hello.txt"
        print(f"[smoke] write {target_file}", flush=True)
        await session.write(target_file, io.BytesIO(b"hello azure sandbox\n"))

        print(f"[smoke] read {target_file}", flush=True)
        readback = await session.read(target_file)
        print(f"[smoke] readback={readback.read()!r}", flush=True)
    finally:
        print("[smoke] deleting sandbox ...", flush=True)
        await client.delete(session)


async def _agent_demo(
    *,
    client: AzureSandboxClient,
    options: AzureSandboxClientOptions,
    model: str,
    question: str,
    stream: bool,
) -> None:
    """Run a single-turn agent that inspects the manifest workspace via shell."""
    _require_env("OPENAI_API_KEY")

    # Lazy-imported so smoke-test mode does not require `openai` or the agent surface.
    from openai.types.responses import ResponseTextDeltaEvent

    from agents import ModelSettings, Runner
    from agents.run import RunConfig
    from agents.sandbox import SandboxAgent, SandboxRunConfig
    from examples.sandbox.misc.workspace_shell import WorkspaceShellCapability

    manifest = _build_manifest()
    agent = SandboxAgent(
        name="Azure Sandbox Assistant",
        model=model,
        instructions=(
            "Answer questions about the sandbox workspace. Inspect the files before "
            "answering and keep the response concise. Do not invent files or statuses "
            "that are not present in the workspace. Cite the file names you inspected."
        ),
        default_manifest=manifest,
        capabilities=[WorkspaceShellCapability()],
        model_settings=ModelSettings(tool_choice="required"),
    )

    run_config = RunConfig(
        sandbox=SandboxRunConfig(client=client, options=options),
        workflow_name="Azure Sandboxes example",
    )

    if not stream:
        result = await Runner.run(agent, question, run_config=run_config)
        print(result.final_output)
        return

    stream_result = Runner.run_streamed(agent, question, run_config=run_config)
    saw_text_delta = False
    async for event in stream_result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            if not saw_text_delta:
                print("assistant> ", end="", flush=True)
                saw_text_delta = True
            print(event.data.delta, end="", flush=True)
    if saw_text_delta:
        print()


async def main(
    *,
    subscription_id: str,
    resource_group: str,
    sandbox_group: str,
    disk: str | None,
    snapshot_id: str | None,
    preset: str | None,
    pause_on_exit: bool,
    ensure_group: bool,
    location: str | None,
    smoke_test: bool,
    model: str,
    question: str,
    stream: bool,
) -> None:
    _check_real_value("--subscription-id", subscription_id)
    _check_real_value("--resource-group", resource_group)
    _check_real_value("--sandbox-group", sandbox_group)

    if not (disk or snapshot_id or preset):
        # Default to a public ubuntu disk if no source was specified.
        disk = DEFAULT_DISK

    credential = DefaultAzureCredential()
    client = AzureSandboxClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        sandbox_group_id=sandbox_group,
    )

    options = AzureSandboxClientOptions(
        disk=disk,
        snapshot_id=snapshot_id,
        preset=preset,
        pause_on_exit=pause_on_exit,
    )

    try:
        if ensure_group:
            if not location:
                raise SystemExit("--ensure-group requires --location (e.g. westus2).")
            print(f"[group] ensuring sandbox group {sandbox_group!r} in {location} ...", flush=True)
            group = await client.ensure_sandbox_group(location=location)
            print(f"[group] sandbox group ready: {group!r}", flush=True)

        if smoke_test:
            await _smoke_test(client=client, options=options)
        else:
            await _agent_demo(
                client=client,
                options=options,
                model=model,
                question=question,
                stream=stream,
            )
    finally:
        await client.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--subscription-id",
        default=DEFAULT_SUBSCRIPTION_ID,
        help="Azure subscription ID (defaults to AZURE_SUBSCRIPTION_ID env var).",
    )
    parser.add_argument(
        "--resource-group",
        default=DEFAULT_RESOURCE_GROUP,
        help="Azure resource group containing the sandbox group "
        "(defaults to AZURE_RESOURCE_GROUP env var).",
    )
    parser.add_argument(
        "--sandbox-group",
        default=DEFAULT_SANDBOX_GROUP,
        help="Sandbox group name (defaults to AZURE_SANDBOX_GROUP env var).",
    )

    source = parser.add_argument_group("sandbox source (pick one)")
    source.add_argument("--disk", default=None, help="Public disk image name (e.g. 'ubuntu').")
    source.add_argument("--snapshot-id", default=None, help="Existing snapshot ID to boot from.")
    source.add_argument("--preset", default=None, help="Preset sandbox type (e.g. 'copilot').")

    group = parser.add_argument_group("sandbox group management")
    group.add_argument(
        "--ensure-group",
        action="store_true",
        default=False,
        help="Idempotently create the sandbox group if it does not already exist.",
    )
    group.add_argument(
        "--location",
        default=None,
        help="Azure region for --ensure-group (e.g. westus2). Required with --ensure-group.",
    )

    parser.add_argument(
        "--pause-on-exit",
        action="store_true",
        default=False,
        help="Stop (suspend) the sandbox on shutdown instead of deleting it.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Skip the OpenAI agent and run a minimal create/exec/read/write/delete round trip.",
    )
    parser.add_argument("--model", default="gpt-5.5", help="Model name for the agent demo.")
    parser.add_argument(
        "--question", default=DEFAULT_QUESTION, help="Prompt to send to the agent demo."
    )
    parser.add_argument(
        "--stream", action="store_true", default=False, help="Stream the agent response."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        main(
            subscription_id=args.subscription_id,
            resource_group=args.resource_group,
            sandbox_group=args.sandbox_group,
            disk=args.disk,
            snapshot_id=args.snapshot_id,
            preset=args.preset,
            pause_on_exit=args.pause_on_exit,
            ensure_group=args.ensure_group,
            location=args.location,
            smoke_test=args.smoke_test,
            model=args.model,
            question=args.question,
            stream=args.stream,
        )
    )
