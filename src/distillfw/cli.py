"""Command-line interface for distillfw."""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from distillfw.config import DistillationConfig
from distillfw.gcp.storage import StorageBackend
from distillfw.pipeline import DistillationPipeline
from distillfw.state import ORDERED_STAGES, StageName, TaskWorkspace

console = Console()


@click.group()
def main() -> None:
    """distillfw: Gemini-to-Gemma Distillation Framework on GCP."""


@main.command("init")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--prompts", "prompts_path", required=True, type=click.Path(exists=True), help="Path to user prompts file (.jsonl/.parquet/.csv).")
def init_cmd(config_path: str, prompts_path: str) -> None:
    """Initialize a new isolated distillation task workspace on GCS."""
    cfg = DistillationConfig.from_yaml(Path(config_path))
    pipeline = DistillationPipeline.init_task(config=cfg, prompts_path=prompts_path)
    console.print(f"[bold green]Initialized task workspace:[/bold green] {pipeline.workspace.task_uri}")


@main.command("resume")
@click.argument("task_uri", type=str)
@click.option(
    "--stop-after",
    type=click.Choice([s.value for s in ORDERED_STAGES]),
    default=None,
    help="Optional stage name to stop after.",
)
def resume_cmd(task_uri: str, stop_after: str | None) -> None:
    """Resume a distillation pipeline using ONLY its GCS task URI."""
    pipeline = DistillationPipeline.from_task_uri(task_uri)
    stop_stage = StageName(stop_after) if stop_after else None
    state = pipeline.resume(stop_after=stop_stage)
    console.print(f"[bold cyan]Task {state.task_id} status:[/bold cyan] {state.status.value}")


@main.command("run-stage")
@click.argument("task_uri", type=str)
@click.option(
    "--stage",
    required=True,
    type=click.Choice([s.value for s in ORDERED_STAGES]),
    help="Specific stage to execute.",
)
@click.option("--local-exec", is_flag=True, help="Force local single-node execution for trainer stage.")
def run_stage_cmd(task_uri: str, stage: str, local_exec: bool) -> None:
    """Execute a single pipeline stage against an existing GCS task URI."""
    pipeline = DistillationPipeline.from_task_uri(task_uri)
    result = pipeline.run_stage(StageName(stage), force_local_train=local_exec)
    console.print_json(json.dumps(result, default=str))


@main.command("status")
@click.argument("task_uri", type=str)
def status_cmd(task_uri: str) -> None:
    """Display live status and artifact cursors of a task from GCS."""
    ws = TaskWorkspace(task_uri=task_uri)
    state = ws.load_state()

    table = Table(title=f"Distillation Task: {state.task_id} ({state.status.value})")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Attempts", justify="right")
    table.add_column("Progress Cursor", style="green")

    for stage in ORDERED_STAGES:
        rec = state.stages[stage]
        table.add_row(
            stage.value,
            rec.status.value,
            str(rec.attempt_count),
            json.dumps(rec.progress_cursor),
        )
    console.print(table)


@main.command("list")
@click.argument("tasks_root_uri", type=str)
def list_cmd(tasks_root_uri: str) -> None:
    """List all tracked distillation tasks under a GCS prefix."""
    storage = StorageBackend()
    uris = storage.list_uris(tasks_root_uri)
    state_files = [u for u in uris if u.endswith("/task_state.json")]

    table = Table(title=f"Tracked Tasks in {tasks_root_uri}")
    table.add_column("Task ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Current Stage", style="yellow")
    table.add_column("Task URI", style="green")

    for state_uri in state_files:
        raw = storage.read_json(state_uri)
        table.add_row(
            raw.get("task_id", ""),
            raw.get("status", ""),
            str(raw.get("current_stage") or "-"),
            raw.get("task_uri", ""),
        )
    console.print(table)


if __name__ == "__main__":
    main()
