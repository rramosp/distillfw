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
from distillfw.state import (
    ORDERED_STAGES,
    StageName,
    TaskInitializationError,
    TaskResetAbortedError,
    TaskWorkspace,
    TaskWorkspaceExistsError,
)

console = Console()


@click.group()
def main() -> None:
    """distillfw: Gemini-to-Gemma Distillation Framework on GCP."""


@main.command("init")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Path to YAML config file.")
@click.option("--prompts", "prompts_path", required=True, type=click.Path(exists=True), help="Path to user prompts file (.jsonl/.parquet/.csv).")
@click.option(
    "--force-reset",
    is_flag=True,
    default=False,
    help="Completely erase existing contents in the target GCS workspace before initializing (prompts for confirmation).",
)
@click.pass_context
def init_cmd(ctx: click.Context, config_path: str, prompts_path: str, force_reset: bool) -> None:
    """Initialize a new isolated distillation task workspace on GCS."""
    cfg = DistillationConfig.from_yaml(Path(config_path))

    def _confirm_force_reset(target_uri: str, existing_objects: list[str]) -> bool:
        console.print(
            f"[bold red]WARNING:[/bold red] Target GCP path [bold]{target_uri}[/bold] "
            f"currently contains [bold]{len(existing_objects)}[/bold] existing object(s).\n"
            f"Proceeding with [bold]--force-reset[/bold] will [bold red]COMPLETELY AND PERMANENTLY ERASE[/bold red] "
            f"all existing contents under [bold]{target_uri}[/bold] before proceeding."
        )
        return click.confirm(
            f"Do you want to permanently erase all contents in {target_uri} and continue?",
            default=False,
        )

    try:
        pipeline = DistillationPipeline.init_task(
            config=cfg,
            prompts_path=prompts_path,
            force_reset=force_reset,
            confirm_reset_callback=_confirm_force_reset,
        )
    except TaskWorkspaceExistsError as exc:
        console.print(f"[bold yellow]Warning:[/bold yellow] {exc}")
        ctx.exit(0)
    except TaskResetAbortedError as exc:
        console.print(f"[bold yellow]Cancelled:[/bold yellow] {exc}")
        ctx.exit(0)
    except TaskInitializationError as exc:
        console.print(f"[bold red]Initialization Error:[/bold red]\n{exc}")
        ctx.exit(1)
    console.print(f"[bold green]Initialized task workspace:[/bold green] {pipeline.workspace.task_uri}")


def _print_evaluation_summary_tables(scorecard: dict) -> None:
    """Render Rich comparison tables for student metrics before and after training on train and test splits."""
    splits = scorecard.get("splits")
    if not isinstance(splits, dict):
        return

    for split_key, split_title in (
        ("test", "Evaluation Summary — TEST Split (Held-Out)"),
        ("train", "Evaluation Summary — TRAIN Split"),
    ):
        split_info = splits.get(split_key)
        if not isinstance(split_info, dict):
            continue
        before_m = split_info.get("before_training", {})
        after_m = split_info.get("after_training", {})
        delta_m = split_info.get("improvement", {})
        n_samples = split_info.get("num_samples", 0)

        table = Table(title=f"{split_title} (n={n_samples})")
        table.add_column("Metric", style="cyan")
        table.add_column("Before Training (Base)", justify="right", style="yellow")
        table.add_column("After Training (Distilled)", justify="right", style="green")
        table.add_column("Improvement (Delta)", justify="right", style="bold magenta")

        for key, label in (
            ("exact_match", "Exact Match"),
            ("rouge1", "ROUGE-1"),
            ("rouge2", "ROUGE-2"),
            ("bleu", "BLEU-4"),
        ):
            b_val = before_m.get("lexical_metrics", {}).get(key)
            a_val = after_m.get("lexical_metrics", {}).get(key)
            d_val = delta_m.get("lexical_metrics", {}).get(key)
            if b_val is not None and a_val is not None:
                sign = "+" if (d_val or 0.0) >= 0 else ""
                table.add_row(label, f"{b_val:.4f}", f"{a_val:.4f}", f"{sign}{d_val:.4f}")

        if "llm_judge" in before_m and "llm_judge" in after_m:
            for key, label, fmt in (
                ("mean_rubric_score", "LLM Judge Score (1-5)", ".3f"),
                ("win_or_tie_rate_vs_teacher", "Win/Tie Rate vs Teacher", ".4f"),
            ):
                b_val = before_m["llm_judge"].get(key)
                a_val = after_m["llm_judge"].get(key)
                d_val = delta_m.get("llm_judge", {}).get(key)
                if b_val is not None and a_val is not None:
                    sign = "+" if (d_val or 0.0) >= 0 else ""
                    table.add_row(
                        label,
                        f"{b_val:{fmt}}",
                        f"{a_val:{fmt}}",
                        f"{sign}{d_val:{fmt}}",
                    )

        if "system_metrics" in before_m and "system_metrics" in after_m:
            for key, label in (
                ("mean_output_tokens", "Output Length (Tokens)"),
                ("time_per_output_token_ms", "Time per Output Token (ms/tok)"),
                ("latency_mean_ms", "Mean Latency (ms)"),
                ("latency_p50_ms", "p50 Latency (ms)"),
                ("latency_p95_ms", "p95 Latency (ms)"),
            ):
                b_val = before_m["system_metrics"].get(key)
                a_val = after_m["system_metrics"].get(key)
                d_val = delta_m.get("system_metrics", {}).get(key)
                if b_val is not None and a_val is not None:
                    sign = "+" if (d_val or 0.0) >= 0 else ""
                    table.add_row(
                        label,
                        f"{b_val:.2f}",
                        f"{a_val:.2f}",
                        f"{sign}{d_val:.2f}",
                    )
        console.print(table)


@main.command("resume")
@click.argument("task_uri", type=str)
@click.option(
    "--stop-after",
    type=click.Choice([s.value for s in ORDERED_STAGES]),
    default=None,
    help="Optional stage name to stop after.",
)
@click.pass_context
def resume_cmd(ctx: click.Context, task_uri: str, stop_after: str | None) -> None:
    """Resume a distillation pipeline using ONLY its GCS task URI."""
    pipeline = DistillationPipeline.from_task_uri(task_uri)
    stop_stage = StageName(stop_after) if stop_after else None
    try:
        state = pipeline.resume(stop_after=stop_stage)
    except TaskInitializationError as exc:
        console.print(f"[bold red]Error:[/bold red]\n{exc}")
        ctx.exit(1)
    eval_rec = state.stages.get(StageName.MODEL_EVALUATOR)
    if eval_rec and isinstance(eval_rec.output_artifacts.get("scorecard"), dict):
        _print_evaluation_summary_tables(eval_rec.output_artifacts["scorecard"])
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
@click.pass_context
def run_stage_cmd(ctx: click.Context, task_uri: str, stage: str, local_exec: bool) -> None:
    """Execute a single pipeline stage against an existing GCS task URI."""
    pipeline = DistillationPipeline.from_task_uri(task_uri)
    try:
        result = pipeline.run_stage(StageName(stage), force_local_train=local_exec)
    except TaskInitializationError as exc:
        console.print(f"[bold red]Error:[/bold red]\n{exc}")
        ctx.exit(1)
    if isinstance(result, dict) and isinstance(result.get("scorecard"), dict):
        _print_evaluation_summary_tables(result["scorecard"])
    console.print_json(json.dumps(result, default=str))


@main.command("status")
@click.argument("task_uri", type=str)
def status_cmd(task_uri: str) -> None:
    """Display live status and artifact cursors of a task from GCS."""
    ws = TaskWorkspace(task_uri=task_uri)
    state = ws.refresh_vertex_training_status()

    table = Table(title=f"Distillation Task: {state.task_id} ({state.status.value})")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Progress Cursor", style="green")

    for stage in ORDERED_STAGES:
        rec = state.stages[stage]
        table.add_row(
            stage.value,
            rec.status.value,
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


@main.command("reset-training")
@click.argument("task_uri", type=str)
@click.option(
    "--config",
    "config_path",
    required=False,
    default=None,
    type=click.Path(exists=True),
    help="Optional path to updated YAML config file to sync to <task_uri>/config.yaml.",
)
def reset_training_cmd(task_uri: str, config_path: str | None = None) -> None:
    """Reset the model_trainer stage status back to PENDING in GCS task_state.json."""
    ws = TaskWorkspace(task_uri=task_uri)
    new_cfg = DistillationConfig.from_yaml(Path(config_path)) if config_path else None
    state = ws.reset_stage(StageName.MODEL_TRAINER, new_config=new_cfg)
    console.print(
        f"[bold green]Reset stage '{StageName.MODEL_TRAINER.value}' to PENDING[/bold green] "
        f"for task [bold]{state.task_id}[/bold] (overall status: {state.status.value})"
    )


@main.command("reset-eval")
@click.argument("task_uri", type=str)
@click.option(
    "--config",
    "config_path",
    required=False,
    default=None,
    type=click.Path(exists=True),
    help="Optional path to updated YAML config file to sync to <task_uri>/config.yaml.",
)
def reset_eval_cmd(task_uri: str, config_path: str | None = None) -> None:
    """Reset the model_evaluator stage status back to PENDING in GCS task_state.json."""
    ws = TaskWorkspace(task_uri=task_uri)
    new_cfg = DistillationConfig.from_yaml(Path(config_path)) if config_path else None
    state = ws.reset_stage(StageName.MODEL_EVALUATOR, new_config=new_cfg)
    console.print(
        f"[bold green]Reset stage '{StageName.MODEL_EVALUATOR.value}' to PENDING[/bold green] "
        f"for task [bold]{state.task_id}[/bold] (overall status: {state.status.value})"
    )


@main.command("ui")
@click.option(
    "--root-uri",
    default="gs://distillfw-storage/tasks",
    show_default=True,
    help="Default GCS root path (or local tasks directory) where distillation tasks are stored.",
)
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host address to bind the web UI server.",
)
@click.option(
    "--port",
    default=8080,
    type=int,
    show_default=True,
    help="Port to bind the web UI server.",
)
def ui_cmd(root_uri: str, host: str, port: int) -> None:
    """Launch the distillfw Web UI to inspect tasks, GCP resources, configs, logs, datasets, and evaluations."""
    from distillfw.ui.server import run_ui_server

    console.print(
        f"[bold green]Starting distillfw Web UI[/bold green] at [bold underline]http://{host}:{port}[/bold underline] "
        f"(default tasks root: [cyan]{root_uri}[/cyan])"
    )
    run_ui_server(host=host, port=port, default_root_uri=root_uri)


if __name__ == "__main__":
    main()

