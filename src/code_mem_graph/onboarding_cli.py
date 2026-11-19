"""User-facing first-run commands for arbitrary supported repositories."""

import json
from pathlib import Path

import click
import yaml

from .onboarding import agent_configuration, initialize_repository, inspect_repository


def _render(report, as_json):
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return
    click.echo(f"Repository: {report['repository_root']}")
    click.echo(f"Status: {report['status']}")
    if report["repository_id"]:
        click.echo(
            f"Knowledge: {report['repository_id']} — {report['active_claims']} active claims"
        )
        click.echo(f"Read: {report['reading_view']} ({report['reading_view_status']})")
    if report["source_status"]:
        status = report["source_status"]
        click.echo(
            f"Source evidence: {status['fresh']} fresh, {status['stale']} stale, {status['unknown']} unknown"
        )
    if report["issues"]:
        click.echo("Needs attention: " + ", ".join(report["issues"]))
    click.echo(
        f"Knowledge gaps: {len(report['open_gaps'])}; claims needing review: {len(report['review_queue'])}"
    )
    if report["open_gaps"]:
        click.echo("Missing knowledge: " + ", ".join(report["open_gaps"]))
    if report["status"] == "not_initialized":
        click.echo(
            "Start with: code-mem-graph init --root <repository> --repository-id <stable-project-id>"
        )
    else:
        click.echo(
            "After source changes: code-mem-graph knowledge refresh --root <repository>"
        )
        click.echo(
            "Inspect gaps and reviews: code-mem-graph knowledge status --root <repository>"
        )
    click.echo("Agent connection configuration (merge into your client settings):")
    click.echo(json.dumps(report["mcp_config"], ensure_ascii=False, indent=2))
    click.echo(
        "Client connection is not verified. Install code-mem-graph in this Python environment before using the configuration."
    )
    click.echo(
        "Agent reads: orient, context, knowledge_item. Supply a positive token cap and tokenizer encoding; automatic sizing is not calibrated."
    )
    click.echo(
        "Missing knowledge stays unknown; readiness does not establish complete architecture coverage."
    )


@click.command("init")
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    show_default=True,
)
@click.option(
    "--repository-id",
    help="Stable project identity, required on first initialization; reused on later runs.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Return setup and agent configuration as JSON.",
)
def init_command(root, repository_id, as_json):
    """Build repository knowledge and show how to read it and connect an agent."""
    try:
        _render(initialize_repository(root, repository_id=repository_id), as_json)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.command("doctor")
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    show_default=True,
)
@click.option("--json", "as_json", is_flag=True)
def doctor_command(root, as_json):
    """Check knowledge freshness, reviews and reading files without changing them."""
    try:
        report = inspect_repository(root)
        _render(report, as_json)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise click.ClickException(str(exc)) from exc
    if not report["ready_for_reads"]:
        raise click.exceptions.Exit(1)


@click.command("mcp-config")
@click.option(
    "--root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    show_default=True,
)
def config_command(root):
    """Print an agent-client configuration pinned to this installation and repository."""
    try:
        click.echo(json.dumps(agent_configuration(root), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
