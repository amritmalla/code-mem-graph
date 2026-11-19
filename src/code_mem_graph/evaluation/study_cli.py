"""Offline study preparation commands; deliberately no execution entry point."""

import json
from pathlib import Path

import click

from .study import (
    CaseCommitment,
    PilotEvidence,
    StudyDesign,
    commit_case,
    load_study,
    lock_policy,
    register_study,
    schedule,
    study_plan,
    verify_case,
)


def _read(path, model):
    return model.model_validate_json(path.read_bytes())


def _emit(value):
    click.echo(json.dumps(value, ensure_ascii=False, indent=2))


def _checked(action):
    try:
        return action()
    except (ValueError, OSError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc


@click.group("study")
def study_group():
    """Plan and freeze a broader comparison without making model calls."""


@study_group.command("schema")
@click.argument("kind", type=click.Choice(["design", "pilot", "case"]))
def schema_command(kind):
    _emit(
        {"design": StudyDesign, "pilot": PilotEvidence, "case": CaseCommitment}[
            kind
        ].model_json_schema()
    )


@study_group.command("plan")
@click.argument("design", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pilot", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def plan_command(design, pilot):
    """Report missing inputs, conditional sample sizes and resource envelope."""
    _emit(
        _checked(
            lambda: study_plan(
                _read(design, StudyDesign),
                _read(pilot, PilotEvidence) if pilot else None,
            )
        )
    )


@study_group.command("commit-case")
@click.argument(
    "artifact", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option("--case-id", required=True)
@click.option(
    "--split",
    required=True,
    type=click.Choice(["calibration", "validation", "heldout"]),
)
@click.option("--domain", required=True, type=click.Choice(["service", "frontend"]))
@click.option("--repository-family", required=True)
@click.option("--scenario", multiple=True, required=True)
@click.option("--curator-reference", required=True)
@click.option(
    "--exposure", required=True, type=click.Choice(["inspected", "curator_only"])
)
@click.option("--synthetic/--real", default=None, required=True)
@click.option("--partial/--complete", default=None, required=True)
@click.option("--kind", type=click.Choice(["discovery", "change"]), default="discovery")
@click.option("--affected-component", multiple=True)
@click.option(
    "--output", required=True, type=click.Path(dir_okay=False, path_type=Path)
)
def commit_command(
    artifact,
    case_id,
    split,
    domain,
    repository_family,
    scenario,
    curator_reference,
    exposure,
    synthetic,
    partial,
    kind,
    affected_component,
    output,
):
    """Curator: seal a task/source artifact; omit source, prompt and rubric bodies."""

    def action():
        commitment = commit_case(
            artifact,
            case_id=case_id,
            split=split,
            domain=domain,
            repository_family=repository_family,
            scenarios=list(scenario),
            curator_reference=curator_reference,
            exposure=exposure,
            synthetic=synthetic,
            partial_snapshot=partial,
            kind=kind,
            affected_components=list(affected_component),
        )
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(commitment.model_dump_json(indent=2))
        return {"output": str(output), "model_calls": 0}

    _emit(_checked(action))


@study_group.command("register")
@click.argument("design", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--pilot", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option(
    "--allow-draft",
    is_flag=True,
    help="Explicitly freeze missing inputs; this cannot authorize trials.",
)
def register_command(design, pilot, output, allow_draft):
    _emit(
        _checked(
            lambda: register_study(
                _read(design, StudyDesign),
                output,
                _read(pilot, PilotEvidence) if pilot else None,
                allow_draft=allow_draft,
            )
        )
    )


@study_group.command("inspect")
@click.argument(
    "registration", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
def inspect_command(registration):
    _emit(_checked(lambda: load_study(registration)[2]))


@study_group.command("lock-policy")
@click.argument(
    "registration", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--recipe-id", required=True)
@click.option("--calibration-report-digest", multiple=True, required=True)
@click.option("--selection-review-reference", required=True)
def lock_command(
    registration, recipe_id, calibration_report_digest, selection_review_reference
):
    """Freeze a reviewed calibration selection; never authenticate model evidence."""
    _emit(
        _checked(
            lambda: lock_policy(
                registration,
                recipe_id=recipe_id,
                calibration_report_digests=list(calibration_report_digest),
                selection_review_reference=selection_review_reference,
            ).model_dump(mode="json")
        )
    )


@study_group.command("schedule")
@click.argument(
    "registration", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--split",
    required=True,
    type=click.Choice(["calibration", "validation", "heldout"]),
)
def schedule_command(registration, split):
    """Print seeded paired run identifiers after registration gates pass; run nothing."""
    _emit(_checked(lambda: schedule(registration, split=split)))


@study_group.command("verify-case")
@click.argument(
    "registration", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--case-id", required=True)
@click.option(
    "--artifact",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def verify_command(registration, case_id, artifact):
    _emit(_checked(lambda: verify_case(registration, case_id, artifact)))
