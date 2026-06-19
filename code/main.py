"""Final prediction entry point.

Reads `dataset/claims.csv`, runs the selected strategy (default: CoT),
and writes `output.csv` with the required column order.

Usage:
    uv run python code/main.py            # Strategy B (CoT, default)
    uv run python code/main.py --direct   # Strategy A (Direct)
"""

import argparse
import sys
import uuid
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CLAIMS_CSV = REPO_ROOT / "dataset" / "claims.csv"
USER_HISTORY = REPO_ROOT / "dataset" / "user_history.csv"
EVIDENCE_REQS = REPO_ROOT / "dataset" / "evidence_requirements.csv"
OUTPUT_CSV = REPO_ROOT / "output.csv"

# Required output column order per problem_statement.md
OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

console = Console()


# ---------------------------------------------------------------------------
# Re-use helpers from evaluation module
# ---------------------------------------------------------------------------

from code.evaluation.main import (  # noqa: E402
    _post_process,
    _get_evidence_requirement,
    _parse_set,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(use_cot: bool = True) -> None:
    """Run predictions on the blind test set and write output.csv.

    Args:
        use_cot: If True, uses Strategy B (CoT). Otherwise Strategy A.
    """
    from code.llm import evaluate_claim
    from code.tracing import init_tracer

    strategy_label = "B_CoT" if use_cot else "A_Direct"
    session_id = f"predict-{strategy_label}-{uuid.uuid4().hex[:8]}"
    init_tracer(session_id=session_id)

    console.print(
        Panel.fit(
            f"[bold cyan]Damage Claim Predictor — Strategy {strategy_label}[/bold cyan]\n"
            f"Session: {session_id}"
        )
    )

    claims_df = pd.read_csv(CLAIMS_CSV)
    history_df = pd.read_csv(USER_HISTORY).set_index("user_id")
    evidence_df = pd.read_csv(EVIDENCE_REQS)

    output_rows: list[dict] = []
    errors = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            "Processing claims ...", total=len(claims_df)
        )

        for _, row in claims_df.iterrows():
            user_id = str(row["user_id"])
            claim_object = str(row["claim_object"])
            user_claim = str(row["user_claim"])
            image_paths_raw = str(row["image_paths"])

            image_paths = [
                p.strip() for p in image_paths_raw.split(";") if p.strip()
            ]
            image_ids = [Path(p).stem for p in image_paths]

            history_row = (
                history_df.loc[user_id]
                if user_id in history_df.index
                else {}
            )
            history_summary = str(
                history_row.get("history_summary", "No prior history.")
            )
            history_flags = str(history_row.get("history_flags", "none"))

            evidence_req = _get_evidence_requirement(
                claim_object, evidence_df
            )

            try:
                raw_prediction = evaluate_claim(
                    claim_object=claim_object,
                    user_claim=user_claim,
                    history_summary=history_summary,
                    evidence_requirement=evidence_req,
                    image_paths=image_paths,
                    image_ids=image_ids,
                    repo_root=REPO_ROOT,
                    use_cot=use_cot,
                    session_id=session_id,
                )
            except Exception as exc:
                console.print(
                    f"[red]ERROR {user_id}: {exc}[/red]"
                )
                errors += 1
                progress.advance(task)
                continue

            prediction = _post_process(raw_prediction.copy(), history_flags)

            output_rows.append({
                "user_id": user_id,
                "image_paths": image_paths_raw,
                "user_claim": user_claim,
                "claim_object": claim_object,
                "evidence_standard_met": str(
                    prediction.get("evidence_standard_met", False)
                ).lower(),
                "evidence_standard_met_reason": prediction.get(
                    "evidence_standard_met_reason", ""
                ),
                "risk_flags": prediction.get("risk_flags", "none"),
                "issue_type": prediction.get("issue_type", "unknown"),
                "object_part": prediction.get("object_part", "unknown"),
                "claim_status": prediction.get(
                    "claim_status", "not_enough_information"
                ),
                "claim_status_justification": prediction.get(
                    "claim_status_justification", ""
                ),
                "supporting_image_ids": prediction.get(
                    "supporting_image_ids", "none"
                ),
                "valid_image": str(
                    prediction.get("valid_image", False)
                ).lower(),
                "severity": prediction.get("severity", "unknown"),
            })

            progress.advance(task)

    output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(OUTPUT_CSV, index=False)

    console.print(
        Panel.fit(
            f"[bold green]Done![/bold green] "
            f"Wrote [cyan]{len(output_df)}[/cyan] rows to "
            f"[cyan]{OUTPUT_CSV}[/cyan]. "
            f"Errors: [red]{errors}[/red]"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate output.csv for claims.csv."
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Use Strategy A (Direct) instead of Strategy B (CoT).",
    )
    args = parser.parse_args()
    run(use_cot=not args.direct)
