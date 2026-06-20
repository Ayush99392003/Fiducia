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
DETAILS_CSV = REPO_ROOT / "output_details.csv"

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


def run(strategy_mode: str = "Smart") -> None:
    """Run predictions on the blind test set and write output.csv.

    Args:
        strategy_mode: "Smart" (dynamic), "A" (Direct), or "B" (CoT).
    """
    from code.llm import evaluate_claim
    from code.tracing import init_tracer

    strategy_label = f"Strategy_{strategy_mode}"
    session_id = f"predict-{strategy_mode}-{uuid.uuid4().hex[:8]}"
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
    details_rows: list[dict] = []
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

            recent_claims = int(history_row.get("last_90_days_claim_count", 0))

            if strategy_mode == "Smart":
                if recent_claims > 0 or history_flags != "none":
                    use_cot = True
                else:
                    use_cot = False
            else:
                use_cot = (strategy_mode == "B")

            evidence_req = _get_evidence_requirement(
                claim_object, evidence_df
            )

            try:
                raw_prediction = evaluate_claim(
                    claim_object=claim_object,
                    user_claim=user_claim,
                    evidence_requirement=evidence_req,
                    image_paths=image_paths,
                    image_ids=image_ids,
                    repo_root=REPO_ROOT,
                    use_cot=use_cot,
                    session_id=session_id,
                )
            except Exception as exc:
                console.print(
                    f"    [yellow]⚠️ Azure API rejected image for {user_id} (Invalid Format). Falling back to 'not_enough_information'[/yellow]"
                )
                errors += 1
                raw_prediction = {
                    "valid_image": False,
                    "claim_status": "not_enough_information",
                    "claim_status_justification": "Automated evaluation failed due to invalid image format or API error.",
                }

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

            details_rows.append({
                "user_id": user_id,
                "strategy_used": "Strategy B (CoT)" if use_cot else "Strategy A (Direct)",
                "input_tokens": raw_prediction.get("input_tokens", 0),
                "output_tokens": raw_prediction.get("output_tokens", 0),
                "latency_seconds": raw_prediction.get("latency_seconds", 0.0),
                **output_rows[-1]  # Include all original outputs as well
            })

            progress.advance(task)

    output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
    output_df.to_csv(OUTPUT_CSV, index=False)

    details_df = pd.DataFrame(details_rows)
    details_df.to_csv(DETAILS_CSV, index=False)

    console.print(
        Panel.fit(
            f"[bold green]Done![/bold green] "
            f"Wrote [cyan]{len(output_df)}[/cyan] rows to [cyan]{OUTPUT_CSV}[/cyan]\n"
            f"Wrote token details to [cyan]{DETAILS_CSV}[/cyan].\n"
            f"Errors: [red]{errors}[/red]"
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate output.csv for claims.csv."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="Smart",
        choices=["Smart", "A", "B"],
        help="Strategy to use (Smart, A, B). Default is Smart.",
    )
    args = parser.parse_args()
    
    # Backwards compatibility for --direct
    if hasattr(args, "direct") and args.direct:
        args.strategy = "A"
        
    run(strategy_mode=args.strategy)
