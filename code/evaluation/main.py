"""Evaluation Orchestrator for the damage claim verification system.

Runs both Strategy A (Direct) and Strategy B (CoT) against the labeled
`dataset/sample_claims.csv` and prints a side-by-side accuracy and
cost comparison using the Rich library.

Usage:
    uv run python code/evaluation/main.py
"""

import sys
import uuid
import json
import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_CLAIMS = REPO_ROOT / "dataset" / "sample_claims.csv"
USER_HISTORY = REPO_ROOT / "dataset" / "user_history.csv"
EVIDENCE_REQS = REPO_ROOT / "dataset" / "evidence_requirements.csv"

# Cost assumptions (USD per 1k tokens) — adjust for your deployment.
COST_INPUT_PER_1K = 0.00015
COST_OUTPUT_PER_1K = 0.0006

console = Console()

# ---------------------------------------------------------------------------
# Graded fields and their scoring type
# ---------------------------------------------------------------------------

EXACT_FIELDS = [
    "evidence_standard_met",
    "valid_image",
    "claim_status",
    "issue_type",
    "object_part",
    "severity",
]

SET_FIELDS = [
    "risk_flags",
    "supporting_image_ids",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_set(value: str) -> set[str]:
    """Parse a semicolon-separated string into a normalised set.

    Args:
        value: Semicolon-separated string (e.g. 'img_1;img_2').

    Returns:
        Set of stripped, lower-cased tokens.
    """
    return {v.strip().lower() for v in str(value).split(";") if v.strip()}


def _post_process(
    prediction: dict,
    history_flags: str,
) -> dict:
    """Apply deterministic post-processing rules to a raw prediction.

    Specifically:
    - Inject user_history_risk and manual_review_required from the user
      profile flags.
    - Enforce output constraints based on claim_status.

    Args:
        prediction: Raw model output dict.
        history_flags: Semicolon-separated history_flags string for this
            user from user_history.csv.

    Returns:
        Post-processed prediction dict.
    """
    flags: set[str] = _parse_set(prediction.get("risk_flags", "none"))
    flags.discard("none")

    history_set = _parse_set(history_flags)
    if "user_history_risk" in history_set:
        flags.add("user_history_risk")
        flags.add("manual_review_required")
    if "manual_review_required" in history_set:
        flags.add("manual_review_required")

    prediction["risk_flags"] = (
        ";".join(sorted(flags)) if flags else "none"
    )

    # Enforce constraints based on claim_status
    status = prediction.get("claim_status", "")
    if status == "not_enough_information":
        prediction["severity"] = "unknown"
        prediction["issue_type"] = "unknown"
        prediction["supporting_image_ids"] = "none"
        prediction["evidence_standard_met"] = False

    return prediction


def _score_case(prediction: dict, ground_truth: dict) -> dict:
    """Score a single prediction against ground truth.

    Args:
        prediction: Post-processed model output dict.
        ground_truth: Expected values dict from sample_claims.csv.

    Returns:
        Dict with 'field_scores' (field -> 0 or 1), 'partial_score',
        and 'perfect_match' keys.
    """
    field_scores: dict[str, int] = {}

    for field in EXACT_FIELDS:
        pred_val = str(prediction.get(field, "")).strip().lower()
        gt_val = str(ground_truth.get(field, "")).strip().lower()
        field_scores[field] = int(pred_val == gt_val)

    for field in SET_FIELDS:
        pred_set = _parse_set(str(prediction.get(field, "")))
        gt_set = _parse_set(str(ground_truth.get(field, "")))
        field_scores[field] = int(pred_set == gt_set)

    total_fields = len(field_scores)
    matched = sum(field_scores.values())
    partial_score = matched / total_fields if total_fields else 0.0
    perfect_match = matched == total_fields

    return {
        "field_scores": field_scores,
        "partial_score": partial_score,
        "perfect_match": perfect_match,
    }


def _get_evidence_requirement(
    claim_object: str,
    evidence_df: pd.DataFrame,
) -> str:
    """Return the relevant evidence requirement text for a claim.

    Selects rules matching the claim_object (plus 'all' rules) and
    joins their minimum_image_evidence strings.

    Args:
        claim_object: 'car', 'laptop', or 'package'.
        evidence_df: Loaded evidence_requirements.csv DataFrame.

    Returns:
        Concatenated requirement strings, or a generic fallback.
    """
    rows = evidence_df[
        evidence_df["claim_object"].isin([claim_object, "all"])
    ]
    texts = rows["minimum_image_evidence"].dropna().tolist()
    return " ".join(texts) if texts else (
        "The claimed object and relevant part should be clearly visible."
    )


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(strategy_mode: str, limit: int | None = None) -> dict:
    """Run the full evaluation loop for one strategy.

    Args:
        strategy_mode: 'A' (Direct), 'B' (CoT), or 'Smart' (Dynamic Routing).
        limit: Optional maximum number of cases to evaluate.

    Returns:
        Dict with per-case results and aggregate metrics.
    """
    from code.llm import evaluate_claim
    from code.tracing import init_tracer

    strategy_label = f"Strategy_{strategy_mode}"
    session_id = f"eval-{strategy_label}-{uuid.uuid4().hex[:8]}"
    init_tracer(session_id=session_id)

    claims_df = pd.read_csv(SAMPLE_CLAIMS)
    if limit is not None:
        claims_df = claims_df.head(limit)
        
    history_df = pd.read_csv(USER_HISTORY).set_index("user_id")
    evidence_df = pd.read_csv(EVIDENCE_REQS)

    results: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0.0
    total_images = 0
    perfect_matches = 0
    field_score_sums: dict[str, float] = {
        f: 0.0 for f in EXACT_FIELDS + SET_FIELDS
    }

    for idx, row in claims_df.iterrows():
        user_id = str(row["user_id"])
        claim_object = str(row["claim_object"])
        user_claim = str(row["user_claim"])
        image_paths_raw = str(row["image_paths"])

        image_paths = [
            p.strip() for p in image_paths_raw.split(";") if p.strip()
        ]
        image_ids = [
            Path(p).stem for p in image_paths
        ]

        # User history context
        history_row = history_df.loc[user_id] if user_id in history_df.index else {}
        history_flags = str(history_row.get("history_flags", "none"))
        recent_claims = int(history_row.get("last_90_days_claim_count", 0))

        # Dynamic routing logic for Smart mode
        if strategy_mode == "Smart":
            # If they have recent claims or existing risk flags, use careful CoT
            if recent_claims > 0 or history_flags != "none":
                use_cot = True
            else:
                use_cot = False
        else:
            use_cot = (strategy_mode == "B")

        # Evidence requirement
        evidence_req = _get_evidence_requirement(claim_object, evidence_df)

        console.print(
            f"  [{idx + 1}/{len(claims_df)}] {user_id} | "
            f"{claim_object} | {len(image_paths)} image(s) [dim]({ 'CoT' if use_cot else 'Direct' })[/dim]...",
            style="dim",
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
                f"    [red]ERROR for {user_id}: {exc}[/red]"
            )
            continue

        prediction = _post_process(raw_prediction.copy(), history_flags)

        ground_truth = row.to_dict()
        score = _score_case(prediction, ground_truth)

        # Accumulate metrics
        in_tok = raw_prediction.get("input_tokens", 0)
        out_tok = raw_prediction.get("output_tokens", 0)
        lat = raw_prediction.get("latency_seconds", 0.0)
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        total_latency += lat
        total_images += len(image_paths)
        if score["perfect_match"]:
            perfect_matches += 1
        for field, val in score["field_scores"].items():
            field_score_sums[field] += val

        results.append({
            "user_id": user_id,
            "strategy": strategy_label,
            "partial_score": score["partial_score"],
            "perfect_match": score["perfect_match"],
            "field_scores": score["field_scores"],
            "prediction": prediction,
            "ground_truth": ground_truth,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_seconds": lat,
        })

    n = len(results) or 1
    cost = (
        total_input_tokens * COST_INPUT_PER_1K / 1000
        + total_output_tokens * COST_OUTPUT_PER_1K / 1000
    )

    return {
        "strategy": strategy_label,
        "session_id": session_id,
        "results": results,
        "n": n,
        "perfect_matches": perfect_matches,
        "perfect_match_pct": perfect_matches / n * 100,
        "avg_partial_score": sum(r["partial_score"] for r in results) / n * 100,
        "field_accuracy": {
            f: field_score_sums[f] / n * 100
            for f in EXACT_FIELDS + SET_FIELDS
        },
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_latency": total_latency,
        "avg_latency": total_latency / n,
        "total_images": total_images,
        "estimated_cost_usd": cost,
    }


def print_comparison(
    metrics_a: dict,
    metrics_b: dict,
) -> None:
    """Print a Rich side-by-side comparison dashboard.

    Args:
        metrics_a: Aggregate metrics for Strategy A.
        metrics_b: Aggregate metrics for Strategy B.
    """
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]EVALUATION RESULTS — Strategy Comparison[/bold cyan]",
            box=box.DOUBLE,
        )
    )

    # Accuracy table
    acc_table = Table(
        title="Accuracy",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    acc_table.add_column("Metric", style="bold")
    acc_table.add_column("Strategy A (Direct)", justify="center")
    acc_table.add_column("Strategy B (CoT)", justify="center")
    if "metrics_smart" in globals():
        acc_table.add_column("Strategy Smart", justify="center")

    def get_row(label, metrics_a, metrics_b, metrics_smart=None, fmt="{}"):
        row = [label, fmt.format(metrics_a), fmt.format(metrics_b)]
        if metrics_smart is not None:
            row.append(fmt.format(metrics_smart))
        return row

    acc_table.add_row(*get_row(
        "Perfect Match %",
        metrics_a['perfect_match_pct'],
        metrics_b['perfect_match_pct'],
        globals().get('metrics_smart', {}).get('perfect_match_pct'),
        "{:.1f}%"
    ))
    acc_table.add_row(*get_row(
        "Avg Partial Score %",
        metrics_a['avg_partial_score'],
        metrics_b['avg_partial_score'],
        globals().get('metrics_smart', {}).get('avg_partial_score'),
        "{:.1f}%"
    ))

    for field in EXACT_FIELDS + SET_FIELDS:
        acc_table.add_row(*get_row(
            f"  {field}",
            metrics_a['field_accuracy'][field],
            metrics_b['field_accuracy'][field],
            globals().get('metrics_smart', {}).get('field_accuracy', {}).get(field),
            "{:.1f}%"
        ))
    console.print(acc_table)

    # Operational table
    ops_table = Table(
        title="Operational Metrics",
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
    )
    ops_table.add_column("Metric", style="bold")
    ops_table.add_column("Strategy A (Direct)", justify="center")
    ops_table.add_column("Strategy B (CoT)", justify="center")
    if "metrics_smart" in globals():
        ops_table.add_column("Strategy Smart", justify="center")

    for label, key, fmt in [
        ("Cases Evaluated", "n", "{}"),
        ("Perfect Matches", "perfect_matches", "{}"),
        ("Total Input Tokens", "total_input_tokens", "{:,}"),
        ("Total Output Tokens", "total_output_tokens", "{:,}"),
        ("Total Images", "total_images", "{}"),
        ("Total Latency (s)", "total_latency", "{:.1f}s"),
        ("Avg Latency / case", "avg_latency", "{:.2f}s"),
        ("Estimated Cost (USD)", "estimated_cost_usd", "${:.4f}"),
    ]:
        ops_table.add_row(*get_row(
            label,
            metrics_a[key],
            metrics_b[key],
            globals().get('metrics_smart', {}).get(key),
            fmt
        ))
    console.print(ops_table)
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Number of claims to evaluate")
    parser.add_argument("--save", type=str, default="evaluation_results.json", help="Path to save results")
    args = parser.parse_args()

    console.print(
        Panel.fit(
            "[bold green]Running Strategy A (Direct) ...[/bold green]"
        )
    )
    metrics_a = run_evaluation(strategy_mode="A", limit=args.limit)

    console.print(
        Panel.fit(
            "[bold green]Running Strategy B (Chain-of-Thought) ...[/bold green]"
        )
    )
    metrics_b = run_evaluation(strategy_mode="B", limit=args.limit)

    console.print(
        Panel.fit(
            "[bold green]Running Strategy Smart (Dynamic Routing) ...[/bold green]"
        )
    )
    global metrics_smart
    metrics_smart = run_evaluation(strategy_mode="Smart", limit=args.limit)

    print_comparison(metrics_a, metrics_b)

    # Recommend the winner (now comparing all 3)
    best_score = max(metrics_a["perfect_match_pct"], metrics_b["perfect_match_pct"], metrics_smart["perfect_match_pct"])
    if metrics_smart["perfect_match_pct"] >= best_score and metrics_smart["estimated_cost_usd"] <= metrics_b["estimated_cost_usd"]:
        winner = "Smart (Dynamic Routing)"
    elif metrics_b["perfect_match_pct"] >= metrics_a["perfect_match_pct"]:
        winner = "B (CoT)"
    else:
        winner = "A (Direct)"
    console.print(
        Panel.fit(
            f"[bold yellow]Recommended strategy for output.csv: "
            f"Strategy {winner}[/bold yellow]"
        )
    )

    # Save results
    with open(args.save, "w") as f:
        json.dump({"Strategy_A": metrics_a, "Strategy_B": metrics_b, "Strategy_Smart": metrics_smart}, f, indent=2)
    console.print(f"[bold green]Results successfully saved to {args.save}[/bold green]")
