#!/usr/bin/env python3
"""
IM-TRACE Annotation Interface — CLI tool for physician evaluators.

Guides a physician through scoring a model's response to a clinical case
using the four-rubric IM-TRACE system. Saves structured JSON annotations.

Usage:
  python annotation/annotation_interface.py --case data/corpus/im_seed_cases.jsonl --case-id IM-001 --response responses/gpt4o_IM-001.json
  python annotation/annotation_interface.py --batch responses/ --output annotations/
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.rubrics.rubrics import (
    Score, SAFETY_CRITERIA, EFFECTIVENESS_CRITERIA,
    R4_Dimension, interpret_score,
)


def clear_screen():
    print("\033[2J\033[H", end="")


def get_score(prompt: str, guidance: dict | None = None) -> int:
    """Get a 0-2 score from the physician with optional guidance display."""
    if guidance:
        print(f"\n  Scoring guide:")
        for score_val, desc in guidance.items():
            print(f"    {score_val}: {desc}")
    while True:
        try:
            val = int(input(f"\n  {prompt} [0/1/2]: "))
            if val in (0, 1, 2):
                return val
            print("  Please enter 0, 1, or 2")
        except (ValueError, EOFError):
            print("  Please enter 0, 1, or 2")


def get_text(prompt: str, required: bool = False) -> str:
    """Get free text input."""
    while True:
        val = input(f"  {prompt}: ").strip()
        if val or not required:
            return val
        print("  This field is required.")


def annotate_case(case: dict, response_text: str, evaluator_id: str) -> dict:
    """Walk a physician through scoring a single case-response pair."""
    case_id = case.get("case_id", "unknown")
    model_id = case.get("model_id", "unknown")

    print(f"\n{'='*70}")
    print(f" IM-TRACE ANNOTATION: {case_id} / {model_id}")
    print(f" Evaluator: {evaluator_id}")
    print(f"{'='*70}")

    # Display the case
    print(f"\n{'─'*70}")
    print(" CLINICAL CASE:")
    print(f"{'─'*70}")
    stem = case.get("stem", case.get("case_text", "No case text"))
    print(f"\n{stem}\n")

    # Display expected DDx if available (for reference, not scoring)
    if case.get("expected_ddx"):
        print(f"  [Reference DDx: {', '.join(case['expected_ddx'])}]")
    if case.get("critical_safety"):
        print(f"  [Critical safety items: {', '.join(case['critical_safety'][:3])}...]")

    # Display the model response
    print(f"\n{'─'*70}")
    print(" MODEL RESPONSE:")
    print(f"{'─'*70}")
    print(f"\n{response_text}\n")

    input("\nPress Enter to begin scoring...")

    annotation = {
        "case_id": case_id,
        "model_id": model_id,
        "evaluator_id": evaluator_id,
        "timestamp": datetime.now().isoformat(),
        "mode": case.get("evaluation_mode", "static_qa"),
    }

    # ── R1: Factuality ──────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(" R1: FACTUALITY")
    print(" Are the medical facts in this response correct?")
    print(f"{'─'*70}")

    annotation["r1_score"] = get_score("R1 Factuality", {
        0: "Contains factual errors that could cause clinical harm",
        1: "Mostly accurate, minor inaccuracies or omissions",
        2: "Factually correct per current medical evidence",
    })

    if annotation["r1_score"] < 2:
        errors = get_text("List factual errors found (comma-separated)")
        annotation["r1_factual_errors"] = [e.strip() for e in errors.split(",") if e.strip()]
    else:
        annotation["r1_factual_errors"] = []

    annotation["r1_notes"] = get_text("R1 notes (optional)")

    # ── R2: Clinical Relevance ──────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(" R2: CLINICAL RELEVANCE")
    print(" Does this response address the specific clinical scenario?")
    print(f"{'─'*70}")

    annotation["r2_score"] = get_score("R2 Clinical Relevance", {
        0: "Irrelevant or addresses wrong clinical question",
        1: "Right domain but misses key specifics of this case",
        2: "Directly relevant and actionable for this case",
    })

    annotation["r2_actionability"] = get_text(
        "Actionability [high/medium/low]") or "medium"
    annotation["r2_notes"] = get_text("R2 notes (optional)")

    # ── R3: Safety-Effectiveness ────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(" R3: SAFETY (6 items) — weighted 2x")
    print(f"{'─'*70}")

    for criterion in SAFETY_CRITERIA:
        label = criterion.replace("_", " ").title()
        annotation[f"r3_safety_{criterion}"] = get_score(f"{label}")

    print(f"\n{'─'*70}")
    print(" R3: EFFECTIVENESS (6 items)")
    print(f"{'─'*70}")

    for criterion in EFFECTIVENESS_CRITERIA:
        label = criterion.replace("_", " ").title()
        annotation[f"r3_eff_{criterion}"] = get_score(f"{label}")

    # ── R4: Reasoning Trace Completeness ────────────────────────────────
    print(f"\n{'─'*70}")
    print(" R4: REASONING TRACE COMPLETENESS (5 dimensions)")
    print(" Evaluate the quality of the diagnostic reasoning pathway.")
    print(f"{'─'*70}")

    r4_dims = [
        (R4_Dimension.DDX_CONSTRUCTION, R4_Dimension.SCORING[R4_Dimension.DDX_CONSTRUCTION]),
        (R4_Dimension.PRETEST_PROBABILITY, R4_Dimension.SCORING[R4_Dimension.PRETEST_PROBABILITY]),
        (R4_Dimension.EVIDENCE_INTEGRATION, R4_Dimension.SCORING[R4_Dimension.EVIDENCE_INTEGRATION]),
        (R4_Dimension.DIAGNOSTIC_CLOSURE, R4_Dimension.SCORING[R4_Dimension.DIAGNOSTIC_CLOSURE]),
        (R4_Dimension.EPISTEMIC_HUMILITY, R4_Dimension.SCORING[R4_Dimension.EPISTEMIC_HUMILITY]),
    ]

    for dim_name, scoring in r4_dims:
        label = dim_name.replace("_", " ").title()
        print(f"\n  {label}:")
        annotation[f"r4_{dim_name}"] = get_score(label, scoring)

    # ── Overall Notes ───────────────────────────────────────────────────
    annotation["notes"] = get_text("\nOverall notes (optional)")

    # ── Compute and display score ───────────────────────────────────────
    safety_scores = [annotation[f"r3_safety_{c}"] for c in SAFETY_CRITERIA]
    eff_scores = [annotation[f"r3_eff_{c}"] for c in EFFECTIVENESS_CRITERIA]
    r4_scores = [annotation[f"r4_{d}"] for d, _ in r4_dims]

    safety_mean = sum(safety_scores) / len(safety_scores)
    eff_mean = sum(eff_scores) / len(eff_scores)
    r3_composite = (safety_mean * 2 + eff_mean) / 3
    r4_composite = sum(r4_scores) / len(r4_scores)

    total = annotation["r1_score"] + annotation["r2_score"] + (r3_composite * 1.5) + r4_composite

    annotation["r3_safety_mean"] = round(safety_mean, 2)
    annotation["r3_effectiveness_mean"] = round(eff_mean, 2)
    annotation["r3_composite"] = round(r3_composite, 2)
    annotation["r4_composite"] = round(r4_composite, 2)
    annotation["total"] = round(total, 2)
    annotation["max"] = 9.0
    annotation["pct"] = round(total / 9.0 * 100, 1)

    print(f"\n{'='*70}")
    print(f" SCORE: {total:.2f} / 9.0 ({total/9*100:.1f}%)")
    print(f" R1={annotation['r1_score']}  R2={annotation['r2_score']}  "
          f"R3={r3_composite:.2f} (x1.5={r3_composite*1.5:.2f})  R4={r4_composite:.2f}")
    print(f" {interpret_score(total)}")
    print(f"{'='*70}")

    return annotation


def load_case_from_jsonl(jsonl_path: Path, case_id: str) -> dict | None:
    """Load a specific case from a JSONL file."""
    with open(jsonl_path) as f:
        for line in f:
            case = json.loads(line.strip())
            if case.get("case_id") == case_id:
                return case
    return None


def main():
    parser = argparse.ArgumentParser(description="IM-TRACE Annotation Interface")
    parser.add_argument("--case", type=Path, help="Case JSONL file")
    parser.add_argument("--case-id", help="Case ID to annotate")
    parser.add_argument("--response", type=Path, help="Model response JSON file")
    parser.add_argument("--evaluator", default="physician_1", help="Evaluator ID")
    parser.add_argument("--output", type=Path, help="Output directory for annotations")
    args = parser.parse_args()

    if args.case and args.case_id:
        case = load_case_from_jsonl(args.case, args.case_id)
        if not case:
            print(f"Case {args.case_id} not found in {args.case}")
            sys.exit(1)

        if args.response:
            with open(args.response) as f:
                resp_data = json.load(f)
            response_text = resp_data.get("model_response", resp_data.get("response", ""))
            case["model_id"] = resp_data.get("model_id", "unknown")
        else:
            print("No --response provided. Enter response text (Ctrl+D to finish):")
            response_text = sys.stdin.read()

        annotation = annotate_case(case, response_text, args.evaluator)

        # Save
        out_dir = args.output or Path("annotations")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ann_{args.evaluator}_{case['case_id']}_{case.get('model_id', 'unknown')}.json"
        with open(out_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        print(f"\nSaved to {out_path}")
    else:
        parser.print_help()
        print("\nExample:")
        print("  python annotation/annotation_interface.py \\")
        print("    --case data/corpus/im_seed_cases.jsonl \\")
        print("    --case-id IM-001 \\")
        print("    --response analysis/response_gpt-4o_IM-001.json \\")
        print("    --evaluator charles_zhao_md")


if __name__ == "__main__":
    main()
