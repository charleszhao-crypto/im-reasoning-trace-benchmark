#!/usr/bin/env python3
"""
IM-TRACE Scorer — Four-rubric clinical LLM evaluation scoring engine.

Usage:
  # Score from JSON annotation file
  python evaluation/scorer.py --annotation annotation.json

  # Score interactively (CLI)
  python evaluation/scorer.py --interactive --case-id IM-001 --model gpt-4o

  # Batch score from directory
  python evaluation/scorer.py --batch annotations/ --output results.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.rubrics.rubrics import (
    IMTraceEvaluation, Score, R1_Factuality, R2_ClinicalRelevance,
    R3_SafetyEffectiveness, R3_SafetyItem, R3_EffectivenessItem,
    R4_ReasoningTraceCompleteness, R4_DimensionScore, R4_Dimension,
    SAFETY_CRITERIA, EFFECTIVENESS_CRITERIA,
    interpret_score,
)


def load_annotation(path: Path) -> dict:
    """Load a JSON annotation file."""
    with open(path) as f:
        return json.load(f)


def build_evaluation_from_dict(data: dict) -> IMTraceEvaluation:
    """Construct an IMTraceEvaluation from a flat annotation dict."""
    ev = IMTraceEvaluation.create_template(
        case_id=data.get("case_id", "unknown"),
        model_id=data.get("model_id", "unknown"),
        evaluator_id=data.get("evaluator_id", "anonymous"),
        mode=data.get("mode", "static_qa"),
    )

    # R1
    ev.r1.score = Score(data.get("r1_score", 0))
    ev.r1.factual_errors = data.get("r1_factual_errors", [])
    ev.r1.notes = data.get("r1_notes", "")

    # R2
    ev.r2.score = Score(data.get("r2_score", 0))
    ev.r2.actionability = data.get("r2_actionability", "")
    ev.r2.notes = data.get("r2_notes", "")

    # R3
    for i, criterion in enumerate(SAFETY_CRITERIA):
        key = f"r3_safety_{criterion}"
        ev.r3.safety_items[i].score = Score(data.get(key, 0))
        ev.r3.safety_items[i].notes = data.get(f"{key}_notes", "")

    for i, criterion in enumerate(EFFECTIVENESS_CRITERIA):
        key = f"r3_eff_{criterion}"
        ev.r3.effectiveness_items[i].score = Score(data.get(key, 0))
        ev.r3.effectiveness_items[i].notes = data.get(f"{key}_notes", "")

    # R4
    r4_dims = [
        R4_Dimension.DDX_CONSTRUCTION,
        R4_Dimension.PRETEST_PROBABILITY,
        R4_Dimension.EVIDENCE_INTEGRATION,
        R4_Dimension.DIAGNOSTIC_CLOSURE,
        R4_Dimension.EPISTEMIC_HUMILITY,
    ]
    for i, dim in enumerate(r4_dims):
        key = f"r4_{dim}"
        ev.r4.dimensions[i].score = Score(data.get(key, 0))
        ev.r4.dimensions[i].notes = data.get(f"{key}_notes", "")

    ev.notes = data.get("notes", "")
    return ev


def score_interactive(case_id: str, model_id: str, evaluator_id: str = "physician_1") -> IMTraceEvaluation:
    """Interactive CLI scoring session."""
    print(f"\n=== IM-TRACE Scoring: {case_id} / {model_id} ===\n")

    def get_score(prompt: str) -> Score:
        while True:
            try:
                val = int(input(f"  {prompt} [0/1/2]: "))
                return Score(val)
            except (ValueError, KeyError):
                print("  Enter 0, 1, or 2")

    ev = IMTraceEvaluation.create_template(case_id, model_id, evaluator_id)

    # R1
    print("--- R1: Factuality ---")
    ev.r1.score = get_score("Score")
    ev.r1.notes = input("  Notes (or Enter to skip): ").strip()

    # R2
    print("\n--- R2: Clinical Relevance ---")
    ev.r2.score = get_score("Score")
    ev.r2.actionability = input("  Actionability [high/medium/low]: ").strip() or "medium"
    ev.r2.notes = input("  Notes (or Enter to skip): ").strip()

    # R3 Safety
    print("\n--- R3: Safety Items ---")
    for item in ev.r3.safety_items:
        item.score = get_score(f"{item.criterion}")

    print("\n--- R3: Effectiveness Items ---")
    for item in ev.r3.effectiveness_items:
        item.score = get_score(f"{item.criterion}")

    # R4
    print("\n--- R4: Reasoning Trace Completeness ---")
    for dim in ev.r4.dimensions:
        dim.score = get_score(f"{dim.dimension}")

    ev.notes = input("\nOverall notes: ").strip()

    return ev


def print_scorecard(ev: IMTraceEvaluation) -> None:
    """Print a formatted scorecard."""
    s = ev.summary()
    print(f"\n{'='*60}")
    print(f" IM-TRACE Scorecard: {s['case_id']} / {s['model_id']}")
    print(f"{'='*60}")
    print(f" R1 Factuality:           {s['r1_factuality']:.1f} / 2.0")
    print(f" R2 Clinical Relevance:   {s['r2_relevance']:.1f} / 2.0")
    print(f" R3 Safety-Effectiveness: {s['r3_safety_effectiveness']:.2f} / 2.0  (x1.5 = {s['r3_safety_effectiveness']*1.5:.2f})")
    print(f"    Safety mean:          {s['r3_safety_mean']:.2f}")
    print(f"    Effectiveness mean:   {s['r3_effectiveness_mean']:.2f}")
    print(f" R4 Reasoning Trace:      {s['r4_reasoning_trace']:.2f} / 2.0")
    for dim, score in s['r4_dimensions'].items():
        print(f"    {dim:25s} {score}")
    print(f"{'─'*60}")
    print(f" TOTAL:                   {s['total']:.2f} / {s['max']:.1f} ({s['pct']:.1f}%)")
    print(f" Interpretation:          {interpret_score(s['total'])}")
    print(f"{'='*60}\n")


def batch_score(annotations_dir: Path, output_path: Path) -> None:
    """Score all annotation files in a directory."""
    results = []
    for f in sorted(annotations_dir.glob("*.json")):
        data = load_annotation(f)
        ev = build_evaluation_from_dict(data)
        results.append(ev.summary())
        print(f"  Scored {f.name}: {ev.total_score:.2f} / 9.0")

    with open(output_path, 'w') as out:
        for r in results:
            out.write(json.dumps(r) + "\n")

    print(f"\n{len(results)} evaluations written to {output_path}")

    if results:
        avg = sum(r['total'] for r in results) / len(results)
        print(f"Mean IM-TRACE score: {avg:.2f} / 9.0 ({avg/9*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="IM-TRACE Scorer")
    parser.add_argument("--annotation", type=Path, help="Single annotation JSON to score")
    parser.add_argument("--interactive", action="store_true", help="Interactive CLI scoring")
    parser.add_argument("--case-id", default="IM-001", help="Case ID for interactive mode")
    parser.add_argument("--model", default="gpt-4o", help="Model ID for interactive mode")
    parser.add_argument("--evaluator", default="physician_1", help="Evaluator ID")
    parser.add_argument("--batch", type=Path, help="Directory of annotation JSONs to batch score")
    parser.add_argument("--output", type=Path, default=Path("results.jsonl"), help="Output path for batch")
    args = parser.parse_args()

    if args.annotation:
        data = load_annotation(args.annotation)
        ev = build_evaluation_from_dict(data)
        print_scorecard(ev)
    elif args.interactive:
        ev = score_interactive(args.case_id, args.model, args.evaluator)
        print_scorecard(ev)
        out_path = Path(f"annotation_{args.case_id}_{args.model}.json")
        with open(out_path, 'w') as f:
            json.dump(ev.summary(), f, indent=2)
        print(f"Saved to {out_path}")
    elif args.batch:
        batch_score(args.batch, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
