#!/usr/bin/env python3
"""
IM-TRACE Benchmark Runner — Generate and evaluate model responses.

Workflow:
  1. Load cases from corpus
  2. For each model in config: generate responses
  3. Auto-grade each response using the grading LLM
  4. Output results as JSONL + summary table

Usage:
  python baselines/run_benchmark.py --config baselines/model_configs.yaml
  python baselines/run_benchmark.py --config baselines/model_configs.yaml --models gpt-4o claude-sonnet
  python baselines/run_benchmark.py --config baselines/model_configs.yaml --cases IM-001 IM-002
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.auto_grader import call_grading_llm, compute_composite
from data.rubrics.rubrics import interpret_score


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_cases(cases_dir: Path, case_ids: list[str] | None = None) -> list[dict]:
    cases = []
    for f in sorted(cases_dir.glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                if case_ids and case.get("case_id") not in case_ids:
                    continue
                cases.append(case)
    return cases


def generate_response(
    case: dict,
    model_config: dict,
    prompt_template: str,
) -> str:
    """Generate a model response for a clinical case."""
    prompt = prompt_template.format(case_stem=case["stem"])
    system = model_config.get("system_prompt", "You are a physician.")
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    max_tokens = model_config.get("max_tokens", 4000)
    temperature = model_config.get("temperature", 0.3)

    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return "[STUB] No ANTHROPIC_API_KEY. Set the environment variable to generate real responses."
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
            msg = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:
            return f"[ERROR] Anthropic: {e}"

    elif provider == "openai":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return "[STUB] No OPENAI_API_KEY. Set the environment variable to generate real responses."
        try:
            import openai
            client = openai.OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"[ERROR] OpenAI: {e}"

    elif provider == "google":
        return "[STUB] Google Gemini integration not yet implemented."

    return f"[STUB] Unknown provider: {provider}"


def run_benchmark(config: dict, model_names: list[str] | None = None, case_ids: list[str] | None = None):
    """Run the full benchmark pipeline."""
    cases_dir = Path(config["evaluation"]["cases_dir"])
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_template = config.get("clinical_prompt", "{case_stem}")
    grading_model = config["evaluation"].get("grading_model", "gpt-4o")

    models = config["models"]
    if model_names:
        models = {k: v for k, v in models.items() if k in model_names}

    cases = load_cases(cases_dir, case_ids)
    if not cases:
        print("No cases found.")
        return

    print(f"Benchmark: {len(cases)} cases x {len(models)} models")
    print(f"Grading model: {grading_model}\n")

    all_results = []

    for model_name, model_config in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")
        print(f"{'='*50}")

        model_results = []

        for case in cases:
            case_id = case["case_id"]
            print(f"  {case_id}: generating...", end=" ", flush=True)

            # Generate response
            response = generate_response(case, model_config, prompt_template)

            # Save response
            resp_path = output_dir / f"response_{model_name}_{case_id}.json"
            with open(resp_path, 'w') as f:
                json.dump({
                    "case_id": case_id,
                    "model_id": model_name,
                    "case_text": case["stem"],
                    "model_response": response,
                }, f, indent=2)

            # Auto-grade
            print("grading...", end=" ", flush=True)
            raw_scores = call_grading_llm(case["stem"], response, grading_model)
            result = compute_composite(raw_scores)
            result["case_id"] = case_id
            result["model_id"] = model_name

            model_results.append(result)
            print(f"score: {result.get('total', 'N/A'):.1f}/9.0")

            time.sleep(0.5)  # Rate limiting

        # Save model results
        results_path = output_dir / f"benchmark_{model_name}.jsonl"
        with open(results_path, 'w') as f:
            for r in model_results:
                f.write(json.dumps(r) + "\n")

        all_results.extend(model_results)

        # Model summary
        if model_results:
            avg = sum(r.get('total', 0) for r in model_results) / len(model_results)
            print(f"\n  {model_name} mean: {avg:.2f}/9.0 ({avg/9*100:.1f}%)")
            print(f"  Interpretation: {interpret_score(avg)}")

    # Cross-model comparison table
    if all_results:
        print(f"\n{'='*70}")
        print("CROSS-MODEL COMPARISON")
        print(f"{'='*70}")
        print(f"{'Model':<20} {'R1':>5} {'R2':>5} {'R3':>5} {'R4':>5} {'Total':>7} {'%':>6}")
        print(f"{'-'*70}")

        by_model = {}
        for r in all_results:
            m = r["model_id"]
            if m not in by_model:
                by_model[m] = []
            by_model[m].append(r)

        for model_name, results in sorted(by_model.items()):
            n = len(results)
            avg_r1 = sum(r.get("r1_score", 0) for r in results) / n
            avg_r2 = sum(r.get("r2_score", 0) for r in results) / n
            avg_r3 = sum(r.get("r3_composite", 0) for r in results) / n
            avg_r4 = sum(r.get("r4_composite", 0) for r in results) / n
            avg_total = sum(r.get("total", 0) for r in results) / n

            print(f"{model_name:<20} {avg_r1:>5.2f} {avg_r2:>5.2f} {avg_r3:>5.2f} {avg_r4:>5.2f} {avg_total:>7.2f} {avg_total/9*100:>5.1f}%")

        # Save comparison
        comparison_path = output_dir / "benchmark_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump({"results": all_results}, f, indent=2)
        print(f"\nFull results: {comparison_path}")


def main():
    parser = argparse.ArgumentParser(description="IM-TRACE Benchmark Runner")
    parser.add_argument("--config", type=Path, required=True, help="Model config YAML")
    parser.add_argument("--models", nargs="+", help="Specific models to run (default: all)")
    parser.add_argument("--cases", nargs="+", help="Specific case IDs to run (default: all)")
    args = parser.parse_args()

    config = load_config(args.config)
    run_benchmark(config, args.models, args.cases)


if __name__ == "__main__":
    main()
