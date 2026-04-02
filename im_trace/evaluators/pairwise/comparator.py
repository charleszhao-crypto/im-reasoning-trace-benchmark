"""
IM-TRACE Pairwise Comparator — Blinded A/B presentation and result recording.

Handles three concerns:
  1. Randomized, blinded presentation of two model responses (create_comparison_pair)
  2. Recording judge preferences into a PairwiseComparisonRecord (record_comparison)
  3. Converting records to the {"winner", "loser"} format expected by fit_bradley_terry
     in aggregate.py, with tie-splitting

Design notes:
  - Blinding is one-way at presentation time; de-blinding uses the returned pair_info dict.
  - Tie-splitting (0.5 wins each) is applied at BT-input construction, not at record time,
    so the canonical record always preserves the raw judgment.
  - Seeds are stored on the record for reproducibility audits.
"""

from __future__ import annotations

import random
from typing import Optional

from im_trace.cases.schema.models import (
    JudgeType,
    ModelResponse,
    PairwiseComparisonRecord,
    RubricPreference,
)


# ── Presentation ─────────────────────────────────────────────────────────────

def create_comparison_pair(
    case_id: str,
    response_a: ModelResponse,
    response_b: ModelResponse,
    seed: int,
) -> dict:
    """
    Prepare a blinded, randomly-ordered presentation of two model responses.

    The display labels "Response A" / "Response B" refer to the PRESENTED
    positions, not to the original response_a / response_b arguments.
    A judge always sees exactly two labeled slots; which underlying response
    maps to which slot is determined by the seed.

    Args:
        case_id:    The clinical case ID both responses are for.
        response_a: First model response (ModelResponse).
        response_b: Second model response (ModelResponse).
        seed:       Integer seed for deterministic order randomization.
                    Using the same seed on the same pair always produces
                    the same display order — useful for reproducibility.

    Returns:
        A dict with:
          "case_id"         — echoed from argument
          "seed"            — echoed from argument
          "presented_order" — "a_first" or "b_first"
          "display" — {
              "Response A": {"response_id": ..., "text": ...},
              "Response B": {"response_id": ..., "text": ...},
          }
          "_deblind_map" — {
              "Response A": {"response_id": ..., "model_id": ...},
              "Response B": {"response_id": ..., "model_id": ...},
          }
          "_response_a_id"  — original response_a.response_id (for record_comparison)
          "_response_b_id"  — original response_b.response_id (for record_comparison)
          "_model_a_id"     — original response_a.model_id
          "_model_b_id"     — original response_b.model_id

    The "_deblind_map" and "_model_*_id" keys are prefixed with "_" to signal
    that they should not be shown to the judge.
    """
    rng = random.Random(seed)
    a_first = rng.random() < 0.5

    if a_first:
        presented_order = "a_first"
        slot_a = response_a
        slot_b = response_b
    else:
        presented_order = "b_first"
        slot_a = response_b
        slot_b = response_a

    return {
        "case_id": case_id,
        "seed": seed,
        "presented_order": presented_order,
        "display": {
            "Response A": {
                "response_id": slot_a.response_id,
                "text": slot_a.response_text,
            },
            "Response B": {
                "response_id": slot_b.response_id,
                "text": slot_b.response_text,
            },
        },
        # De-blinding metadata — keep away from judges
        "_deblind_map": {
            "Response A": {
                "response_id": slot_a.response_id,
                "model_id": slot_a.model_id,
            },
            "Response B": {
                "response_id": slot_b.response_id,
                "model_id": slot_b.model_id,
            },
        },
        "_response_a_id": response_a.response_id,
        "_response_b_id": response_b.response_id,
        "_model_a_id": response_a.model_id,
        "_model_b_id": response_b.model_id,
    }


# ── Recording ─────────────────────────────────────────────────────────────────

def record_comparison(
    pair_info: dict,
    preferences: list[RubricPreference],
    judge_type: JudgeType,
    judge_id: str,
    rubric_version: str = "1.0",
) -> PairwiseComparisonRecord:
    """
    Convert a completed judgment into a PairwiseComparisonRecord.

    Preferences supplied by the judge use presented-position labels
    ("a", "b", "tie"), where "a" means the response shown as "Response A"
    won.  The mapping back to actual model IDs is stored on the record
    via model_a_id / model_b_id and presented_order.

    Args:
        pair_info:       The dict returned by create_comparison_pair.
        preferences:     List of RubricPreference objects from the judge.
                         RubricPreference.winner should be "a", "b", or "tie",
                         where "a"/"b" refer to the DISPLAYED positions.
        judge_type:      JudgeType enum value (PHYSICIAN, LLM_JUDGE, ADJUDICATED).
        judge_id:        Evaluator ID string or judge model ID.
        rubric_version:  Rubric version string (default "1.0").

    Returns:
        A fully populated PairwiseComparisonRecord.  The record stores
        the original response_a / response_b IDs and model IDs so that
        downstream analysis can de-blind results independently of pair_info.
    """
    return PairwiseComparisonRecord(
        case_id=pair_info["case_id"],
        response_a_id=pair_info["_response_a_id"],
        response_b_id=pair_info["_response_b_id"],
        model_a_id=pair_info["_model_a_id"],
        model_b_id=pair_info["_model_b_id"],
        presented_order=pair_info["presented_order"],
        seed=pair_info["seed"],
        preferences=preferences,
        judge_type=judge_type,
        judge_id=judge_id,
        rubric_version=rubric_version,
    )


# ── BT Input Conversion ───────────────────────────────────────────────────────

def comparisons_to_bt_input(
    records: list[PairwiseComparisonRecord],
    rubric: str = "overall",
    subscale: Optional[str] = None,
) -> list[dict]:
    """
    Convert pairwise comparison records to Bradley-Terry input format.

    Produces the list[{"winner": model_id, "loser": model_id}] consumed
    by fit_bradley_terry() in aggregate.py.

    Tie handling:
        Ties are split into two fractional entries — one with each model as
        winner — to preserve the information that a matchup occurred without
        crediting either model with a full win.  Since fit_bradley_terry
        uses integer counts internally, ties contribute 0.5 to each model's
        numerator, approximated here by emitting two entries.  This is the
        standard approach in BT literature (see Davidson 1970).

    Args:
        records:   List of PairwiseComparisonRecord objects.
        rubric:    Which rubric preference to use for winner determination.
                   Defaults to "overall".  Use "r1", "r2", "r3", "r4", or
                   a subscale key (e.g., "ddx_construction" with subscale set).
        subscale:  Optional subscale filter.  If set, only RubricPreference
                   entries where preference.subscale == subscale are used.

    Returns:
        List of {"winner": model_id, "loser": model_id} dicts.
        Ties produce two entries (one for each direction at 0.5 weight),
        implemented as two full dicts — the BT fitter counts entries.

    Raises:
        ValueError: If a record has no matching preference for the requested
                    rubric/subscale combination.
    """
    bt_input: list[dict] = []

    for record in records:
        preference = _find_preference(record.preferences, rubric, subscale)
        if preference is None:
            # Skip records that have no judgment for this rubric/subscale
            continue

        winner_label = preference.winner  # "a", "b", or "tie"

        if winner_label == "tie":
            # Split the tie: emit one entry per direction
            bt_input.append({"winner": record.model_a_id, "loser": record.model_b_id})
            bt_input.append({"winner": record.model_b_id, "loser": record.model_a_id})
        elif winner_label == "a":
            bt_input.append({"winner": record.model_a_id, "loser": record.model_b_id})
        elif winner_label == "b":
            bt_input.append({"winner": record.model_b_id, "loser": record.model_a_id})
        else:
            raise ValueError(
                f"Unexpected preference winner value '{winner_label}' in record "
                f"{record.comparison_id}. Expected 'a', 'b', or 'tie'."
            )

    return bt_input


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _find_preference(
    preferences: list[RubricPreference],
    rubric: str,
    subscale: Optional[str],
) -> Optional[RubricPreference]:
    """
    Find the first RubricPreference matching the requested rubric and subscale.

    Returns None if no match is found (caller decides how to handle).
    """
    for pref in preferences:
        if pref.rubric != rubric:
            continue
        if subscale is not None and pref.subscale != subscale:
            continue
        return pref
    return None
