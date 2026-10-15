"""
app/components/rank_resolution.py
Explains, for a low-confidence result, how far down the taxonomy the evidence
actually supports an answer, and what that kind of match has achieved in
leave-one-out testing on the reference library.

Presentation only. It reads fields already produced by the engine and the stored
evaluation results; it never changes a status, a prediction or a threshold.
"""

import json
import os
from typing import Any, Dict, Optional

import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(BASE_DIR, "reports", "evaluation", "evaluation_results.json")
RANKS = ["class", "order", "family", "genus", "species"]
# Only ranks that have leave-one-out evidence may be called "resolved" in the
# low-confidence band. Genus and species cannot be validated with this library.
RESOLVABLE_RANKS = ["class", "order", "family"]
NOT_AVAILABLE = "Not available in reference data"


@st.cache_data(show_spinner=False)
def _load_eval(path: str, mtime: float) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def deepest_concordant_rank(concordance: Dict[str, Any], allowed=RANKS) -> Optional[str]:
    """Deepest rank (within `allowed`) such that it and every rank above it agree with the nearest reference."""
    resolved = None
    for rank in allowed:
        if concordance.get(f"{rank}_concordant") is True:
            resolved = rank
        else:
            break
    return resolved


def _evidence_sentence(rank: str, distance: float) -> Optional[str]:
    mtime = os.path.getmtime(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else 0.0
    res = _load_eval(RESULTS_PATH, mtime)
    if not res:
        return None
    entry = res.get("taxonomy", {}).get(rank)
    if not entry or entry.get("status") != "ok":
        return None
    nn = entry.get("nearest_reference_baseline") or {}
    rows = nn.get("accuracy_vs_distance") or []
    candidates = [r for r in rows if r["distance_threshold"] >= distance and r["accepted"] > 0]
    if not candidates:
        return None
    row = min(candidates, key=lambda r: r["distance_threshold"])
    n_total = entry["support"]["n_eligible_sequences"]
    return (
        f"In leave-one-out testing on this reference library, when a sequence's nearest other reference "
        f"was within distance {row['distance_threshold']:.2f}, that reference's **{rank}** label was correct "
        f"**{row['accuracy_of_accepted']:.0%}** of the time ({row['accepted']} of {n_total} evaluable sequences). "
        "This describes the method on a small library; it is not the probability that this particular call is correct."
    )


def render_rank_resolution(result: Dict[str, Any]) -> None:
    prediction = result.get("prediction") or {}
    if prediction.get("status") != "low_confidence":
        return

    similarity = result.get("similarity") or {}
    evidence = result.get("evidence") or {}
    concordance = evidence.get("taxonomic_concordance") or {}
    distance = similarity.get("nearest_distance")
    nearest_id = similarity.get("nearest_reference_id")

    rank = deepest_concordant_rank(concordance, RESOLVABLE_RANKS)
    st.markdown("#### What the evidence supports")
    if rank is None:
        st.warning(
            "**No taxonomic rank can be resolved reliably.** The classifier's lineage does not agree "
            f"with the nearest reference ({nearest_id}) even at class level, so nothing is assigned."
        )
        return

    value = prediction.get(rank)
    if not value or value == NOT_AVAILABLE:
        st.warning("**No taxonomic rank can be resolved reliably** for this sequence.")
        return

    below = RANKS[RANKS.index(rank) + 1:]
    nearest_genus = (similarity.get("top_references") or [{}])[0].get("genus")
    note = ""
    if below:
        note = f"Ranks below it ({', '.join(below)}) are **not** assigned"
        if nearest_genus:
            note += (f"; the nearest reference belongs to genus *{nearest_genus}*, but genus- and "
                     "species-level assignment cannot be validated with this reference library")
        note += "."
    st.success(
        f"**Resolved to {rank}: {value}.** The classifier and the nearest reference "
        f"({nearest_id}, distance {distance:.3f}) agree down to this rank. " + note
    )
    sentence = _evidence_sentence(rank, float(distance)) if distance is not None else None
    if sentence:
        st.caption(sentence)
    else:
        st.caption("No leave-one-out evaluation is available for this rank, so no accuracy estimate is shown.")
