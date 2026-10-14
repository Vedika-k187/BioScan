"""
src/prediction/evidence.py
Phase 8: Multi-Stream Evidence Aggregation & Tri-Tier Unknown Detection

Synthesizes:
1. Supervised Model Evidence (top-1 prob, margin, entropy)
2. Sequence Similarity Evidence (nearest distance, similarity, exact match, distance gap)
3. Taxonomic Concordance (concordance across hierarchical ranks)
4. Reference Provenance & Biodiversity Context

Enforces Non-Negotiable Distance Veto:
- If d_min > 0.150 and not an exact match -> ALWAYS 'out_of_reference',
  regardless of classifier probability!
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Validated operational thresholds from Phase 6
THRESHOLD_CLOSE_SIMILARITY = 0.050   # >= 95% sequence identity
THRESHOLD_OUT_OF_REFERENCE = 0.150   # < 85% sequence identity
THRESHOLD_MODEL_PROB_HIGH = 0.700    # Decisive probability
THRESHOLD_MODEL_PROB_LOW = 0.400     # Ambiguous probability
THRESHOLD_PROB_MARGIN_MIN = 0.150    # Minimum top-1 vs top-2 separation


def compute_probability_metrics(probs: np.ndarray) -> Dict[str, float]:
    """Calculate top-1 prob, margin, and Shannon entropy from class posterior probabilities."""
    sorted_probs = np.sort(probs)[::-1]
    top1 = float(sorted_probs[0])
    top2 = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = float(top1 - top2)

    # Shannon entropy (bits)
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = float(-np.sum(clipped * np.log2(clipped)))

    return {
        "top1_prob": round(top1, 4),
        "top2_prob": round(top2, 4),
        "prob_margin": round(margin, 4),
        "entropy": round(entropy, 4)
    }


def evaluate_taxonomic_concordance(
    predicted_tax: Dict[str, Optional[str]],
    nearest_ref: Dict[str, Any]
) -> Dict[str, Optional[bool]]:
    """Compare hierarchical classifier predictions with nearest reference taxonomy."""
    concordance: Dict[str, Optional[bool]] = {}
    for rank in ["class", "order", "family", "genus", "species"]:
        pred_val = predicted_tax.get(rank)
        ref_val = nearest_ref.get(rank)

        if pred_val in [None, "", "Not available in reference data"] or ref_val in [None, "", "Not available in reference data"]:
            concordance[rank] = None
        else:
            concordance[rank] = (str(pred_val).strip().lower() == str(ref_val).strip().lower())
    return concordance


def synthesize_evidence_and_status(
    nearest_dist: float,
    nearest_ref: Dict[str, Any],
    prob_metrics: Dict[str, float],
    concordance: Dict[str, Optional[bool]],
    is_exact_match: bool = False,
    second_dist: Optional[float] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Synthesize multi-stream evidence into tri-tier status:
    1. 'known_reference_like'
    2. 'low_confidence'
    3. 'out_of_reference'

    CRITICAL RULE: Distance Veto.
    A sequence with nearest_dist > 0.150 and not an exact match MUST be classified
    as 'out_of_reference', regardless of model classifier probabilities!
    """
    top1_prob = prob_metrics.get("top1_prob", 0.0)
    margin = prob_metrics.get("prob_margin", 0.0)
    dist_gap = round(second_dist - nearest_dist, 4) if second_dist is not None else None

    class_agrees = concordance.get("class") is True
    order_agrees = concordance.get("order") is True
    family_agrees = concordance.get("family") is True

    # 1. Enforce Distance Veto: Out-of-Reference
    if nearest_dist > THRESHOLD_OUT_OF_REFERENCE and not is_exact_match:
        status = "out_of_reference"
        confidence_tier = "low"
        explanation = (
            f"Sequence exhibits substantial divergence (distance {nearest_dist:.4f}, "
            f"similarity {(1.0 - nearest_dist)*100:.1f}%) from all 101 reference barcodes in the library. "
            "Because this exceeds the out-of-reference boundary (0.150), the sequence is classified as "
            "out-of-reference regardless of classifier probability. This indicates lack of representation "
            "in the current reference library, NOT proof of a novel biological species."
        )

    # 2. Known / Reference-Like
    elif (is_exact_match or nearest_dist <= THRESHOLD_CLOSE_SIMILARITY) and class_agrees and (order_agrees or order_agrees is None):
        status = "known_reference_like"
        if is_exact_match:
            explanation = "Sequence is an exact match (100% sequence identity) to a cataloged reference barcode."
            confidence_tier = "high"
        else:
            explanation = (
                f"Sequence is highly similar to reference (distance {nearest_dist:.4f}, "
                f"similarity {(1.0 - nearest_dist)*100:.1f}%) with concordant taxonomic classification."
            )
            confidence_tier = "high" if top1_prob >= THRESHOLD_MODEL_PROB_HIGH else "moderate"

    # 3. Default: Low-Confidence / Ambiguous
    else:
        status = "low_confidence"
        reasons = []
        if nearest_dist > THRESHOLD_CLOSE_SIMILARITY:
            reasons.append(f"moderate reference divergence (distance {nearest_dist:.4f})")
        if top1_prob < THRESHOLD_MODEL_PROB_LOW:
            reasons.append(f"low model probability ({top1_prob:.4f})")
        if margin < THRESHOLD_PROB_MARGIN_MIN:
            reasons.append(f"narrow probability margin ({margin:.4f})")
        if not class_agrees and concordance.get("class") is not None:
            reasons.append("taxonomic disagreement with nearest reference at Class level")
        elif not order_agrees and concordance.get("order") is not None:
            reasons.append("taxonomic disagreement with nearest reference at Order level")

        reason_str = "; ".join(reasons) if reasons else "ambiguous support across orthogonal evidence streams"
        explanation = f"Prediction has low/moderate confidence due to {reason_str}."
        confidence_tier = "low" if nearest_dist > 0.10 or top1_prob < 0.30 else "moderate"

    evidence_summary = {
        "status": status,
        "confidence_tier": confidence_tier,
        "explanation": explanation,
        "nearest_distance": nearest_dist,
        "nearest_similarity": round(1.0 - nearest_dist, 4),
        "is_exact_match": is_exact_match,
        "model_top1_prob": top1_prob,
        "model_prob_margin": margin,
        "class_agreement": class_agrees,
        "order_agreement": order_agrees,
        "family_agreement": family_agrees,
        "model_evidence": {
            "top1_probability": top1_prob,
            "top2_probability": prob_metrics.get("top2_prob", 0.0),
            "probability_margin": margin,
            "entropy": prob_metrics.get("entropy", 0.0)
        },
        "sequence_evidence": {
            "nearest_reference_id": nearest_ref.get("sequence_id"),
            "nearest_distance": nearest_dist,
            "nearest_similarity": round(1.0 - nearest_dist, 4),
            "is_exact_match": is_exact_match,
            "distance_gap_to_second": dist_gap
        },
        "taxonomic_concordance": {
            "class_concordant": class_agrees,
            "order_concordant": order_agrees,
            "family_concordant": family_agrees
        }
    }

    return status, confidence_tier, evidence_summary
