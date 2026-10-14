"""
src/prediction/confidence.py
Phase 6: BioScan Confidence Analysis and Unknown / Out-of-Reference Detection

Synthesizes multiple orthogonal evidence streams:
1. Supervised model predicted probabilities (top-1 prob, margin, entropy)
2. Reference library sequence similarity (nearest distance, distance margin)
3. Taxonomic consensus between classifier prediction and nearest reference

Categorizes inputs into three scientifically defensible statuses:
- "known_reference_like": High sequence identity to reference + strong taxonomic agreement
- "low_confidence": Ambiguous similarity, low probability margin, or conflicting taxonomy
- "out_of_reference": Substantial divergence from all 101 reference sequences
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------
# Calibrated Threshold Constants
# -------------------------------------------------------------
# Derived from empirical leave-one-reference-out and reference distribution analysis:
# - Minimum inter-sequence distance in reference library: 0.0093
# - 25th percentile of nearest-neighbor distances: 0.0377 (~4 bp diff)
# - Conspecific Ophidion marginatum distance: 0.0408 (~4 bp diff)
# - Median nearest-neighbor distance: 0.0660 (~7 bp diff)
# - 75th percentile of nearest-neighbor distances: 0.1215 (~12 bp diff)
# - Maximum nearest-neighbor distance: 0.2500 (~25 bp diff)

THRESHOLD_CLOSE_SIMILARITY = 0.050        # >= 95% sequence identity
THRESHOLD_OUT_OF_REFERENCE = 0.150       # < 85% sequence identity (cross-family divergence)
THRESHOLD_MODEL_PROB_HIGH = 0.700        # High top-1 probability
THRESHOLD_MODEL_PROB_LOW = 0.400         # Low top-1 probability
THRESHOLD_PROB_MARGIN_MIN = 0.150        # Minimum top-1 vs top-2 separation

def compute_probability_metrics(probs: np.ndarray) -> Dict[str, float]:
    """
    Compute confidence metrics from a multiclass probability distribution.
    """
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
        "entropy": round(entropy, 4),
    }

def evaluate_taxonomy_agreement(
    predicted_taxonomy: Dict[str, Optional[str]],
    nearest_reference: Dict
) -> Dict[str, bool]:
    """
    Check whether the classifier's predicted taxonomy matches the nearest reference.
    """
    agreement = {}
    for rank in ["class", "order", "family", "genus", "species"]:
        pred_val = predicted_taxonomy.get(rank)
        ref_val = nearest_reference.get(rank)
        
        # If either is None or "Not available", agreement is undefined/neutral
        if pred_val in [None, "Not available in reference data"] or ref_val in [None, "Not available in reference data"]:
            agreement[rank] = None
        else:
            agreement[rank] = (pred_val.strip().lower() == ref_val.strip().lower())
    return agreement

def determine_sequence_status(
    nearest_dist: float,
    prob_metrics: Dict[str, float],
    agreement: Dict[str, Optional[bool]],
    is_exact_match: bool = False
) -> Tuple[str, str, Dict[str, any]]:
    """
    Synthesize model probabilities, reference distance, and taxonomic agreement
    into a three-tier status:
    1. 'known_reference_like'
    2. 'low_confidence'
    3. 'out_of_reference'
    """
    top1_prob = prob_metrics.get("top1_prob", 0.0)
    margin = prob_metrics.get("prob_margin", 0.0)
    
    # Class and order agreement
    class_agrees = agreement.get("class") is True
    order_agrees = agreement.get("order") is True
    family_agrees = agreement.get("family") is True

    # 1. Check for Out-of-Reference
    if nearest_dist > THRESHOLD_OUT_OF_REFERENCE and not is_exact_match:
        status = "out_of_reference"
        explanation = (
            f"Sequence exhibits substantial divergence (distance {nearest_dist:.4f}, "
            f"similarity {(1.0 - nearest_dist)*100:.1f}%) from all 101 cataloged 12S rDNA V5 references. "
            "This indicates that the sequence falls outside the coverage of the current reference library."
        )
        level_conf = "low"

    # 2. Check for Known / Reference-Like
    elif (is_exact_match or nearest_dist <= THRESHOLD_CLOSE_SIMILARITY) and class_agrees and (order_agrees or order_agrees is None):
        status = "known_reference_like"
        if is_exact_match:
            explanation = "Sequence is an exact match (100% identity) to a cataloged reference barcode."
            level_conf = "high"
        else:
            explanation = (
                f"Sequence is highly similar to reference (distance {nearest_dist:.4f}, "
                f"similarity {(1.0 - nearest_dist)*100:.1f}%) with concordant taxonomic classification."
            )
            level_conf = "high" if top1_prob >= THRESHOLD_MODEL_PROB_HIGH else "moderate"

    # 3. Default: Low-Confidence / Ambiguous
    else:
        status = "low_confidence"
        reasons = []
        if nearest_dist > THRESHOLD_CLOSE_SIMILARITY:
            reasons.append(f"moderate reference divergence (distance {nearest_dist:.4f})")
        if top1_prob < THRESHOLD_MODEL_PROB_LOW:
            reasons.append(f"low model probability ({top1_prob:.4f})")
        if margin < THRESHOLD_PROB_MARGIN_MIN:
            reasons.append(f"narrow top-1 vs top-2 margin ({margin:.4f})")
        if not class_agrees and agreement.get("class") is not None:
            reasons.append("taxonomic disagreement with nearest reference at Class level")
        elif not order_agrees and agreement.get("order") is not None:
            reasons.append("taxonomic disagreement with nearest reference at Order level")
        
        reason_str = "; ".join(reasons) if reasons else "ambiguous support across evidence streams"
        explanation = f"Prediction has low/moderate confidence due to {reason_str}."
        level_conf = "low" if nearest_dist > 0.10 or top1_prob < 0.30 else "moderate"

    evidence_summary = {
        "status": status,
        "confidence_tier": level_conf,
        "explanation": explanation,
        "nearest_distance": nearest_dist,
        "nearest_similarity": round(1.0 - nearest_dist, 4),
        "is_exact_match": is_exact_match,
        "model_top1_prob": top1_prob,
        "model_prob_margin": margin,
        "class_agreement": class_agrees,
        "order_agreement": order_agrees,
        "family_agreement": family_agrees,
    }

    return status, level_conf, evidence_summary
