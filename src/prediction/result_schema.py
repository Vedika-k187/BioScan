"""
src/prediction/result_schema.py
Phase 8: BioScan Prediction Result Schema

Defines the structured, scientific output contract for BioScan predictions.
Guarantees consistent keys, explicit null handling, and complete transparency.
"""

from typing import Dict, List, Any, Optional

DEFAULT_DISCLAIMER = (
    "BioScan predictions are derived from a curated reference library of 101 unique 12S rDNA V5 "
    "teleost and elasmobranch barcodes and 2019 New Jersey Ocean Trawl Survey occurrence metadata. "
    "Because 89 of 90 cataloged species are singletons, species predictions operate under "
    "reference-library constraints. A status of 'out_of_reference' indicates insufficient support "
    "in the current library, NOT biological proof of a novel species."
)


def format_qc_failure_result(
    sequence: Any,
    sequence_id: Optional[str],
    errors: List[str],
    warnings: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Format a structured result for an input that failed biological QC."""
    seq_str = str(sequence) if sequence is not None else ""
    return {
        # Root-level backward-compatibility keys (Phases 6 & 7)
        "qc_status": "FAILED",
        "qc_errors": errors,
        "status": "qc_failed",
        "input_raw": seq_str[:50] if sequence is not None else None,
        "prediction": {
            "status": "qc_failed",
            "confidence_tier": "unassigned",
            "kingdom": None,
            "class": None,
            "order": None,
            "family": None,
            "genus": None,
            "species": None,
            "species_prediction_scope": "unassigned"
        },
        "confidence": None,
        "similarity": None,
        "evidence": None,
        "nearest_references": [],
        "biodiversity": None,
        "disclaimer": "Sequence failed biological quality control. No classification attempted.",
        # Structured Phase 8 section keys
        "input": {
            "sequence_id": sequence_id or "UNKNOWN_INPUT",
            "sequence": seq_str[:50] + ("..." if len(seq_str) > 50 else ""),
            "length": len(seq_str)
        },
        "qc": {
            "status": "failed",
            "is_valid": False,
            "errors": errors,
            "warnings": warnings or []
        }
    }


def format_prediction_result(
    input_info: Dict[str, Any],
    qc_info: Dict[str, Any],
    prediction_info: Dict[str, Any],
    confidence_info: Dict[str, Any],
    similarity_info: Dict[str, Any],
    evidence_info: Dict[str, Any],
    biodiversity_info: Optional[Dict[str, Any]] = None,
    disclaimer: Optional[str] = None
) -> Dict[str, Any]:
    """Assemble the complete, production-grade structured prediction dictionary."""
    return {
        # Root-level backward-compatibility keys (Phases 6 & 7)
        "qc_status": "PASSED",
        "qc_metrics": qc_info.get("metrics"),
        "status": prediction_info.get("status"),
        "confidence_tier": prediction_info.get("confidence_tier"),
        "predicted_taxonomy": {
            "class": prediction_info.get("class"),
            "order": prediction_info.get("order"),
            "family": prediction_info.get("family"),
            "genus": prediction_info.get("genus"),
            "species": prediction_info.get("species")
        },
        "level_confidences": confidence_info.get("level_confidences"),
        "nearest_references": similarity_info.get("top_references", []),
        "evidence": evidence_info,
        "disclaimer": disclaimer or DEFAULT_DISCLAIMER,
        # Structured Phase 8 section keys
        "input": input_info,
        "qc": qc_info,
        "prediction": prediction_info,
        "confidence": confidence_info,
        "similarity": similarity_info,
        "biodiversity": biodiversity_info,
    }
