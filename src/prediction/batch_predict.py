"""
src/prediction/batch_predict.py
Phase 8: BioScan Batch Prediction & CSV Export Module

Handles:
- Independent batch prediction over lists of sequences or FASTA records.
- Robust error containment: an invalid sequence never crashes the batch.
- Preservation of input identifiers, order, and metadata.
- Flattening structured results into publication/reporting-ready CSV tables.
"""

import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union

from .fasta_parser import parse_fasta


def predict_batch(
    sequences: List[Union[str, Dict[str, str]]],
    predictor: Any,
    top_k_similar: int = 5
) -> List[Dict[str, Any]]:
    """
    Execute prediction over a batch of DNA sequences or parsed sequence dictionaries.

    Parameters:
    - sequences: list of strings (raw DNA) or dicts ({"id": ..., "sequence": ...})
    - predictor: BioScanPredictor instance
    - top_k_similar: number of nearest references to retrieve

    Returns:
    - List of structured prediction result dictionaries, matching input order.
    """
    if sequences is None:
        return []

    results: List[Dict[str, Any]] = []

    for idx, item in enumerate(sequences, start=1):
        if isinstance(item, dict):
            seq_id = item.get("id") or f"SEQ_INPUT_{idx:03d}"
            seq_str = item.get("sequence", "")
        elif isinstance(item, str):
            seq_id = f"SEQ_INPUT_{idx:03d}"
            seq_str = item
        else:
            seq_id = f"SEQ_INPUT_{idx:03d}"
            seq_str = str(item)

        try:
            res = predictor.predict(seq_str, sequence_id=seq_id, top_k_similar=top_k_similar)
            results.append(res)
        except Exception as e:
            # Contain unexpected errors without crashing batch
            from .result_schema import format_qc_failure_result
            err_res = format_qc_failure_result(
                sequence=seq_str,
                sequence_id=seq_id,
                errors=[f"Unexpected engine exception during batch prediction: {str(e)}"]
            )
            err_res["prediction"]["status"] = "system_error"
            results.append(err_res)

    return results


def predict_fasta(
    fasta_input: Union[str, os.PathLike],
    predictor: Any,
    top_k_similar: int = 5
) -> List[Dict[str, Any]]:
    """
    Parse a FASTA file or string and execute independent predictions for all sequences.
    """
    records = parse_fasta(fasta_input)
    return predict_batch(records, predictor=predictor, top_k_similar=top_k_similar)


def batch_results_to_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten a list of structured BioScan prediction results into a tabular DataFrame.
    """
    rows: List[Dict[str, Any]] = []

    for res in results:
        inp = res.get("input") or {}
        qc = res.get("qc") or {}
        pred = res.get("prediction") or {}
        conf = res.get("confidence") or {}
        sim = res.get("similarity") or {}
        ev = res.get("evidence") or {}
        bio = res.get("biodiversity") or {}

        status = pred.get("status", "unknown")
        is_out = (status == "out_of_reference")
        is_exact = sim.get("is_exact_match", False) if sim else False

        # Extract model evidence
        mod_ev = ev.get("model_evidence") if ev else {}
        top1_prob = mod_ev.get("top1_probability") if mod_ev else conf.get("top1_probability")
        prob_margin = mod_ev.get("probability_margin") if mod_ev else conf.get("probability_margin")

        countries = bio.get("countries_represented_in_dataset") if bio else []
        country_str = "; ".join(countries) if countries else "None"

        rows.append({
            "sequence_id": inp.get("sequence_id"),
            "sequence_length": inp.get("length"),
            "qc_status": qc.get("status"),
            "prediction_status": status,
            "confidence_tier": pred.get("confidence_tier"),
            "kingdom": pred.get("kingdom"),
            "class": pred.get("class"),
            "order": pred.get("order"),
            "family": pred.get("family"),
            "genus": pred.get("genus"),
            "species": pred.get("species"),
            "top1_probability": top1_prob,
            "probability_margin": prob_margin,
            "nearest_reference": sim.get("nearest_reference_id") if sim else None,
            "nearest_distance": sim.get("nearest_distance") if sim else None,
            "nearest_similarity": sim.get("nearest_similarity") if sim else None,
            "is_exact_match": is_exact,
            "is_out_of_reference": is_out,
            "supporting_reference_records": bio.get("supporting_reference_records_in_dataset", 0) if bio else 0,
            "countries_represented": country_str,
            "explanation": ev.get("explanation") if ev else None
        })

    return pd.DataFrame(rows)


def export_batch_results_csv(
    results: List[Dict[str, Any]],
    output_path: str
) -> pd.DataFrame:
    """
    Export batch prediction results to a clean CSV file.
    """
    df = batch_results_to_dataframe(results)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
