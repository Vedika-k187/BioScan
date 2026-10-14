"""
src/biodiversity/prediction_biodiversity.py
Phase 7: Prediction-to-Biodiversity Integration Layer

Connects Phase 6 taxonomy/status prediction with reference occurrence metadata,
geographic sampling distributions, and biodiversity statistics.

Strictly enforces:
- "supporting reference records" means records in THIS dataset.
- out_of_reference sequences NEVER project a species distribution.
- Clear disclaimers: observed reference-record distribution != complete biological species range.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from src.prediction.predictor import predict_sequence, get_predictor

PROCESSED_DIR = os.path.join("data", "processed")
RECORDS_PATH = os.path.join(PROCESSED_DIR, "biodiversity_reference_records.csv")
REF_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
DIST_PATH = os.path.join(PROCESSED_DIR, "species_distribution_summary.csv")


class BioScanBiodiversityIntelligence:
    """
    Integrates classifier predictions with spatial biodiversity reference data.
    """
    def __init__(
        self,
        records_path: str = RECORDS_PATH,
        ref_path: str = REF_PATH,
        dist_path: str = DIST_PATH
    ):
        self.ref_df = pd.read_csv(ref_path) if os.path.exists(ref_path) else None
        self.records_df = pd.read_csv(records_path) if os.path.exists(records_path) else None
        self.dist_df = pd.read_csv(dist_path) if os.path.exists(dist_path) else None

    def _get_reference_occurrence_summary(self, seq_id: str) -> Dict[str, Any]:
        """Extract geographic and survey occurrence statistics for a specific sequence_id."""
        if self.records_df is None or self.records_df.empty:
            return {"supporting_reference_records_count": 0, "status": "No occurrence database loaded"}

        matched = self.records_df[self.records_df["sequence_id"] == seq_id]
        if matched.empty:
            return {
                "sequence_id": seq_id,
                "supporting_reference_records_count": 0,
                "survey_detection_status": "Assayed in survey but 0 positive field detections (negative screening across 68 samples)",
                "countries_represented": ["United States"],
                "stations_detected_count": 0,
                "depth_range_meters": None,
                "sampling_stations": []
            }

        stations = matched[["decimalLatitude", "decimalLongitude", "sampname", "depth"]].drop_duplicates().to_dict(orient="records")
        min_depth = float(matched["depth"].min())
        max_depth = float(matched["depth"].max())
        mean_reads = round(float(matched["organismQuantity"].mean()), 1)

        return {
            "sequence_id": seq_id,
            "supporting_reference_records_count": len(matched),
            "survey_detection_status": f"Positive field detection in {len(matched)} survey samples",
            "countries_represented": ["United States"],
            "state_province": "New Jersey",
            "water_body": "Atlantic Ocean",
            "stations_detected_count": len(stations),
            "depth_range_meters": (min_depth, max_depth),
            "mean_read_count": mean_reads,
            "sampling_stations_sample": stations[:5],
            "dates_detected": (str(matched["eventDate"].min())[:10], str(matched["eventDate"].max())[:10])
        }

    def analyze(self, sequence: Any, top_k_similar: int = 5) -> Dict[str, Any]:
        """
        Analyze an input DNA sequence:
        QC -> Hierarchical Prediction -> Similarity Search -> Evidence Synthesis -> Geographic Integration.
        """
        # 1. Run Phase 6 Prediction
        pred = predict_sequence(sequence, top_k_similar=top_k_similar)

        if pred["qc_status"] != "PASSED":
            return {
                "qc_status": "FAILED",
                "qc_errors": pred.get("qc_errors", []),
                "status": "qc_failed",
                "prediction": None,
                "biodiversity_intelligence": None,
                "disclaimer": "Sequence failed biological quality control. No biodiversity analysis attempted."
            }

        status = pred["status"]
        confidence_tier = pred["confidence_tier"]
        predicted_tax = pred["predicted_taxonomy"]
        evidence = pred["evidence"]
        nearest_refs = pred["nearest_references"]
        nearest_seq_id = nearest_refs[0]["sequence_id"] if nearest_refs else None
        nearest_dist = nearest_refs[0]["distance"] if nearest_refs else 1.0
        nearest_sim = nearest_refs[0]["similarity"] if nearest_refs else 0.0

        # 2. Status-Dependent Biodiversity Synthesis
        if status == "known_reference_like":
            # Confident match to cataloged barcode
            occ_summary = self._get_reference_occurrence_summary(nearest_seq_id)
            geo_evidence = {
                "projection_status": "SUPPORTED_BY_REFERENCE_RECORDS",
                "reference_barcode_id": nearest_seq_id,
                "taxon_identified": predicted_tax.get("species") or predicted_tax.get("genus"),
                "supporting_reference_records_in_dataset": occ_summary["supporting_reference_records_count"],
                "countries_represented_in_dataset": occ_summary["countries_represented"],
                "ocean_basin": "Northwest Atlantic Ocean",
                "stations_detected_count": occ_summary["stations_detected_count"],
                "depth_range_meters": occ_summary["depth_range_meters"],
                "sampling_stations_sample": occ_summary.get("sampling_stations_sample", []),
                "scientific_scope_note": (
                    f"Reference records for this sequence ({nearest_seq_id}) in the available dataset "
                    "were collected offshore New Jersey, USA (NJ OTS survey 2019). "
                    "This describes the observed reference-record distribution in this dataset, "
                    "NOT the global biological species range."
                )
            }
            uncertainty_warning = None

        elif status == "low_confidence":
            # Ambiguous or moderate distance
            occ_summary = self._get_reference_occurrence_summary(nearest_seq_id)
            geo_evidence = {
                "projection_status": "CAUTIONARY_REFERENCE_CONTEXT_ONLY",
                "reference_barcode_id": nearest_seq_id,
                "candidate_taxonomy": predicted_tax,
                "supporting_reference_records_in_dataset": occ_summary["supporting_reference_records_count"],
                "countries_represented_in_dataset": occ_summary["countries_represented"],
                "stations_detected_count": occ_summary["stations_detected_count"],
                "depth_range_meters": occ_summary["depth_range_meters"],
                "scientific_scope_note": (
                    "Query sequence has moderate divergence or taxonomic ambiguity. "
                    "Geographic data shown above represents the NEAREST reference sequence in the catalog, "
                    "provided strictly as contextual reference evidence, not as a confirmed specimen location."
                )
            }
            uncertainty_warning = (
                f"Moderate confidence prediction (distance to nearest reference: {nearest_dist:.4f}). "
                "Class- or Order-level assignment is supported, but lower ranks require manual confirmation."
            )

        else: # out_of_reference
            # Divergent sequence outside reference library bounds
            occ_summary = self._get_reference_occurrence_summary(nearest_seq_id)
            geo_evidence = {
                "projection_status": "NO_PROJECTION_PERMITTED",
                "reference_barcode_id": None,
                "supporting_reference_records_in_dataset": 0,
                "countries_represented_in_dataset": [],
                "nearest_reference_context": {
                    "nearest_reference_id": nearest_seq_id,
                    "nearest_reference_distance": nearest_dist,
                    "nearest_reference_similarity": nearest_sim,
                    "nearest_reference_taxonomy": {
                        "class": nearest_refs[0].get("class"),
                        "order": nearest_refs[0].get("order"),
                        "family": nearest_refs[0].get("family"),
                        "genus": nearest_refs[0].get("genus"),
                        "species": nearest_refs[0].get("species")
                    },
                    "nearest_reference_stations_count": occ_summary["stations_detected_count"],
                    "nearest_reference_depth_range": occ_summary["depth_range_meters"],
                    "context_note": "Provenance of nearest cataloged barcode provided solely for distance reference."
                },
                "scientific_scope_note": (
                    "Query sequence exhibits substantial divergence (distance > 0.150) from all 101 reference sequences "
                    "in this library. No species or geographic distribution can be projected from the reference data."
                )
            }
            uncertainty_warning = (
                "Sequence flagged as 'out_of_reference'. This indicates lack of representation in the 101-barcode "
                "reference library; it does NOT constitute biological proof of a novel or undescribed species."
            )

        return {
            "qc_status": pred["qc_status"],
            "qc_metrics": pred.get("qc_metrics"),
            "status": status,
            "confidence_tier": confidence_tier,
            "predicted_taxonomy": predicted_tax,
            "nearest_references": nearest_refs,
            "prediction_evidence": evidence,
            "biodiversity_intelligence": geo_evidence,
            "uncertainty_warning": uncertainty_warning,
            "dataset_disclaimer": (
                "BioScan biodiversity analytics are derived from 101 curated 12S rDNA V5 reference barcodes "
                "and 2019 New Jersey Ocean Trawl Survey eDNA occurrence records. "
                "Observed patterns describe reference-record distributions and must not be interpreted as "
                "complete global species ranges or comprehensive population abundance."
            )
        }


# Global singleton instance
_INTELLIGENCE_INSTANCE: Optional[BioScanBiodiversityIntelligence] = None

def get_biodiversity_intelligence() -> BioScanBiodiversityIntelligence:
    """Obtain or initialize global BioScanBiodiversityIntelligence instance."""
    global _INTELLIGENCE_INSTANCE
    if _INTELLIGENCE_INSTANCE is None:
        _INTELLIGENCE_INSTANCE = BioScanBiodiversityIntelligence()
    return _INTELLIGENCE_INSTANCE

def analyze_prediction(sequence: Any, top_k_similar: int = 5) -> Dict[str, Any]:
    """
    Main user-facing API: classify an input DNA sequence and synthesize
    taxonomic, similarity, and geographic reference intelligence.
    """
    intelligence = get_biodiversity_intelligence()
    return intelligence.analyze(sequence, top_k_similar=top_k_similar)
