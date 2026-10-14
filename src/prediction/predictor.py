"""
src/prediction/predictor.py
Phase 8: BioScan End-to-End Prediction Engine

The unified biological classification and biodiversity evidence engine for BioScan.
Integrates:
1. DNA Quality Control (QC)
2. DNA Feature Extraction (k-mer and compositional features)
3. Hierarchical Taxonomic Classification (Class -> Order -> Family -> Genus -> Species)
4. Reference Library Similarity Search (Normalized Levenshtein edit distance, Top-k)
5. Multi-Stream Confidence Assessment & Distance Veto Out-of-Reference Detection
6. Scientific Evidence Aggregation & Biodiversity Occurrence Reporting
7. Batch & FASTA Prediction Integration
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.features.dna_features import (
    KMerEncoder,
    extract_basic_and_complexity_features
)
from src.similarity.reference_search import (
    get_reference_searcher,
    ReferenceSearcher
)
from .input_validation import (
    validate_dna_sequence,
    validate_and_clean_sequence
)
from .evidence import (
    compute_probability_metrics,
    evaluate_taxonomic_concordance,
    synthesize_evidence_and_status,
    THRESHOLD_CLOSE_SIMILARITY,
    THRESHOLD_OUT_OF_REFERENCE
)
from .result_schema import (
    format_prediction_result,
    format_qc_failure_result,
    DEFAULT_DISCLAIMER
)

MODELS_DIR = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TAXONOMY_REF_PATH = os.path.join(PROCESSED_DIR, "taxonomy_reference.csv")
REF_BARCODES_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
RECORDS_PATH = os.path.join(PROCESSED_DIR, "biodiversity_reference_records.csv")
DIST_PATH = os.path.join(PROCESSED_DIR, "species_distribution_summary.csv")

TAXONOMY_LEVELS = ["class", "order", "family", "genus", "species"]


class BioScanPredictor:
    """
    Unified BioScan production prediction engine.
    Loads and caches models, feature encoders, reference library, and biodiversity occurrences.
    """
    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        ref_path: str = REF_BARCODES_PATH,
        tax_ref_path: str = TAXONOMY_REF_PATH,
        records_path: str = RECORDS_PATH,
        dist_path: str = DIST_PATH
    ):
        self.models_dir = models_dir
        self.ref_path = ref_path
        self.tax_ref_path = tax_ref_path
        self.records_path = records_path
        self.dist_path = dist_path

        # Validate and load artifacts
        self.artifacts: Dict[str, Any] = {}
        self._load_and_validate_models()
        self._load_taxonomy_hierarchy()
        self._load_biodiversity_reference_data()

        # Shared searcher and encoders
        self.searcher = get_reference_searcher()
        self.k2_encoder = KMerEncoder(k=2)
        self.k3_encoder = KMerEncoder(k=3)

    def _load_and_validate_models(self):
        """Validate artifact integrity and load the 5 Phase 5 hierarchical models."""
        for lvl in TAXONOMY_LEVELS:
            path = os.path.join(self.models_dir, f"taxonomy_{lvl}_model.joblib")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing validated model artifact for level '{lvl}': {path}")

            artifact = joblib.load(path)
            
            # Verify artifact schema
            required_keys = ["model", "scaler", "feature_set", "feature_columns", "target_classes", "taxonomy_level"]
            for key in required_keys:
                if key not in artifact:
                    raise KeyError(f"Model artifact for '{lvl}' is missing required key: {key}")

            # Verify scaler dimensionality matches feature columns
            n_cols = len(artifact["feature_columns"])
            if hasattr(artifact["scaler"], "n_features_in_"):
                if artifact["scaler"].n_features_in_ != n_cols:
                    raise ValueError(
                        f"Dimension mismatch in '{lvl}' artifact: scaler expects {artifact['scaler'].n_features_in_} "
                        f"features, but feature_columns has {n_cols}."
                    )

            self.artifacts[lvl] = artifact

    def _load_taxonomy_hierarchy(self):
        """Load parent-child and child-parent hierarchy maps."""
        if not os.path.exists(self.tax_ref_path):
            raise FileNotFoundError(f"Missing taxonomy reference file: {self.tax_ref_path}")

        tax_df = pd.read_csv(self.tax_ref_path)
        self.child_to_parent = {
            "order": dict(tax_df[["order", "class"]].dropna().drop_duplicates().values),
            "family": dict(tax_df[["family", "order"]].dropna().drop_duplicates().values),
            "genus": dict(tax_df[["genus", "family"]].dropna().drop_duplicates().values),
            "species": dict(tax_df[["species", "genus"]].dropna().drop_duplicates().values),
        }
        self.parent_to_children = {
            "class": {}, "order": {}, "family": {}, "genus": {}
        }
        for child, parent in self.child_to_parent["order"].items():
            self.parent_to_children["class"].setdefault(parent, set()).add(child)
        for child, parent in self.child_to_parent["family"].items():
            self.parent_to_children["order"].setdefault(parent, set()).add(child)
        for child, parent in self.child_to_parent["genus"].items():
            self.parent_to_children["family"].setdefault(parent, set()).add(child)
        for child, parent in self.child_to_parent["species"].items():
            self.parent_to_children["genus"].setdefault(parent, set()).add(child)

    def _load_biodiversity_reference_data(self):
        """Load occurrence and species distribution tables for context integration."""
        self.records_df = pd.read_csv(self.records_path) if os.path.exists(self.records_path) else None
        self.dist_df = pd.read_csv(self.dist_path) if os.path.exists(self.dist_path) else None

    def _extract_features_for_model(self, clean_seq: str, artifact: Dict) -> np.ndarray:
        """Extract and assemble feature vectors matching a specific model artifact's columns."""
        feat_set = artifact["feature_set"]
        cols = artifact["feature_columns"]

        if feat_set == "Basic":
            raw_dict = extract_basic_and_complexity_features(clean_seq)
        elif feat_set == "k=2":
            raw_dict = self.k2_encoder.encode_sequence(clean_seq)
        elif feat_set == "k=3":
            raw_dict = self.k3_encoder.encode_sequence(clean_seq)
        elif feat_set == "Basic + k=2":
            raw_dict = extract_basic_and_complexity_features(clean_seq)
            raw_dict.update(self.k2_encoder.encode_sequence(clean_seq))
        else:
            raise ValueError(f"Unsupported feature set: {feat_set}")

        # Assemble in exact column order expected by the scaler
        feat_vector = np.array([raw_dict[col] for col in cols]).reshape(1, -1)
        scaled_vector = artifact["scaler"].transform(feat_vector)
        return scaled_vector

    def _retrieve_biodiversity_context(
        self,
        status: str,
        nearest_ref: Dict[str, Any],
        predicted_tax: Dict[str, Optional[str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Synthesize spatial and survey occurrence metadata:
        - If known_reference_like: provides supporting reference occurrence records.
        - If low_confidence: provides nearest reference metadata as cautionary reference context.
        - If out_of_reference: STRICTLY FORBIDS projecting a species geographic distribution!
        """
        nearest_seq_id = nearest_ref.get("sequence_id")

        if status == "out_of_reference":
            return {
                "projection_status": "NO_PROJECTION_PERMITTED",
                "reference_barcode_id": None,
                "supporting_reference_records_in_dataset": 0,
                "countries_represented_in_dataset": [],
                "survey_region": None,
                "nearest_reference_context": {
                    "nearest_reference_id": nearest_seq_id,
                    "nearest_reference_distance": nearest_ref.get("distance"),
                    "nearest_reference_similarity": nearest_ref.get("similarity"),
                    "nearest_reference_taxonomy": {
                        "class": nearest_ref.get("class"),
                        "order": nearest_ref.get("order"),
                        "family": nearest_ref.get("family"),
                        "genus": nearest_ref.get("genus"),
                        "species": nearest_ref.get("species")
                    },
                    "context_note": "Provenance of nearest cataloged barcode provided solely for distance reference."
                },
                "scientific_scope_note": (
                    "Query sequence exhibits substantial divergence (distance > 0.150) from all 101 reference sequences "
                    "in this library. No species or geographic distribution can be projected from the reference data."
                )
            }

        # Query occurrence records for nearest_seq_id
        if self.records_df is None or nearest_seq_id is None:
            return None

        matched = self.records_df[self.records_df["sequence_id"] == nearest_seq_id]
        n_supporting = len(matched)

        if n_supporting == 0:
            stations = []
            depth_range = None
            dates = None
        else:
            stations = matched[["decimalLatitude", "decimalLongitude", "sampname", "depth"]].drop_duplicates().to_dict(orient="records")
            depth_range = (float(matched["depth"].min()), float(matched["depth"].max()))
            dates = (str(matched["eventDate"].min())[:10], str(matched["eventDate"].max())[:10])

        if status == "known_reference_like":
            proj_status = "SUPPORTED_BY_REFERENCE_RECORDS"
            scope_note = (
                f"Reference records for sequence ({nearest_seq_id}) in the available dataset "
                "were collected offshore New Jersey, USA (NJ OTS survey 2019). "
                "This describes the observed reference-record distribution in this dataset, "
                "NOT the global biological species range."
            )
        else: # low_confidence
            proj_status = "CAUTIONARY_REFERENCE_CONTEXT_ONLY"
            scope_note = (
                "Query sequence has moderate divergence or taxonomic ambiguity. "
                "Geographic data represents the NEAREST reference sequence in the catalog, "
                "provided strictly as contextual reference evidence, not as a confirmed specimen location."
            )

        return {
            "projection_status": proj_status,
            "reference_barcode_id": nearest_seq_id,
            "taxon_identified": predicted_tax.get("species") or predicted_tax.get("genus"),
            "supporting_reference_records_in_dataset": n_supporting,
            "countries_represented_in_dataset": ["United States"] if n_supporting > 0 else [],
            "survey_region": "New Jersey Continental Shelf (Northwest Atlantic Ocean)",
            "stations_detected_count": len(stations),
            "depth_range_meters": depth_range,
            "dates_detected": dates,
            "sampling_stations_sample": stations[:5],
            "scientific_scope_note": scope_note
        }

    def predict(
        self,
        sequence: Any,
        sequence_id: Optional[str] = None,
        top_k_similar: int = 5
    ) -> Dict[str, Any]:
        """
        Execute end-to-end biological prediction for a single DNA sequence.
        """
        # 1. Biological Quality Control (QC)
        qc_ok, clean_seq, warnings, errors, qc_metrics = validate_dna_sequence(sequence, sequence_id=sequence_id)
        if not qc_ok:
            return format_qc_failure_result(sequence, sequence_id, errors, warnings)

        seq_id = sequence_id or "QUERY_001"
        seq_len = len(clean_seq)

        # 2. Reference Library Similarity Search (Normalized Levenshtein edit distance)
        similar_refs = self.searcher.search(clean_seq, top_k=max(top_k_similar, 2))
        nearest_ref = similar_refs[0] if similar_refs else {}
        second_ref = similar_refs[1] if len(similar_refs) > 1 else nearest_ref
        nearest_dist = nearest_ref.get("distance", 1.0)
        second_dist = second_ref.get("distance", nearest_dist)
        is_exact = nearest_ref.get("is_exact_match", False)

        # 3. Model Predictions (Class posterior probabilities per rank)
        level_probs = {}
        level_metrics = {}
        for lvl in TAXONOMY_LEVELS:
            art = self.artifacts[lvl]
            X_scaled = self._extract_features_for_model(clean_seq, art)
            probs = art["model"].predict_proba(X_scaled)[0]
            level_probs[lvl] = probs
            level_metrics[lvl] = compute_probability_metrics(probs)

        # 4. Top-Down Parent-Restricted Hierarchical Inference
        predicted_tax: Dict[str, Optional[str]] = {}

        # 4a. Class
        art_cls = self.artifacts["class"]
        p_c = level_probs["class"]
        cls_idx = np.argmax(p_c)
        pred_cls = art_cls["target_classes"][cls_idx]
        predicted_tax["class"] = pred_cls

        # 4b. Order (constrained by pred_cls)
        art_ord = self.artifacts["order"]
        valid_orders = self.parent_to_children["class"].get(pred_cls, set())
        p_o = level_probs["order"].copy()
        ord_classes = art_ord["target_classes"]
        mask_o = np.array([o in valid_orders for o in ord_classes])
        if mask_o.any():
            p_o[~mask_o] = -1.0
            pred_ord = ord_classes[np.argmax(p_o)]
        else:
            pred_ord = ord_classes[np.argmax(level_probs["order"])]
        predicted_tax["order"] = pred_ord

        # 4c. Family (constrained by pred_ord)
        art_fam = self.artifacts["family"]
        valid_fams = self.parent_to_children["order"].get(pred_ord, set())
        p_f = level_probs["family"].copy()
        fam_classes = art_fam["target_classes"]
        mask_f = np.array([f in valid_fams for f in fam_classes])
        if mask_f.any():
            p_f[~mask_f] = -1.0
            pred_fam = fam_classes[np.argmax(p_f)]
        else:
            pred_fam = fam_classes[np.argmax(level_probs["family"])]
        predicted_tax["family"] = pred_fam

        # 4d. Genus (constrained by pred_fam)
        art_gen = self.artifacts["genus"]
        valid_genera = self.parent_to_children["family"].get(pred_fam, set())
        p_g = level_probs["genus"].copy()
        gen_classes = art_gen["target_classes"]
        mask_g = np.array([g in valid_genera for g in gen_classes])
        if mask_g.any():
            p_g[~mask_g] = -1.0
            pred_gen = gen_classes[np.argmax(p_g)]
        else:
            pred_gen = "Not available in reference data"
        predicted_tax["genus"] = pred_gen

        # 4e. Species (constrained by pred_gen and distance <= 0.05)
        art_sp = self.artifacts["species"]
        valid_species = self.parent_to_children["genus"].get(pred_gen, set())
        p_s = level_probs["species"].copy()
        sp_classes = art_sp["target_classes"]
        mask_s = np.array([s in valid_species for s in sp_classes])
        
        if mask_s.any() and (is_exact or nearest_dist <= THRESHOLD_CLOSE_SIMILARITY):
            p_s[~mask_s] = -1.0
            pred_sp = sp_classes[np.argmax(p_s)]
        else:
            pred_sp = "Not available in reference data"
        predicted_tax["species"] = pred_sp

        # 5. Taxonomic Concordance & Multi-Stream Evidence Synthesis
        concordance = evaluate_taxonomic_concordance(predicted_tax, nearest_ref)
        status, conf_tier, evidence_summary = synthesize_evidence_and_status(
            nearest_dist=nearest_dist,
            nearest_ref=nearest_ref,
            prob_metrics=level_metrics["class"],
            concordance=concordance,
            is_exact_match=is_exact,
            second_dist=second_dist
        )

        # 6. Biodiversity Occurrence Synthesis
        biodiversity_context = self._retrieve_biodiversity_context(status, nearest_ref, predicted_tax)

        # 7. Assemble Structured Contract
        input_info = {
            "sequence_id": seq_id,
            "sequence": clean_seq,
            "length": seq_len
        }
        qc_info = {
            "status": "passed",
            "is_valid": True,
            "warnings": warnings,
            "errors": [],
            "metrics": qc_metrics
        }
        prediction_info = {
            "status": status,
            "confidence_tier": conf_tier,
            "kingdom": "Animalia",
            "class": predicted_tax["class"],
            "order": predicted_tax["order"],
            "family": predicted_tax["family"],
            "genus": predicted_tax["genus"],
            "species": predicted_tax["species"],
            "species_prediction_scope": "reference-library constrained"
        }
        confidence_info = {
            "top1_probability": level_metrics["class"]["top1_prob"],
            "top2_probability": level_metrics["class"]["top2_prob"],
            "probability_margin": level_metrics["class"]["prob_margin"],
            "entropy": level_metrics["class"]["entropy"],
            "level_confidences": level_metrics
        }
        similarity_info = {
            "nearest_reference_id": nearest_ref.get("sequence_id"),
            "nearest_distance": nearest_dist,
            "nearest_similarity": nearest_ref.get("similarity"),
            "is_exact_match": is_exact,
            "distance_margin_to_second": round(second_dist - nearest_dist, 4),
            "top_references": similar_refs[:top_k_similar]
        }

        # Preserve legacy root-level keys for backward-compatibility with Phase 6 & 7
        result = format_prediction_result(
            input_info=input_info,
            qc_info=qc_info,
            prediction_info=prediction_info,
            confidence_info=confidence_info,
            similarity_info=similarity_info,
            evidence_info=evidence_summary,
            biodiversity_info=biodiversity_context
        )
        # Root-level backward-compatibility aliases
        result["qc_status"] = "PASSED"
        result["status"] = status
        result["confidence_tier"] = conf_tier
        result["predicted_taxonomy"] = predicted_tax
        result["level_confidences"] = level_metrics
        result["nearest_references"] = similar_refs[:top_k_similar]

        return result

    def predict_batch(
        self,
        sequences: List[Union[str, Dict[str, str]]],
        top_k_similar: int = 5
    ) -> List[Dict[str, Any]]:
        """Execute independent predictions over a list of sequences."""
        from .batch_predict import predict_batch
        return predict_batch(sequences, predictor=self, top_k_similar=top_k_similar)

    def predict_fasta(
        self,
        fasta_input: Union[str, os.PathLike],
        top_k_similar: int = 5
    ) -> List[Dict[str, Any]]:
        """Parse FASTA input and execute independent predictions."""
        from .batch_predict import predict_fasta
        return predict_fasta(fasta_input, predictor=self, top_k_similar=top_k_similar)


# -------------------------------------------------------------
# Global Singleton Instance & Public API
# -------------------------------------------------------------
_PREDICTOR_INSTANCE: Optional[BioScanPredictor] = None

def get_predictor() -> BioScanPredictor:
    """Obtain or initialize the global BioScanPredictor singleton instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = BioScanPredictor()
    return _PREDICTOR_INSTANCE


def predict_sequence(
    sequence: Any,
    sequence_id: Optional[str] = None,
    top_k_similar: int = 5
) -> Dict[str, Any]:
    """
    Main user-facing API: classify an input DNA sequence and generate
    evidence-based confidence and biodiversity assessment.
    """
    engine = get_predictor()
    return engine.predict(sequence, sequence_id=sequence_id, top_k_similar=top_k_similar)


def predict_batch(
    sequences: List[Union[str, Dict[str, str]]],
    top_k_similar: int = 5
) -> List[Dict[str, Any]]:
    """Execute batch predictions across multiple sequences."""
    engine = get_predictor()
    return engine.predict_batch(sequences, top_k_similar=top_k_similar)


def predict_fasta(
    fasta_input: Union[str, os.PathLike],
    top_k_similar: int = 5
) -> List[Dict[str, Any]]:
    """Parse FASTA input and predict all sequences."""
    engine = get_predictor()
    return engine.predict_fasta(fasta_input, top_k_similar=top_k_similar)


def export_predictions_csv(
    results: List[Dict[str, Any]],
    output_path: str
) -> pd.DataFrame:
    """Export prediction results to a flat CSV file."""
    from .batch_predict import export_batch_results_csv
    return export_batch_results_csv(results, output_path)
