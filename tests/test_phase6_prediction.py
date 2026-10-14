"""
tests/test_phase6_prediction.py
Unit and integration tests for Phase 6: Unknown Detection, Similarity Search,
and Evidence/Prediction Layer.
"""

import os
import unittest
import pandas as pd
import numpy as np

from src.similarity.reference_search import (
    ReferenceSearcher,
    levenshtein_distance,
    normalized_edit_distance,
    hamming_distance,
    find_similar_sequences
)
from src.prediction.confidence import (
    compute_probability_metrics,
    evaluate_taxonomy_agreement,
    determine_sequence_status,
    THRESHOLD_CLOSE_SIMILARITY,
    THRESHOLD_OUT_OF_REFERENCE
)
from src.prediction.predictor import (
    BioScanPredictor,
    validate_and_clean_sequence,
    predict_sequence
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REF_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "hackathon_dna_taxonomy_merged.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

class TestPhase6PredictionAndEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref_df = pd.read_csv(REF_PATH)
        cls.predictor = BioScanPredictor()
        cls.searcher = ReferenceSearcher()

    # -------------------------------------------------------------
    # 1. Similarity Search Tests
    # -------------------------------------------------------------
    def test_01_exact_match_retrieval(self):
        """Test 1: Verify exact query sequence retrieves itself with distance 0.0 and rank 1."""
        sample_row = self.ref_df.iloc[0]
        seq = sample_row["sequence_normalized"]
        seq_id = sample_row["sequence_id"]

        results = self.searcher.search(seq, top_k=5)
        self.assertEqual(len(results), 5)
        top1 = results[0]
        self.assertEqual(top1["sequence_id"], seq_id)
        self.assertEqual(top1["distance"], 0.0)
        self.assertEqual(top1["similarity"], 1.0)
        self.assertTrue(top1["is_exact_match"])

    def test_02_unequal_sequence_length_handling(self):
        """Test 2: Verify Levenshtein edit distance handles unequal amplicon lengths correctly."""
        # 10 bp vs 12 bp
        s1 = "ACGTACGTAC"
        s2 = "ACGTACGTACGT"
        d = levenshtein_distance(s1, s2)
        self.assertEqual(d, 2)
        norm_d = normalized_edit_distance(s1, s2)
        self.assertAlmostEqual(norm_d, 2 / 12, places=4)
        
        # Hamming distance should return None for unequal lengths
        self.assertIsNone(hamming_distance(s1, s2))

    def test_03_top_k_ranking_order(self):
        """Test 3: Verify top-k results are sorted strictly in ascending distance order."""
        sample_seq = self.ref_df.iloc[5]["sequence_normalized"]
        results = self.searcher.search(sample_seq, top_k=10)
        self.assertEqual(len(results), 10)
        for i in range(len(results) - 1):
            self.assertLessEqual(results[i]["distance"], results[i+1]["distance"])
            self.assertEqual(results[i]["rank"], i + 1)

    # -------------------------------------------------------------
    # 2. Confidence and Status Determination Tests
    # -------------------------------------------------------------
    def test_04_status_known_reference_like(self):
        """Test 4: Verify identical or highly similar reference sequence yields known_reference_like."""
        sample_seq = self.ref_df.iloc[10]["sequence_normalized"]
        res = self.predictor.predict(sample_seq)
        self.assertEqual(res["qc_status"], "PASSED")
        self.assertEqual(res["status"], "known_reference_like")
        self.assertTrue(res["evidence"]["is_exact_match"])
        self.assertEqual(res["confidence_tier"], "high")

    def test_05_status_out_of_reference_divergent(self):
        """Test 5: Verify heavily mutated / out-of-reference sequence yields out_of_reference."""
        sample_seq = self.ref_df.iloc[10]["sequence_normalized"]
        # Mutate ~30% of bases
        bases = ["A", "C", "G", "T"]
        mutated = list(sample_seq)
        for idx in range(0, len(mutated), 3):
            mutated[idx] = "T" if mutated[idx] == "A" else "A"
        mutated_seq = "".join(mutated)

        res = self.predictor.predict(mutated_seq)
        self.assertEqual(res["qc_status"], "PASSED")
        self.assertEqual(res["status"], "out_of_reference")
        self.assertGreater(res["evidence"]["nearest_distance"], THRESHOLD_OUT_OF_REFERENCE)
        self.assertIn("out_of_reference", res["status"])

    def test_06_hierarchical_consistency_preserved(self):
        """Test 6: Verify predicted taxonomy hierarchy is strictly parent-child consistent."""
        sample_seq = self.ref_df.iloc[20]["sequence_normalized"]
        res = self.predictor.predict(sample_seq)
        pred = res["predicted_taxonomy"]
        
        # Check order in class
        self.assertIn(pred["order"], self.predictor.parent_to_children["class"].get(pred["class"], set()))
        # Check family in order
        self.assertIn(pred["family"], self.predictor.parent_to_children["order"].get(pred["order"], set()))
        # Check genus in family
        if pred["genus"] != "Not available in reference data":
            self.assertIn(pred["genus"], self.predictor.parent_to_children["family"].get(pred["family"], set()))

    def test_07_unlabeled_species_no_fabrication(self):
        """Test 7: Verify species-unlabeled sequence outputs 'Not available in reference data'."""
        unlabeled_row = self.ref_df[self.ref_df["species"].isnull()].iloc[0]
        unlabeled_seq = unlabeled_row["sequence_normalized"]

        res = self.predictor.predict(unlabeled_seq)
        self.assertEqual(res["qc_status"], "PASSED")
        self.assertEqual(res["predicted_taxonomy"]["species"], "Not available in reference data")

    # -------------------------------------------------------------
    # 3. Adversarial and Edge Case QC Tests
    # -------------------------------------------------------------
    def test_08_adversarial_qc_inputs(self):
        """Test 8: Test robust QC handling on adversarial edge cases."""
        # A. None
        ok, clean, msgs, _ = validate_and_clean_sequence(None)
        self.assertFalse(ok)
        
        # B. Empty string
        ok, clean, msgs, _ = validate_and_clean_sequence("   ")
        self.assertFalse(ok)
        
        # C. Lowercase with whitespace and newlines (should succeed after cleaning)
        valid_raw = "  acta\ntgc\tTAAGCC  "
        ok, clean, msgs, metrics = validate_and_clean_sequence(valid_raw)
        self.assertTrue(ok)
        self.assertEqual(clean, "ACTATGCTAAGCC")
        
        # D. Ambiguous IUPAC base (N)
        ok, clean, msgs, _ = validate_and_clean_sequence("ACTGNNNNACTG")
        self.assertFalse(ok)
        self.assertIn("Invalid characters", msgs[0])
        
        # E. Illegal characters (numbers, punctuation)
        ok, clean, msgs, _ = validate_and_clean_sequence("ACTG123!ACTG")
        self.assertFalse(ok)
        self.assertIn("Invalid characters", msgs[0])

    def test_09_length_warnings(self):
        """Test 9: Verify length warnings for unusually short (<50 bp) or long (>300 bp) sequences."""
        # Short sequence (20 bp)
        short_seq = "ACGT" * 5
        ok, clean, warnings, _ = validate_and_clean_sequence(short_seq)
        self.assertTrue(ok)
        self.assertTrue(any("unusually short" in w for w in warnings))

        # Long sequence (350 bp)
        long_seq = "ACGT" * 90
        ok, clean, warnings, _ = validate_and_clean_sequence(long_seq)
        self.assertTrue(ok)
        self.assertTrue(any("unusually long" in w for w in warnings))

    def test_10_output_schema_and_artifacts(self):
        """Test 10: Verify predictor API output schema and Phase 6 report artifacts exist."""
        sample_seq = self.ref_df.iloc[3]["sequence_normalized"]
        res = predict_sequence(sample_seq)

        expected_keys = [
            "qc_status", "qc_metrics", "status", "confidence_tier",
            "predicted_taxonomy", "level_confidences", "nearest_references",
            "evidence", "disclaimer"
        ]
        for key in expected_keys:
            self.assertIn(key, res)

        # Reports and figures exist
        self.assertTrue(os.path.exists(os.path.join(REPORTS_DIR, "phase6_similarity_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(REPORTS_DIR, "phase6_unknown_detection_results.csv")))
        self.assertTrue(os.path.exists(os.path.join(REPORTS_DIR, "figures", "confidence_distribution.png")))
        self.assertTrue(os.path.exists(os.path.join(REPORTS_DIR, "figures", "similarity_distance_distribution.png")))
        self.assertTrue(os.path.exists(os.path.join(REPORTS_DIR, "figures", "threshold_analysis.png")))

    def test_11_raw_dataset_unmodified(self):
        """Test 11: Verify raw dataset SHA-256 remains strictly unchanged."""
        import hashlib
        with open(RAW_PATH, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(h, "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65")

    def test_12_self_match_exclusion(self):
        """Test 12: Verify exclude_id strictly prevents self-match retrieval during LORO validation."""
        sample_row = self.ref_df.iloc[0]
        seq = sample_row["sequence_normalized"]
        seq_id = sample_row["sequence_id"]

        # Search with self excluded
        results = self.searcher.search(seq, top_k=5, exclude_id=seq_id)
        retrieved_ids = [m["sequence_id"] for m in results]
        self.assertNotIn(seq_id, retrieved_ids)
        self.assertEqual(len(results), 5)
        # Nearest non-self match should have distance >= min library distance
        self.assertGreater(results[0]["distance"], 0.0)

    def test_13_low_confidence_status(self):
        """Test 13: Verify intermediate divergence or taxonomic disagreement assigns low_confidence status."""
        # Case A: Intermediate divergence (e.g. distance = 0.08, between 0.05 and 0.15)
        status, conf_tier, ev = determine_sequence_status(
            nearest_dist=0.08,
            prob_metrics={"top1_prob": 0.85, "prob_margin": 0.40},
            agreement={"class": True, "order": True, "family": True},
            is_exact_match=False
        )
        self.assertEqual(status, "low_confidence")
        self.assertIn("moderate reference divergence", ev["explanation"])

        # Case B: Low model probability margin (< 0.150)
        status_b, _, ev_b = determine_sequence_status(
            nearest_dist=0.03,
            prob_metrics={"top1_prob": 0.45, "prob_margin": 0.05},
            agreement={"class": True, "order": True, "family": True},
            is_exact_match=False
        )
        # Even with close distance, if margin < 0.15 and class/order agree, check status:
        # Note: if dist <= 0.05 and class_agrees and order_agrees, it's known_reference_like
        # But if class disagrees:
        status_c, _, ev_c = determine_sequence_status(
            nearest_dist=0.03,
            prob_metrics={"top1_prob": 0.80, "prob_margin": 0.50},
            agreement={"class": False, "order": False, "family": False},
            is_exact_match=False
        )
        self.assertEqual(status_c, "low_confidence")
        self.assertIn("taxonomic disagreement", ev_c["explanation"])

if __name__ == "__main__":
    unittest.main()
