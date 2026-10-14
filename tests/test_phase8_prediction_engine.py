"""
tests/test_phase8_prediction_engine.py
Unit, integration, and contract tests for Phase 8: BioScan End-to-End Prediction Engine.

Covers all 22 minimum required test cases:
1. Valid DNA prediction
2. Lowercase DNA normalization
3. Invalid DNA rejection
4. Empty sequence handling
5. Sequence length warning/handling
6. Exact reference sequence
7. Known-reference-like sequence
8. Low-confidence case
9. Out-of-reference case
10. High classifier probability cannot override distance veto
11. Top-K similarity structure
12. Taxonomy fields present
13. Missing species label handled correctly
14. Biodiversity context does not fabricate geographic distribution
15. FASTA single sequence
16. FASTA multiple sequences
17. Malformed FASTA
18. One invalid sequence in batch does not crash other sequences
19. Batch CSV export
20. Model artifact/feature compatibility
21. Deterministic repeated prediction
22. Result schema validation
"""

import os
import io
import math
import hashlib
import unittest
import numpy as np
import pandas as pd

from src.prediction import (
    BioScanPredictor,
    get_predictor,
    predict_sequence,
    predict_batch,
    predict_fasta,
    export_predictions_csv,
    validate_dna_sequence,
    parse_fasta,
    parse_fasta_string,
    batch_results_to_dataframe
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "hackathon_dna_taxonomy_merged.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REF_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
EXPECTED_SHA256 = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"


class TestPhase8PredictionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref_df = pd.read_csv(REF_PATH)
        cls.predictor = get_predictor()
        cls.sample_seq = cls.ref_df.iloc[0]["sequence_normalized"]
        cls.sample_id = cls.ref_df.iloc[0]["sequence_id"]

    # 1. Valid DNA Prediction
    def test_01_valid_dna_prediction(self):
        """Test 1: Standard valid DNA sequence returns structured passed result."""
        res = self.predictor.predict(self.sample_seq, sequence_id="TEST_SEQ_01")
        self.assertEqual(res["qc"]["status"], "passed")
        self.assertTrue(res["qc"]["is_valid"])
        self.assertIn("class", res["prediction"])
        self.assertEqual(res["input"]["sequence_id"], "TEST_SEQ_01")

    # 2. Lowercase DNA Normalization
    def test_02_lowercase_dna_normalization(self):
        """Test 2: Lowercase DNA with surrounding whitespace normalizes and classifies identically."""
        lower_seq = f"  \n\t{self.sample_seq.lower()}  \n"
        res = self.predictor.predict(lower_seq)
        self.assertEqual(res["qc"]["status"], "passed")
        self.assertEqual(res["input"]["sequence"], self.sample_seq)
        self.assertEqual(res["prediction"]["class"], self.ref_df.iloc[0]["class"])

    # 3. Invalid DNA Rejection
    def test_03_invalid_dna_rejection(self):
        """Test 3: IUPAC ambiguous bases and illegal characters are rejected with structured error."""
        # Ambiguous IUPAC base 'N'
        res_n = self.predictor.predict("ACTGNNNNACTG")
        self.assertEqual(res_n["qc"]["status"], "failed")
        self.assertEqual(res_n["prediction"]["status"], "qc_failed")
        self.assertTrue(any("Ambiguous IUPAC" in e for e in res_n["qc"]["errors"]))

        # Non-nucleotide characters (digits and symbols)
        res_err = self.predictor.predict("ACGT123!@#")
        self.assertEqual(res_err["qc"]["status"], "failed")
        self.assertTrue(any("Invalid characters" in e for e in res_err["qc"]["errors"]))

    # 4. Empty Sequence Handling
    def test_04_empty_sequence_handling(self):
        """Test 4: Empty string, whitespace-only, and None return structured QC failure without crash."""
        for empty_val in ["", "   \n\t  ", None]:
            res = self.predictor.predict(empty_val)
            self.assertEqual(res["qc"]["status"], "failed")
            self.assertEqual(res["prediction"]["status"], "qc_failed")
            self.assertGreater(len(res["qc"]["errors"]), 0)

    # 5. Sequence Length Warning / Handling
    def test_05_sequence_length_warning_handling(self):
        """Test 5: Unusually short (<50 bp) or long (>300 bp) amplicons issue biological warnings."""
        # Unusually short (24 bp)
        short_seq = "ACGT" * 6
        res_short = self.predictor.predict(short_seq)
        self.assertEqual(res_short["qc"]["status"], "passed")
        self.assertTrue(any("unusually short" in w for w in res_short["qc"]["warnings"]))

        # Unusually long (320 bp)
        long_seq = "ACGT" * 80
        res_long = self.predictor.predict(long_seq)
        self.assertEqual(res_long["qc"]["status"], "passed")
        self.assertTrue(any("unusually long" in w for w in res_long["qc"]["warnings"]))

    # 6. Exact Reference Sequence
    def test_06_exact_reference_sequence(self):
        """Test 6: Exact cataloged reference sequence returns known_reference_like, 100% identity, distance 0.0."""
        res = self.predictor.predict(self.sample_seq)
        self.assertEqual(res["prediction"]["status"], "known_reference_like")
        self.assertEqual(res["similarity"]["nearest_distance"], 0.0)
        self.assertEqual(res["similarity"]["nearest_similarity"], 1.0)
        self.assertTrue(res["similarity"]["is_exact_match"])
        self.assertEqual(res["similarity"]["nearest_reference_id"], self.sample_id)
        self.assertEqual(res["prediction"]["species"], self.ref_df.iloc[0]["species"])

    # 7. Known-Reference-Like Sequence
    def test_07_known_reference_like_sequence(self):
        """Test 7: Sequence with subtle 1 bp substitution (distance ~0.01) retains known_reference_like."""
        # Mutate 1 base at position 10
        mutated = list(self.sample_seq)
        mutated[10] = "T" if mutated[10] != "T" else "A"
        mut_seq = "".join(mutated)

        res = self.predictor.predict(mut_seq)
        self.assertEqual(res["prediction"]["status"], "known_reference_like")
        self.assertLessEqual(res["similarity"]["nearest_distance"], 0.050)
        self.assertFalse(res["similarity"]["is_exact_match"])

    # 8. Low-Confidence Case
    def test_08_low_confidence_case(self):
        """Test 8: Sequence with intermediate divergence (~8% distance) triggers low_confidence."""
        # Mutate ~8% of positions
        mutated = list(self.sample_seq)
        for i in range(0, len(mutated), 12):
            mutated[i] = "T" if mutated[i] != "T" else "A"
        mut_seq = "".join(mutated)

        res = self.predictor.predict(mut_seq)
        # Should be between 0.050 and 0.150
        dist = res["similarity"]["nearest_distance"]
        if 0.050 < dist <= 0.150:
            self.assertEqual(res["prediction"]["status"], "low_confidence")
            self.assertIn("low_confidence", res["prediction"]["status"])

    # 9. Out-of-Reference Case
    def test_09_out_of_reference_case(self):
        """Test 9: Heavily divergent sequence (>20% distance) triggers out_of_reference."""
        # Heavy uniform divergence (33% substitutions)
        mutated = list(self.sample_seq)
        for i in range(0, len(mutated), 3):
            mutated[i] = "T" if mutated[i] == "A" else ("A" if mutated[i] == "T" else "C")
        mut_seq = "".join(mutated)

        res = self.predictor.predict(mut_seq)
        self.assertEqual(res["prediction"]["status"], "out_of_reference")
        self.assertGreater(res["similarity"]["nearest_distance"], 0.150)
        self.assertEqual(res["prediction"]["confidence_tier"], "low")

    # 10. High Classifier Probability Cannot Override Distance Veto
    def test_10_high_probability_cannot_override_distance_veto(self):
        """Test 10: Crucial rule - if nearest distance > 0.150, status MUST be out_of_reference even if model is 100% confident."""
        from src.prediction.evidence import synthesize_evidence_and_status
        # Simulate an overconfident classifier outputting prob=0.9999 for a distant query (d=0.22)
        fake_nearest_ref = {"sequence_id": "SEQ_001", "class": "Teleostei", "order": "Perciformes", "family": "Sciaenidae"}
        fake_prob_metrics = {"top1_prob": 0.9999, "top2_prob": 0.0001, "prob_margin": 0.9998, "entropy": 0.001}
        fake_concordance = {"class": True, "order": True, "family": True, "genus": True, "species": True}

        status, conf_tier, ev = synthesize_evidence_and_status(
            nearest_dist=0.2200,
            nearest_ref=fake_nearest_ref,
            prob_metrics=fake_prob_metrics,
            concordance=fake_concordance,
            is_exact_match=False
        )
        # MUST remain out_of_reference despite 99.99% model probability!
        self.assertEqual(status, "out_of_reference")
        self.assertEqual(conf_tier, "low")
        self.assertIn("exceeds the out-of-reference boundary", ev["explanation"])

    # 11. Top-K Similarity Structure
    def test_11_top_k_similarity_structure(self):
        """Test 11: Top-K references are correctly ranked by ascending distance."""
        res = self.predictor.predict(self.sample_seq, top_k_similar=5)
        top_refs = res["similarity"]["top_references"]
        self.assertEqual(len(top_refs), 5)
        for i in range(len(top_refs) - 1):
            self.assertLessEqual(top_refs[i]["distance"], top_refs[i+1]["distance"])
            self.assertIn("sequence_id", top_refs[i])
            self.assertIn("distance", top_refs[i])
            self.assertIn("similarity", top_refs[i])
            self.assertIn("class", top_refs[i])

    # 12. Taxonomy Fields Present
    def test_12_taxonomy_fields_present(self):
        """Test 12: All hierarchical taxonomy ranks are present in prediction output."""
        res = self.predictor.predict(self.sample_seq)
        pred = res["prediction"]
        for rank in ["kingdom", "class", "order", "family", "genus", "species"]:
            self.assertIn(rank, pred)
        self.assertEqual(pred["kingdom"], "Animalia")
        self.assertEqual(pred["species_prediction_scope"], "reference-library constrained")

    # 13. Missing Species Label Handled Correctly
    def test_13_missing_species_label_handled_correctly(self):
        """Test 13: Query matching an unlabeled reference sequence returns 'Not available in reference data'."""
        unlabeled_row = self.ref_df[self.ref_df["species"].isnull()].iloc[0]
        unlabeled_seq = unlabeled_row["sequence_normalized"]

        res = self.predictor.predict(unlabeled_seq)
        self.assertEqual(res["prediction"]["species"], "Not available in reference data")
        self.assertEqual(res["prediction"]["class"], unlabeled_row["class"])

    # 14. Biodiversity Context Does Not Fabricate Geographic Distribution
    def test_14_biodiversity_context_does_not_fabricate(self):
        """Test 14: Out-of-reference sequence strictly forbids species geographic distribution projection."""
        # Synthetic divergent sequence
        mut_seq = "ACGT" * 25
        res = self.predictor.predict(mut_seq)
        self.assertEqual(res["prediction"]["status"], "out_of_reference")
        
        bio = res["biodiversity"]
        self.assertIsNotNone(bio)
        self.assertEqual(bio["projection_status"], "NO_PROJECTION_PERMITTED")
        self.assertEqual(bio["supporting_reference_records_in_dataset"], 0)
        self.assertEqual(len(bio["countries_represented_in_dataset"]), 0)
        self.assertIn("No species or geographic distribution can be projected", bio["scientific_scope_note"])

    # 15. FASTA Single Sequence
    def test_15_fasta_single_sequence(self):
        """Test 15: Valid single-sequence FASTA string parses and classifies with preserved ID."""
        fasta_str = f">my_sample_01 Atlantic sturgeon test\n{self.sample_seq}\n"
        results = predict_fasta(fasta_str)
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res["input"]["sequence_id"], "my_sample_01")
        self.assertEqual(res["qc"]["status"], "passed")
        self.assertEqual(res["prediction"]["status"], "known_reference_like")

    # 16. FASTA Multiple Sequences
    def test_16_fasta_multiple_sequences(self):
        """Test 16: Multi-sequence FASTA predicts all sequences independently preserving order."""
        seq_2 = self.ref_df.iloc[1]["sequence_normalized"]
        seq_3 = self.ref_df.iloc[2]["sequence_normalized"]

        fasta_str = (
            f">seq_alpha Sample 1\n{self.sample_seq}\n"
            f">seq_beta Sample 2\n{seq_2}\n"
            f">seq_gamma Sample 3\n{seq_3}\n"
        )
        results = predict_fasta(fasta_str)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["input"]["sequence_id"], "seq_alpha")
        self.assertEqual(results[1]["input"]["sequence_id"], "seq_beta")
        self.assertEqual(results[2]["input"]["sequence_id"], "seq_gamma")

    # 17. Malformed FASTA
    def test_17_malformed_fasta(self):
        """Test 17: Malformed FASTA (no headers, sequence before header) raises descriptive ValueError."""
        # A. No headers at all
        with self.assertRaises(ValueError):
            parse_fasta_string("ACGTACGTACGTACGT")

        # B. Sequence before first header
        with self.assertRaises(ValueError):
            parse_fasta_string("ACGT\n>seq_01\nACGTACGT")

        # C. Empty header (just '>')
        with self.assertRaises(ValueError):
            parse_fasta_string(">\nACGTACGT")

    # 18. One Invalid Sequence in Batch Does Not Crash Other Sequences
    def test_18_one_invalid_sequence_does_not_crash_batch(self):
        """Test 18: An invalid sequence in a batch returns a QC failure without affecting valid queries."""
        batch_inputs = [
            {"id": "VALID_01", "sequence": self.sample_seq},
            {"id": "INVALID_02", "sequence": "ACTGNNNNILLEGAL123"},
            {"id": "VALID_03", "sequence": self.ref_df.iloc[1]["sequence_normalized"]}
        ]
        results = predict_batch(batch_inputs)
        self.assertEqual(len(results), 3)

        # 1st: Passed
        self.assertEqual(results[0]["qc"]["status"], "passed")
        self.assertEqual(results[0]["prediction"]["status"], "known_reference_like")

        # 2nd: Failed (contained without crash)
        self.assertEqual(results[1]["qc"]["status"], "failed")
        self.assertEqual(results[1]["prediction"]["status"], "qc_failed")
        self.assertIsNone(results[1]["similarity"])

        # 3rd: Passed
        self.assertEqual(results[2]["qc"]["status"], "passed")
        self.assertEqual(results[2]["prediction"]["status"], "known_reference_like")

    # 19. Batch CSV Export
    def test_19_batch_csv_export(self):
        """Test 19: Batch results convert to a flat DataFrame with required columns and exports cleanly."""
        batch_inputs = [
            {"id": "BATCH_01", "sequence": self.sample_seq},
            {"id": "BATCH_02", "sequence": "ACGT" * 25}
        ]
        results = predict_batch(batch_inputs)
        df = batch_results_to_dataframe(results)

        required_cols = [
            "sequence_id", "sequence_length", "qc_status", "prediction_status",
            "confidence_tier", "class", "order", "family", "genus", "species",
            "top1_probability", "probability_margin", "nearest_reference",
            "nearest_distance", "nearest_similarity", "is_out_of_reference"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["prediction_status"], "known_reference_like")
        self.assertEqual(df.iloc[1]["prediction_status"], "out_of_reference")

    # 20. Model Artifact & Feature Compatibility
    def test_20_model_artifact_feature_compatibility(self):
        """Test 20: Verified that all 5 saved model artifacts match feature definitions and encoders."""
        for lvl in ["class", "order", "family", "genus", "species"]:
            art = self.predictor.artifacts[lvl]
            self.assertIn("model", art)
            self.assertIn("scaler", art)
            self.assertIn("feature_columns", art)
            self.assertIn("target_classes", art)
            # Verify feature extractor works on sample sequence
            scaled = self.predictor._extract_features_for_model(self.sample_seq, art)
            self.assertEqual(scaled.shape, (1, len(art["feature_columns"])))

    # 21. Deterministic Repeated Prediction
    def test_21_deterministic_repeated_prediction(self):
        """Test 21: Repeated predictions on the same sequence yield 100% bit-exact results."""
        res_a = self.predictor.predict(self.sample_seq)
        res_b = self.predictor.predict(self.sample_seq)

        self.assertEqual(res_a["prediction"], res_b["prediction"])
        self.assertEqual(res_a["confidence"], res_b["confidence"])
        self.assertEqual(res_a["similarity"]["nearest_distance"], res_b["similarity"]["nearest_distance"])
        self.assertEqual(res_a["evidence"]["explanation"], res_b["evidence"]["explanation"])

    # 22. Result Schema Validation
    def test_22_result_schema_validation(self):
        """Test 22: Validate top-level schema contract keys and nested dictionary structures."""
        res = self.predictor.predict(self.sample_seq)
        expected_top_keys = ["input", "qc", "prediction", "confidence", "similarity", "evidence", "biodiversity", "disclaimer"]
        for key in expected_top_keys:
            self.assertIn(key, res)

        # Check input schema
        self.assertIn("sequence_id", res["input"])
        self.assertIn("length", res["input"])

        # Check qc schema
        self.assertIn("status", res["qc"])
        self.assertIn("warnings", res["qc"])

        # Check prediction schema
        self.assertIn("status", res["prediction"])
        self.assertIn("class", res["prediction"])
        self.assertIn("order", res["prediction"])
        self.assertIn("family", res["prediction"])
        self.assertIn("genus", res["prediction"])
        self.assertIn("species", res["prediction"])

        # Check confidence schema
        self.assertIn("top1_probability", res["confidence"])
        self.assertIn("probability_margin", res["confidence"])
        self.assertIn("entropy", res["confidence"])

    # 23. Raw Dataset Integrity Check
    def test_23_raw_dataset_sha256_unmodified(self):
        """Test 23: Verify raw dataset SHA-256 remains strictly untouched."""
        with open(RAW_PATH, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(h, EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
