"""
Unit tests for Phase 4: XGBoost Species Classification Pipeline.
Verifies model artifact loading, metadata integrity, feature dimensions,
prediction shape, probability sums, experiment results existence, and raw immutability.
"""

import os
import unittest
import hashlib
import numpy as np
import pandas as pd
import joblib

MODELS_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")
RAW_DATA_PATH = os.path.join("data", "raw", "hackathon_dna_taxonomy_merged.csv")
REF_BARCODES_PATH = os.path.join("data", "processed", "reference_barcodes.csv")
PROCESSED_DIR = os.path.join("data", "processed")
EXPECTED_RAW_SHA256 = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"

class TestPhase4XGBoost(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_path = os.path.join(MODELS_DIR, "xgboost_species.joblib")
        assert os.path.exists(cls.model_path), f"Missing {cls.model_path}"
        cls.artifact = joblib.load(cls.model_path)
        cls.results_path = os.path.join(REPORTS_DIR, "phase4_xgboost_results.csv")
        assert os.path.exists(cls.results_path), f"Missing {cls.results_path}"
        cls.results_df = pd.read_csv(cls.results_path)

        # Load reference feature matrix for model's feature set
        cls.ref_df = pd.read_csv(REF_BARCODES_PATH)
        labeled = cls.ref_df[cls.ref_df["species"].notnull()]
        feat_file = f"features_{cls.artifact['feature_set'].lower().replace('=', '').replace(' + ', '_')}.csv"
        cls.feat_df = pd.read_csv(os.path.join(PROCESSED_DIR, feat_file))
        cls.merged = pd.merge(labeled[["sequence_id", "species"]], cls.feat_df, on="sequence_id")
        cls.X = cls.merged[cls.artifact["feature_columns"]].values

    def test_01_model_loading_and_type(self):
        """Test 1: Verify model artifact loads and contains expected keys."""
        self.assertIn("model", self.artifact)
        self.assertIn("feature_set", self.artifact)
        self.assertIn("feature_columns", self.artifact)
        self.assertIn("label_encoder", self.artifact)
        self.assertIn("target_classes", self.artifact)
        self.assertIn("metrics", self.artifact)

    def test_02_model_metadata_consistency(self):
        """Test 2: Verify model metadata matches 90 species classes and random seed."""
        self.assertEqual(len(self.artifact["target_classes"]), 90)
        self.assertEqual(self.artifact["random_state"], 42)
        self.assertEqual(len(self.artifact["feature_columns"]), self.artifact["model"].n_features_in_)

    def test_03_expected_feature_dimensions(self):
        """Test 3: Verify input feature dimension matches model expectation."""
        self.assertEqual(self.X.shape[1], len(self.artifact["feature_columns"]))
        self.assertEqual(self.X.shape[0], 91)

    def test_04_prediction_shape_and_validity(self):
        """Test 4: Verify prediction output shape and class membership."""
        model = self.artifact["model"]
        preds = model.predict(self.X)
        self.assertEqual(preds.shape, (91,))
        self.assertTrue((preds >= 0).all() and (preds < 90).all())

    def test_05_probability_shape_and_sum_to_one(self):
        """Test 5: Verify probability matrix shape (91, 90) and row sum equals 1.0."""
        model = self.artifact["model"]
        probs = model.predict_proba(self.X)
        self.assertEqual(probs.shape, (91, 90))
        # Sum of probabilities across classes must be ~1.0
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_06_valid_feature_values_non_negative(self):
        """Test 6: Verify input feature matrix values are finite and non-negative."""
        self.assertTrue(np.isfinite(self.X).all())
        self.assertTrue((self.X >= 0.0).all())

    def test_07_experiment_results_exist_and_complete(self):
        """Test 7: Verify phase4_xgboost_results.csv contains all 72 configurations."""
        self.assertEqual(len(self.results_df), 72)
        required_cols = [
            "model", "feature_set", "n_features", "n_estimators", "max_depth",
            "learning_rate", "ref_top1_accuracy", "ref_top3_accuracy",
            "evaluable_holdout_accuracy", "loocv_total_accuracy"
        ]
        for col in required_cols:
            self.assertIn(col, self.results_df.columns)

    def test_08_feature_importance_file_exists(self):
        """Test 8: Verify feature importance CSV exists and contains ranked values."""
        fi_path = os.path.join(REPORTS_DIR, "xgboost_feature_importance.csv")
        self.assertTrue(os.path.exists(fi_path))
        fi_df = pd.read_csv(fi_path)
        self.assertGreater(len(fi_df), 0)
        self.assertIn("feature", fi_df.columns)
        self.assertIn("importance_gain", fi_df.columns)
        self.assertTrue((fi_df["importance_gain"] >= 0).all())

    def test_09_raw_dataset_remains_unchanged(self):
        """Test 9: Verify raw CSV SHA256 checksum has not changed."""
        sha256 = hashlib.sha256()
        with open(RAW_DATA_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest().upper()
        self.assertEqual(actual_hash, EXPECTED_RAW_SHA256, "Raw data hash modified during Phase 4!")

if __name__ == "__main__":
    unittest.main()
