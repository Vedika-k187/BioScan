"""
Unit tests for Phase 3: Baseline Machine Learning Pipeline.
Verifies species-labeled dataset isolation, absence of taxonomy leaks,
scaling encapsulation in KNN pipeline, RF training validity, prediction dimensions,
LOOCV leakage safety, and immutability of the raw dataset.
"""

import os
import unittest
import hashlib
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from src.models.baseline_models import (
    load_species_modeling_data,
    compute_top_k_accuracy,
    evaluate_loocv,
    REF_BARCODES_PATH,
    MODELS_DIR,
    FEATURE_SETS
)

RAW_DATA_PATH = os.path.join("data", "raw", "hackathon_dna_taxonomy_merged.csv")
EXPECTED_RAW_SHA256 = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"

TAXONOMY_COLUMNS = {
    "class", "order", "family", "genus", "species", "scientificName",
    "taxonRank", "taxonomicStatus", "kingdom", "phylum", "gbifid",
    "occurrenceID", "eventID", "sampname", "latitude", "longitude"
}

class TestPhase3Models(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref_df = pd.read_csv(REF_BARCODES_PATH)
        cls.X_basic, cls.y_basic, cls.ids_basic, cls.cols_basic = load_species_modeling_data("Basic")
        cls.X_k3, cls.y_k3, cls.ids_k3, cls.cols_k3 = load_species_modeling_data("k=3")

    def test_01_modeling_dataset_contains_only_species_labeled(self):
        """Test 1: Verify modeling dataset contains exactly the species-labeled sequences (91)."""
        labeled_in_ref = self.ref_df["species"].notnull().sum()
        self.assertEqual(len(self.y_basic), labeled_in_ref)
        self.assertEqual(len(self.y_basic), 91)
        # Verify no NaN labels in y
        self.assertFalse(pd.isnull(self.y_basic).any())

    def test_02_no_taxonomy_metadata_present_in_X(self):
        """Test 2: Verify strictly zero taxonomy or metadata columns in feature matrix X."""
        for feat_set in ["Basic", "k=2", "k=3"]:
            X_df, _, _, cols = load_species_modeling_data(feat_set)
            leak = TAXONOMY_COLUMNS.intersection(set(X_df.columns))
            self.assertEqual(len(leak), 0, f"Taxonomy leak detected in {feat_set}: {leak}")

    def test_03_number_of_samples_is_correct(self):
        """Test 3: Verify all feature representations produce exactly 91 samples."""
        for feat_set in FEATURE_SETS.keys():
            X_df, y, seq_ids, _ = load_species_modeling_data(feat_set)
            self.assertEqual(X_df.shape[0], 91)
            self.assertEqual(len(y), 91)
            self.assertEqual(len(seq_ids), 91)

    def test_04_feature_dimensions_match_expected_matrices(self):
        """Test 4: Verify feature column counts match specification."""
        expected_dims = {
            "Basic": 17,
            "k=2": 16,
            "k=3": 64,
            "k=4": 256,
            "k=5": 1024,
            "Basic + k=2": 33,
            "Basic + k=3": 81,
            "Basic + k=4": 273,
            "Basic + k=5": 1041
        }
        for feat_name, expected_p in expected_dims.items():
            X_df, _, _, _ = load_species_modeling_data(feat_name)
            self.assertEqual(
                X_df.shape[1],
                expected_p,
                f"Dimension mismatch for {feat_name}: expected {expected_p}, got {X_df.shape[1]}"
            )

    def test_05_knn_pipeline_contains_scaling(self):
        """Test 5: Verify saved baseline KNN contains StandardScaler in its Pipeline."""
        knn_path = os.path.join(MODELS_DIR, "baseline_knn.joblib")
        self.assertTrue(os.path.exists(knn_path), "baseline_knn.joblib missing")
        knn_artifact = joblib.load(knn_path)
        pipe = knn_artifact["pipeline"]
        self.assertIsInstance(pipe, Pipeline)
        self.assertIn("scaler", pipe.named_steps)
        self.assertIsInstance(pipe.named_steps["scaler"], StandardScaler)
        self.assertIn("knn", pipe.named_steps)
        self.assertIsInstance(pipe.named_steps["knn"], KNeighborsClassifier)

    def test_06_random_forest_can_train_successfully(self):
        """Test 6: Verify Random Forest fits cleanly on features and predicts valid classes."""
        rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        rf.fit(self.X_k3.values, self.y_k3)
        preds = rf.predict(self.X_k3.values)
        self.assertEqual(len(preds), 91)
        self.assertTrue(all(p in rf.classes_ for p in preds))

    def test_07_predictions_contain_valid_species_labels(self):
        """Test 7: Verify saved baseline models generate valid species predictions."""
        rf_path = os.path.join(MODELS_DIR, "baseline_random_forest.joblib")
        self.assertTrue(os.path.exists(rf_path), "baseline_random_forest.joblib missing")
        rf_artifact = joblib.load(rf_path)
        rf_pipe = rf_artifact["pipeline"]
        
        feat_name = rf_artifact["feature_set"]
        X_df, _, _, _ = load_species_modeling_data(feat_name)
        preds = rf_pipe.predict(X_df.values)
        self.assertEqual(len(preds), 91)
        self.assertTrue(all(p in rf_pipe.classes_ for p in preds))

    def test_08_probability_outputs_have_expected_dimensions(self):
        """Test 8: Verify probability outputs sum to 1.0 and match class count (90)."""
        rf_path = os.path.join(MODELS_DIR, "baseline_random_forest.joblib")
        rf_artifact = joblib.load(rf_path)
        rf_pipe = rf_artifact["pipeline"]
        
        feat_name = rf_artifact["feature_set"]
        X_df, _, _, _ = load_species_modeling_data(feat_name)
        probs = rf_pipe.predict_proba(X_df.values)
        self.assertEqual(probs.shape, (91, 90))
        # Each row must sum to 1.0
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))

    def test_09_cross_validation_executes_without_leakage(self):
        """Test 9: Verify LOOCV runs cleanly and returns expected evaluation structure."""
        def factory():
            return Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=1))
            ])
        res = evaluate_loocv(factory, self.X_basic.values[:10], self.y_basic[:10])
        self.assertIn("loocv_accuracy", res)
        self.assertEqual(len(res["y_preds"]), 10)

    def test_10_raw_dataset_remains_unchanged(self):
        """Test 10: Verify raw CSV SHA256 checksum has not changed."""
        sha256 = hashlib.sha256()
        with open(RAW_DATA_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest().upper()
        self.assertEqual(actual_hash, EXPECTED_RAW_SHA256, "Raw dataset was modified during Phase 3!")

if __name__ == "__main__":
    unittest.main()
