"""
tests/test_phase5_taxonomy.py
Unit and integration tests for Phase 5: Hierarchical Taxonomy Prediction.
"""

import os
import unittest
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

REF_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
TAX_PATH = os.path.join(PROCESSED_DIR, "taxonomy_reference.csv")
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "hackathon_dna_taxonomy_merged.csv")

TAXONOMY_LEVELS = ["class", "order", "family", "genus", "species"]

class TestPhase5HierarchicalTaxonomy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref_df = pd.read_csv(REF_PATH)
        cls.tax_df = pd.read_csv(TAX_PATH)
        cls.results_df = pd.read_csv(os.path.join(REPORTS_DIR, "phase5_taxonomy_results.csv"))
        cls.hier_df = pd.read_csv(os.path.join(REPORTS_DIR, "phase5_hierarchical_results.csv"))

    def test_01_taxonomy_levels_and_sequence_count(self):
        """Test 1: Verify exactly 101 unique sequences and expected non-null counts."""
        self.assertEqual(len(self.ref_df), 101)
        self.assertEqual(self.ref_df["sequence_id"].nunique(), 101)
        
        # Non-null counts per level
        self.assertEqual(self.ref_df["class"].notnull().sum(), 101)
        self.assertEqual(self.ref_df["order"].notnull().sum(), 101)
        self.assertEqual(self.ref_df["family"].notnull().sum(), 101)
        self.assertEqual(self.ref_df["genus"].notnull().sum(), 97)
        self.assertEqual(self.ref_df["species"].notnull().sum(), 91)

    def test_02_preservation_of_unlabeled_sequences(self):
        """Test 2: Verify exactly 10 sequences lack species labels and are preserved."""
        unlabeled_sp = self.ref_df[self.ref_df["species"].isnull()]
        self.assertEqual(len(unlabeled_sp), 10)
        
        # Verify they have valid higher-level taxonomy
        self.assertEqual(unlabeled_sp["class"].notnull().sum(), 10)
        self.assertEqual(unlabeled_sp["order"].notnull().sum(), 10)
        self.assertEqual(unlabeled_sp["family"].notnull().sum(), 10)
        self.assertEqual(unlabeled_sp["genus"].notnull().sum(), 6)

    def test_03_internal_taxonomy_hierarchy_consistency(self):
        """Test 3: Verify clean child-to-parent mapping across the reference taxonomy."""
        pairs = [
            ("order", "class"),
            ("family", "order"),
            ("genus", "family"),
            ("species", "genus")
        ]
        for child, parent in pairs:
            df_valid = self.ref_df[[child, parent]].dropna().drop_duplicates()
            counts = df_valid[child].value_counts()
            multi_parents = counts[counts > 1]
            self.assertEqual(len(multi_parents), 0, f"{child} mapped to multiple {parent}s")

    def test_04_model_artifacts_loading_and_metadata(self):
        """Test 4: Verify all 5 taxonomic level model artifacts load with required keys."""
        for lvl in TAXONOMY_LEVELS:
            model_path = os.path.join(MODELS_DIR, f"taxonomy_{lvl}_model.joblib")
            self.assertTrue(os.path.exists(model_path), f"Missing artifact: {model_path}")
            
            art = joblib.load(model_path)
            self.assertEqual(art["taxonomy_level"], lvl)
            self.assertIn("model", art)
            self.assertIn("scaler", art)
            self.assertIn("label_encoder", art)
            self.assertIn("target_classes", art)
            self.assertIn("feature_set", art)
            self.assertIn("feature_columns", art)
            self.assertIn("metrics", art)

    def test_05_no_taxonomy_or_metadata_used_as_features(self):
        """Test 5: Ensure DNA features ONLY are used (no taxonomy names or spatial IDs)."""
        forbidden_substrings = [
            "class", "order", "family", "genus", "species", "taxon",
            "lat", "lon", "country", "date", "occurrence", "gbif", "scientific"
        ]
        for lvl in TAXONOMY_LEVELS:
            model_path = os.path.join(MODELS_DIR, f"taxonomy_{lvl}_model.joblib")
            art = joblib.load(model_path)
            for col in art["feature_columns"]:
                col_lower = col.lower()
                for forb in forbidden_substrings:
                    self.assertNotIn(forb, col_lower, f"Forbidden column found: {col}")

    def test_06_prediction_shapes_and_probabilities(self):
        """Test 6: Verify prediction and probability output shapes on reference features."""
        for lvl in TAXONOMY_LEVELS:
            art = joblib.load(model_path := os.path.join(MODELS_DIR, f"taxonomy_{lvl}_model.joblib"))
            feat_path = os.path.join(PROCESSED_DIR, f"features_{art['feature_set'].lower().replace(' ', '_').replace('+', '_')}.csv")
            # Feature loading
            if art["feature_set"] == "Basic":
                feat_path = os.path.join(PROCESSED_DIR, "features_basic.csv")
            elif art["feature_set"] == "k=2":
                feat_path = os.path.join(PROCESSED_DIR, "features_k2.csv")
            elif art["feature_set"] == "k=3":
                feat_path = os.path.join(PROCESSED_DIR, "features_k3.csv")
            elif art["feature_set"] == "Basic + k=2":
                feat_path = os.path.join(PROCESSED_DIR, "features_basic_k2.csv")
                
            feat_df = pd.read_csv(feat_path)
            X = feat_df[art["feature_columns"]].values
            X_scaled = art["scaler"].transform(X)
            
            probs = art["model"].predict_proba(X_scaled)
            self.assertEqual(probs.shape, (101, len(art["target_classes"])))
            # Probabilities sum to ~1
            sums = np.sum(probs, axis=1)
            np.testing.assert_allclose(sums, np.ones(101), rtol=1e-4)

    def test_07_hierarchical_consistency_guarantee(self):
        """Test 7: Verify hierarchically-constrained pipeline achieves 100% taxonomic consistency."""
        tax_df = pd.read_csv(TAX_PATH)
        child_to_parent = {
            "order": dict(tax_df[["order", "class"]].dropna().drop_duplicates().values),
            "family": dict(tax_df[["family", "order"]].dropna().drop_duplicates().values),
            "genus": dict(tax_df[["genus", "family"]].dropna().drop_duplicates().values),
            "species": dict(tax_df[["species", "genus"]].dropna().drop_duplicates().values),
        }
        
        for idx, row in self.hier_df.iterrows():
            c = row["hier_class"]
            o = row["hier_order"]
            f = row["hier_family"]
            g = row["hier_genus"]
            s = row["hier_species"]
            
            # Check Order in Class
            self.assertEqual(child_to_parent["order"].get(o), c)
            # Check Family in Order
            self.assertEqual(child_to_parent["family"].get(f), o)
            # Check Genus in Family (unless Not available)
            if g != "Not available in reference data":
                self.assertEqual(child_to_parent["genus"].get(g), f)
            # Check Species in Genus (unless Not available)
            if s != "Not available in reference data":
                self.assertEqual(child_to_parent["species"].get(s), g)

    def test_08_partial_taxonomy_unlabeled_species_behavior(self):
        """Test 8: Verify all 10 unlabeled species sequences output 'Not available in reference data'."""
        unlabeled_rows = self.hier_df[~self.hier_df["has_species_label"]]
        self.assertEqual(len(unlabeled_rows), 10)
        
        for idx, row in unlabeled_rows.iterrows():
            self.assertEqual(row["hier_species"], "Not available in reference data")
            self.assertIsNotNone(row["hier_class"])
            self.assertIsNotNone(row["hier_order"])
            self.assertIsNotNone(row["hier_family"])
            self.assertIsNotNone(row["hier_genus"])

    def test_09_experiment_results_and_figures_exist(self):
        """Test 9: Verify Phase 5 results CSVs and figures exist and are non-empty."""
        res_path = os.path.join(REPORTS_DIR, "phase5_taxonomy_results.csv")
        hier_path = os.path.join(REPORTS_DIR, "phase5_hierarchical_results.csv")
        self.assertTrue(os.path.exists(res_path))
        self.assertTrue(os.path.exists(hier_path))
        self.assertEqual(len(pd.read_csv(res_path)), 60) # 5 levels * 4 feature sets * 3 models
        self.assertEqual(len(pd.read_csv(hier_path)), 101)

        # Figures
        fig1 = os.path.join(REPORTS_DIR, "figures", "hierarchical_accuracy_by_level.png")
        fig2 = os.path.join(REPORTS_DIR, "figures", "hierarchical_error_depth.png")
        fig3 = os.path.join(REPORTS_DIR, "figures", "class_confusion_matrix.png")
        self.assertTrue(os.path.exists(fig1))
        self.assertTrue(os.path.exists(fig2))
        self.assertTrue(os.path.exists(fig3))

    def test_10_raw_dataset_integrity(self):
        """Test 10: Verify raw dataset remains strictly untouched with original SHA-256."""
        import hashlib
        with open(RAW_PATH, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(h, "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65")

if __name__ == "__main__":
    unittest.main()
