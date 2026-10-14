"""
Unit tests for Phase 2: DNA Feature Engineering Pipeline.
Verifies mathematical correctness of basic features, k-mer dimensions,
sum-to-one constraints, sequence alignment, label isolation, and raw immutability.
"""

import os
import re
import unittest
import hashlib
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw", "hackathon_dna_taxonomy_merged.csv")
REF_BARCODES_PATH = os.path.join("data", "processed", "reference_barcodes.csv")
PROCESSED_DIR = os.path.join("data", "processed")
EXPECTED_RAW_SHA256 = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"

TAXONOMY_COLS = {
    "class", "order", "family", "genus", "species", "scientificName",
    "taxonRank", "taxonomicStatus", "kingdom", "phylum"
}

class TestPhase2Features(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ref_df = pd.read_csv(REF_BARCODES_PATH)
        cls.basic_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features_basic.csv"))
        cls.k2_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features_k2.csv"))
        cls.k3_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features_k3.csv"))
        cls.k4_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features_k4.csv"))
        cls.k5_df = pd.read_csv(os.path.join(PROCESSED_DIR, "features_k5.csv"))

    def test_01_basic_feature_values_mathematically_correct(self):
        """Test 1: Verify basic composition counts and lengths match sequence strings exactly."""
        for idx, row in self.ref_df.iterrows():
            seq_id = row["sequence_id"]
            seq = row["sequence_normalized"]
            basic_row = self.basic_df[self.basic_df["sequence_id"] == seq_id].iloc[0]

            self.assertEqual(basic_row["sequence_length"], len(seq))
            self.assertEqual(basic_row["A_count"], seq.count('A'))
            self.assertEqual(basic_row["T_count"], seq.count('T'))
            self.assertEqual(basic_row["G_count"], seq.count('G'))
            self.assertEqual(basic_row["C_count"], seq.count('C'))

            expected_gc = (seq.count('G') + seq.count('C')) / len(seq)
            expected_at = (seq.count('A') + seq.count('T')) / len(seq)
            self.assertAlmostEqual(basic_row["GC_content"], expected_gc, places=6)
            self.assertAlmostEqual(basic_row["AT_content"], expected_at, places=6)

    def test_02_base_frequencies_sum_to_one(self):
        """Test 2: Verify A_freq + T_freq + G_freq + C_freq == 1.0 for all sequences."""
        freq_sum = (
            self.basic_df["A_frequency"]
            + self.basic_df["T_frequency"]
            + self.basic_df["G_frequency"]
            + self.basic_df["C_frequency"]
        )
        self.assertTrue(np.allclose(freq_sum, 1.0, atol=1e-6))

    def test_03_gc_plus_at_equals_one(self):
        """Test 3: Verify GC_content + AT_content == 1.0 for all sequences."""
        content_sum = self.basic_df["GC_content"] + self.basic_df["AT_content"]
        self.assertTrue(np.allclose(content_sum, 1.0, atol=1e-6))

    def test_04_k2_produces_16_features(self):
        """Test 4: Verify k=2 has exactly 16 k-mer features plus sequence_id."""
        kmer_cols = [c for c in self.k2_df.columns if c != "sequence_id"]
        self.assertEqual(len(kmer_cols), 16)
        pattern = re.compile(r'^[ACGT]{2}$')
        for c in kmer_cols:
            self.assertTrue(bool(pattern.match(c)), f"Invalid k=2 column name: {c}")

    def test_05_k3_produces_64_features(self):
        """Test 5: Verify k=3 has exactly 64 k-mer features plus sequence_id."""
        kmer_cols = [c for c in self.k3_df.columns if c != "sequence_id"]
        self.assertEqual(len(kmer_cols), 64)
        pattern = re.compile(r'^[ACGT]{3}$')
        for c in kmer_cols:
            self.assertTrue(bool(pattern.match(c)), f"Invalid k=3 column name: {c}")

    def test_06_k4_produces_256_features(self):
        """Test 6: Verify k=4 has exactly 256 k-mer features plus sequence_id."""
        kmer_cols = [c for c in self.k4_df.columns if c != "sequence_id"]
        self.assertEqual(len(kmer_cols), 256)
        pattern = re.compile(r'^[ACGT]{4}$')
        for c in kmer_cols:
            self.assertTrue(bool(pattern.match(c)), f"Invalid k=4 column name: {c}")

    def test_07_k5_produces_1024_features(self):
        """Test 7: Verify k=5 has exactly 1024 k-mer features plus sequence_id."""
        kmer_cols = [c for c in self.k5_df.columns if c != "sequence_id"]
        self.assertEqual(len(kmer_cols), 1024)
        pattern = re.compile(r'^[ACGT]{5}$')
        for c in kmer_cols:
            self.assertTrue(bool(pattern.match(c)), f"Invalid k=5 column name: {c}")

    def test_08_kmer_frequencies_are_valid_and_sum_to_one(self):
        """Test 8: Verify k-mer frequencies are non-negative and sum to ~1.0 for each sequence."""
        for k_df, k in [(self.k2_df, 2), (self.k3_df, 3), (self.k4_df, 4), (self.k5_df, 5)]:
            kmer_cols = [c for c in k_df.columns if c != "sequence_id"]
            # Non-negative
            self.assertTrue((k_df[kmer_cols] >= 0.0).all().all())
            # Sum to 1.0
            row_sums = k_df[kmer_cols].sum(axis=1)
            self.assertTrue(
                np.allclose(row_sums, 1.0, atol=1e-5),
                f"k={k} frequencies do not sum to 1.0"
            )

    def test_09_every_feature_matrix_contains_exactly_101_sequences(self):
        """Test 9: Verify all feature matrices have exactly 101 rows."""
        self.assertEqual(len(self.basic_df), 101)
        self.assertEqual(len(self.k2_df), 101)
        self.assertEqual(len(self.k3_df), 101)
        self.assertEqual(len(self.k4_df), 101)
        self.assertEqual(len(self.k5_df), 101)

    def test_10_sequence_ids_are_unique_and_aligned(self):
        """Test 10: Verify sequence_ids are unique and identical across all feature tables."""
        ref_ids = self.ref_df["sequence_id"].tolist()
        for df_name, df in [
            ("basic", self.basic_df),
            ("k2", self.k2_df),
            ("k3", self.k3_df),
            ("k4", self.k4_df),
            ("k5", self.k5_df)
        ]:
            self.assertEqual(df["sequence_id"].nunique(), 101)
            self.assertEqual(df["sequence_id"].tolist(), ref_ids, f"{df_name} sequence_id alignment mismatch")

    def test_11_no_taxonomy_labels_in_feature_matrices(self):
        """Test 11: Verify no taxonomy or metadata labels leaked into numerical feature files."""
        for df_name, df in [
            ("basic", self.basic_df),
            ("k2", self.k2_df),
            ("k3", self.k3_df),
            ("k4", self.k4_df),
            ("k5", self.k5_df)
        ]:
            overlap = TAXONOMY_COLS.intersection(set(df.columns))
            self.assertEqual(
                len(overlap),
                0,
                f"Leaked taxonomy labels found in {df_name}: {overlap}"
            )

    def test_12_raw_dataset_remains_unchanged(self):
        """Test 12: Verify raw dataset SHA256 matches expected checksum exactly."""
        sha256 = hashlib.sha256()
        with open(RAW_DATA_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest().upper()
        self.assertEqual(actual_hash, EXPECTED_RAW_SHA256, "Raw data hash modified during Phase 2!")

if __name__ == "__main__":
    unittest.main()
