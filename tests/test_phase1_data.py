"""
Unit and integration tests for Phase 1 Data Cleaning & Dataset Construction.
Verifies data integrity, absence of duplicates, taxonomic consistency,
preservation of raw data, and compliance with project invariants.
"""

import os
import re
import unittest
import hashlib
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw", "hackathon_dna_taxonomy_merged.csv")
REF_BARCODES_PATH = os.path.join("data", "processed", "reference_barcodes.csv")
TAXONOMY_REF_PATH = os.path.join("data", "processed", "taxonomy_reference.csv")
EXPECTED_RAW_SHA256 = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"

class TestPhase1DataQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Verify files exist
        assert os.path.exists(RAW_DATA_PATH), f"Raw data file missing at {RAW_DATA_PATH}"
        assert os.path.exists(REF_BARCODES_PATH), f"Reference barcodes missing at {REF_BARCODES_PATH}"
        assert os.path.exists(TAXONOMY_REF_PATH), f"Taxonomy reference missing at {TAXONOMY_REF_PATH}"
        
        cls.raw_df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
        cls.ref_df = pd.read_csv(REF_BARCODES_PATH)
        cls.tax_df = pd.read_csv(TAXONOMY_REF_PATH)

    def test_01_raw_dataset_integrity_unmodified(self):
        """Test 1: Verify the raw dataset has not been modified (SHA256 match)."""
        sha256 = hashlib.sha256()
        with open(RAW_DATA_PATH, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest().upper()
        self.assertEqual(
            actual_hash,
            EXPECTED_RAW_SHA256,
            "Raw dataset hash mismatch! The raw dataset must never be modified."
        )
        self.assertEqual(len(self.raw_df), 13736)
        self.assertEqual(self.raw_df.shape[1], 59)

    def test_02_reference_barcodes_no_duplicate_sequences(self):
        """Test 2: Verify reference_barcodes has no duplicate normalized DNA sequences."""
        seqs = self.ref_df["sequence_normalized"]
        self.assertEqual(
            len(seqs),
            seqs.nunique(),
            f"Found {len(seqs) - seqs.nunique()} duplicate sequences in reference_barcodes.csv!"
        )

    def test_03_every_sequence_strictly_valid_acgt(self):
        """Test 3: Verify all normalized sequences contain strictly uppercase A, C, G, T."""
        pattern = re.compile(r'^[ACGT]+$')
        for idx, row in self.ref_df.iterrows():
            seq = row["sequence_normalized"]
            self.assertTrue(
                bool(pattern.fullmatch(seq)),
                f"Row {row['sequence_id']} has invalid characters: {seq}"
            )
            # Check length matches reported length
            self.assertEqual(
                len(seq),
                row["sequence_length"],
                f"Length mismatch for {row['sequence_id']}"
            )
            # Check composition counts match sequence
            self.assertEqual(seq.count('A'), row["A_count"])
            self.assertEqual(seq.count('T'), row["T_count"])
            self.assertEqual(seq.count('G'), row["G_count"])
            self.assertEqual(seq.count('C'), row["C_count"])

    def test_04_taxonomy_consistency_within_each_sequence(self):
        """Test 4: Verify that in the raw dataset, taxonomy was 100% consistent per sequence."""
        tax_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
        grouped = self.raw_df.groupby("dnasequence")
        for seq, group in grouped:
            for lvl in tax_levels:
                if lvl in group.columns:
                    unique_vals = group[lvl].drop_duplicates().tolist()
                    self.assertLessEqual(
                        len(unique_vals),
                        1,
                        f"Inconsistent {lvl} for sequence {seq[:20]}...: {unique_vals}"
                    )

    def test_05_processed_dataset_contains_101_unique_sequences(self):
        """Test 5: Verify processed reference dataset contains exactly 101 unique sequences."""
        self.assertEqual(
            len(self.ref_df),
            101,
            f"Expected exactly 101 reference sequences, got {len(self.ref_df)}"
        )
        self.assertEqual(
            len(self.tax_df),
            101,
            f"Expected exactly 101 taxonomy rows, got {len(self.tax_df)}"
        )

    def test_06_species_missing_sequences_preserved(self):
        """Test 6: Verify sequences lacking species labels are preserved and not dropped."""
        missing_species = self.ref_df[self.ref_df["species"].isnull()]
        self.assertEqual(
            len(missing_species),
            10,
            f"Expected exactly 10 species-missing sequences, found {len(missing_species)}"
        )
        # All 10 must still have complete class, order, and family
        self.assertEqual(missing_species["class"].isnull().sum(), 0)
        self.assertEqual(missing_species["order"].isnull().sum(), 0)
        self.assertEqual(missing_species["family"].isnull().sum(), 0)
        # 6 must have genus, 4 missing genus
        self.assertEqual(missing_species["genus"].notnull().sum(), 6)
        self.assertEqual(missing_species["genus"].isnull().sum(), 4)

    def test_07_required_columns_exist(self):
        """Test 7: Verify required columns exist in both processed tables."""
        req_ref_cols = [
            "sequence_id", "sequence_raw", "sequence_normalized",
            "sequence_length", "A_count", "T_count", "G_count", "C_count",
            "GC_content", "class", "order", "family", "genus", "species"
        ]
        for col in req_ref_cols:
            self.assertIn(col, self.ref_df.columns, f"Missing {col} in reference_barcodes.csv")

        req_tax_cols = [
            "sequence_id", "sequence_normalized", "class", "order", "family", "genus", "species"
        ]
        for col in req_tax_cols:
            self.assertIn(col, self.tax_df.columns, f"Missing {col} in taxonomy_reference.csv")

    def test_08_sequence_ids_unique_and_aligned(self):
        """Test 8: Verify sequence_id values are unique and aligned across both tables."""
        self.assertEqual(self.ref_df["sequence_id"].nunique(), 101)
        self.assertEqual(self.tax_df["sequence_id"].nunique(), 101)
        self.assertTrue(
            (self.ref_df["sequence_id"] == self.tax_df["sequence_id"]).all(),
            "sequence_id alignment mismatch between reference_barcodes and taxonomy_reference"
        )
        self.assertTrue(
            (self.ref_df["sequence_normalized"] == self.tax_df["sequence_normalized"]).all(),
            "sequence_normalized alignment mismatch between reference_barcodes and taxonomy_reference"
        )

if __name__ == "__main__":
    unittest.main()
