"""
tests/test_phase7_biodiversity.py
Unit and integration tests for Phase 7: Biodiversity Intelligence,
Geographic Analysis & Analytics Layer.
"""

import os
import math
import hashlib
import unittest
import numpy as np
import pandas as pd

from src.biodiversity.data_quality import (
    validate_coordinates,
    categorize_geographic_records,
    audit_geographic_data_quality
)
from src.biodiversity.taxonomy_analysis import (
    compute_taxonomic_richness,
    analyze_taxonomic_hierarchy,
    build_taxonomic_summary
)
from src.biodiversity.geographic_analysis import (
    build_biodiversity_reference_records,
    summarize_country_biodiversity,
    summarize_species_distribution,
    analyze_sampling_coverage
)
from src.biodiversity.diversity_metrics import (
    calculate_shannon_diversity,
    calculate_simpson_diversity,
    calculate_pielou_evenness,
    compute_station_diversity_metrics
)
from src.biodiversity.prediction_biodiversity import (
    analyze_prediction,
    BioScanBiodiversityIntelligence
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "hackathon_dna_taxonomy_merged.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REF_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EXPECTED_SHA256 = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"


class TestPhase7Biodiversity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ref_df = pd.read_csv(REF_PATH)
        cls.records_path = os.path.join(PROCESSED_DIR, "biodiversity_reference_records.csv")
        cls.records_df = pd.read_csv(cls.records_path) if os.path.exists(cls.records_path) else None

    # -------------------------------------------------------------
    # 1. Deterministic Diversity Mathematical Tests (Toy Datasets)
    # -------------------------------------------------------------
    def test_01_shannon_diversity_known_toy_examples(self):
        """Test 1: Verify Shannon diversity on exact mathematical theoretical cases."""
        # Case A: Two equally abundant taxa [50, 50] -> H = ln(2) approx 0.693147
        h_two = calculate_shannon_diversity([50, 50])
        self.assertAlmostEqual(h_two, math.log(2), places=5)

        # Case B: Four equally abundant taxa [25, 25, 25, 25] -> H = ln(4) approx 1.386294
        h_four = calculate_shannon_diversity([25, 25, 25, 25])
        self.assertAlmostEqual(h_four, math.log(4), places=5)

        # Case C: Ten equally abundant taxa -> H = ln(10) approx 2.302585
        h_ten = calculate_shannon_diversity([10] * 10)
        self.assertAlmostEqual(h_ten, math.log(10), places=5)

        # Case D: Unequal community [90, 10] -> p = [0.9, 0.1]
        # H = -(0.9 * ln(0.9) + 0.1 * ln(0.1)) approx 0.325083
        expected_unequal = -(0.9 * math.log(0.9) + 0.1 * math.log(0.1))
        h_unequal = calculate_shannon_diversity([90, 10])
        self.assertAlmostEqual(h_unequal, expected_unequal, places=5)

    def test_02_diversity_zero_and_empty_edge_cases(self):
        """Test 2: Verify zero/empty/single taxon inputs return exactly 0.0."""
        # Empty list
        self.assertEqual(calculate_shannon_diversity([]), 0.0)
        self.assertEqual(calculate_simpson_diversity([]), 0.0)
        self.assertEqual(calculate_pielou_evenness(0.0, 0), 0.0)

        # All zeros
        self.assertEqual(calculate_shannon_diversity([0, 0, 0]), 0.0)
        self.assertEqual(calculate_simpson_diversity([0, 0, 0]), 0.0)

        # Single taxon (richness S=1, no uncertainty)
        self.assertEqual(calculate_shannon_diversity([100]), 0.0)
        self.assertEqual(calculate_simpson_diversity([100]), 0.0)
        self.assertEqual(calculate_pielou_evenness(0.0, 1), 0.0)

        # None input
        self.assertEqual(calculate_shannon_diversity(None), 0.0)

    def test_03_simpson_and_pielou_evenness_properties(self):
        """Test 3: Verify Simpson diversity and Pielou evenness boundaries [0.0, 1.0]."""
        # Equal community of 4 species: Simpson = 1 - 4 * (0.25^2) = 1 - 0.25 = 0.75
        simpson = calculate_simpson_diversity([10, 10, 10, 10])
        self.assertAlmostEqual(simpson, 0.75, places=5)

        # Pielou evenness of completely even community must be exactly 1.0
        h_even = calculate_shannon_diversity([10, 10, 10, 10])
        j_even = calculate_pielou_evenness(h_even, 4)
        self.assertAlmostEqual(j_even, 1.0, places=5)

        # Highly skewed community: [999, 1] -> evenness should be very low
        h_skewed = calculate_shannon_diversity([999, 1])
        j_skewed = calculate_pielou_evenness(h_skewed, 2)
        self.assertLess(j_skewed, 0.1)

    # -------------------------------------------------------------
    # 2. Geographic & Coordinate Validation Tests
    # -------------------------------------------------------------
    def test_04_coordinate_validation_physical_boundaries(self):
        """Test 4: Verify physical boundary checks [-90, 90] and [-180, 180]."""
        # Valid mid-Atlantic coordinates
        ok, cat, _ = validate_coordinates(40.224, -73.977)
        self.assertTrue(ok)
        self.assertEqual(cat, "valid_coordinates")

        # Latitude out of bounds (+95.0)
        ok, cat, msg = validate_coordinates(95.0, -73.0)
        self.assertFalse(ok)
        self.assertEqual(cat, "invalid_coordinates")
        self.assertIn("Latitude 95.0 outside", msg)

        # Longitude out of bounds (-195.0)
        ok, cat, msg = validate_coordinates(40.0, -195.0)
        self.assertFalse(ok)
        self.assertEqual(cat, "invalid_coordinates")
        self.assertIn("Longitude -195.0 outside", msg)

        # Missing values (None, NaN)
        ok, cat, _ = validate_coordinates(None, -73.0)
        self.assertFalse(ok)
        self.assertEqual(cat, "missing_coordinates")

        ok, cat, _ = validate_coordinates(40.0, np.nan)
        self.assertFalse(ok)
        self.assertEqual(cat, "missing_coordinates")

        # Null Island artifact (0.0, 0.0)
        ok, cat, msg = validate_coordinates(0.0, 0.0)
        self.assertFalse(ok)
        self.assertEqual(cat, "invalid_coordinates")
        self.assertIn("Null Island", msg)

    # -------------------------------------------------------------
    # 3. Taxonomic Richness & Hierarchy Consistency Tests
    # -------------------------------------------------------------
    def test_05_taxonomic_richness_counts(self):
        """Test 5: Verify exact taxonomic counts on approved reference library."""
        richness = compute_taxonomic_richness(self.ref_df)
        self.assertEqual(richness["unique_sequences"], 101)
        self.assertEqual(richness["species_richness"], 90)
        self.assertEqual(richness["species_labeled_sequences"], 91)
        self.assertEqual(richness["species_unlabeled_sequences"], 10)
        self.assertEqual(richness["genus_richness"], 87)
        self.assertEqual(richness["family_richness"], 66)
        self.assertEqual(richness["order_richness"], 30)
        self.assertEqual(richness["class_richness"], 4)

    def test_06_taxonomic_hierarchy_and_singletons(self):
        """Test 6: Verify singleton counts and multi-species genera in reference library."""
        hierarchy = analyze_taxonomic_hierarchy(self.ref_df)
        self.assertEqual(hierarchy["singleton_species_count"], 89)
        self.assertEqual(hierarchy["doubleton_species_count"], 1)
        self.assertEqual(hierarchy["doubleton_species_name"], "Ophidion marginatum")
        # Genera with named species: 81 total (73 singleton genera, 8 multi-species genera)
        self.assertEqual(hierarchy["singleton_genera_count"], 73)
        self.assertEqual(hierarchy["multi_species_genera_count"], 8)
        self.assertIn("Anchoa", hierarchy["multi_species_genera_list"])

    def test_07_missing_species_handling_no_fabrication(self):
        """Test 7: Verify missing species labels are counted without fabricating names."""
        unlabeled_sub = self.ref_df[self.ref_df["species"].isnull()]
        self.assertEqual(len(unlabeled_sub), 10)
        # All 10 unlabeled records retain valid Family, Order, and Class
        self.assertTrue(unlabeled_sub["family"].notnull().all())
        self.assertTrue(unlabeled_sub["order"].notnull().all())
        self.assertTrue(unlabeled_sub["class"].notnull().all())
        # 6 of the 10 retain valid Genus, while 4 stop at Family
        self.assertEqual(int(unlabeled_sub["genus"].notnull().sum()), 6)
        self.assertEqual(int(unlabeled_sub["genus"].isnull().sum()), 4)

    # -------------------------------------------------------------
    # 4. Duplicate Rows & Occurrence Distinction Tests
    # -------------------------------------------------------------
    def test_08_duplicate_row_handling_and_occurrence_distinction(self):
        """Test 8: Verify raw 13,736 rows are distinguished from 1,604 unique positive occurrences."""
        self.assertIsNotNone(self.records_df)
        # Deduplicated positive occurrences should be exactly 1,604
        self.assertEqual(len(self.records_df), 1604)
        # Across 98 detected sequences
        self.assertEqual(self.records_df["sequence_id"].nunique(), 98)
        # Total reference sequences remains 101
        self.assertEqual(len(self.ref_df), 101)

    def test_09_country_aggregation(self):
        """Test 9: Verify country aggregation identifies United States (US) waters."""
        country_csv = os.path.join(PROCESSED_DIR, "country_biodiversity_summary.csv")
        self.assertTrue(os.path.exists(country_csv))
        country_df = pd.read_csv(country_csv)
        self.assertEqual(len(country_df), 1)
        self.assertEqual(country_df.iloc[0]["country_code"], "US")
        self.assertEqual(country_df.iloc[0]["sampling_stations_count"], 36)

    # -------------------------------------------------------------
    # 5. Prediction-to-Biodiversity Integration Tests
    # -------------------------------------------------------------
    def test_10_prediction_integration_known_reference(self):
        """Test 10: Verify known_reference_like sequence returns valid geographic reference evidence."""
        sample_seq = self.ref_df.iloc[0]["sequence_normalized"]
        res = analyze_prediction(sample_seq)
        self.assertEqual(res["qc_status"], "PASSED")
        self.assertEqual(res["status"], "known_reference_like")
        
        bio = res["biodiversity_intelligence"]
        self.assertIsNotNone(bio)
        self.assertEqual(bio["projection_status"], "SUPPORTED_BY_REFERENCE_RECORDS")
        self.assertGreater(bio["supporting_reference_records_in_dataset"], 0)
        self.assertIn("United States", bio["countries_represented_in_dataset"])
        self.assertIn("NJ OTS survey", bio["scientific_scope_note"])

    def test_11_prediction_integration_out_of_reference(self):
        """Test 11: Verify out_of_reference sequence strictly forbids species distribution projection."""
        sample_seq = self.ref_df.iloc[0]["sequence_normalized"]
        # Heavy mutation (33%)
        mutated = list(sample_seq)
        for i in range(0, len(mutated), 3):
            mutated[i] = "T" if mutated[i] == "A" else "A"
        mutated_seq = "".join(mutated)

        res = analyze_prediction(mutated_seq)
        self.assertEqual(res["qc_status"], "PASSED")
        self.assertEqual(res["status"], "out_of_reference")
        
        bio = res["biodiversity_intelligence"]
        self.assertEqual(bio["projection_status"], "NO_PROJECTION_PERMITTED")
        self.assertEqual(bio["supporting_reference_records_in_dataset"], 0)
        self.assertEqual(len(bio["countries_represented_in_dataset"]), 0)
        self.assertIn("out_of_reference", res["uncertainty_warning"])
        self.assertIn("nearest_reference_context", bio)

    def test_12_prediction_integration_qc_failure(self):
        """Test 12: Verify biological QC failure halts pipeline before biodiversity analysis."""
        res = analyze_prediction("ACTGNNNN123!")
        self.assertEqual(res["qc_status"], "FAILED")
        self.assertEqual(res["status"], "qc_failed")
        self.assertIsNone(res["prediction"])
        self.assertIsNone(res["biodiversity_intelligence"])

    # -------------------------------------------------------------
    # 6. Schema Validation & File Integrity Tests
    # -------------------------------------------------------------
    def test_13_processed_tables_schema_validation(self):
        """Test 13: Verify all 5 generated processed CSV tables exist and have valid schemas."""
        required_tables = {
            "biodiversity_reference_records.csv": ["record_id", "sequence_id", "species", "decimalLatitude", "decimalLongitude", "organismQuantity"],
            "country_biodiversity_summary.csv": ["country_code", "survey_region", "total_raw_assays", "sampling_stations_count"],
            "taxonomic_summary.csv": ["taxonomic_rank", "distinct_taxa_count", "coverage_pct"],
            "species_distribution_summary.csv": ["sequence_id", "species", "positive_occurrences", "occurrence_prevalence_pct"],
            "diversity_metrics.csv": ["station_latitude", "station_longitude", "shannon_diversity_H_reads", "simpson_diversity_1_minus_D"]
        }
        for filename, required_cols in required_tables.items():
            path = os.path.join(PROCESSED_DIR, filename)
            self.assertTrue(os.path.exists(path), f"Missing data product: {filename}")
            df = pd.read_csv(path)
            for col in required_cols:
                self.assertIn(col, df.columns, f"Column {col} missing from {filename}")

    def test_14_figures_exist_and_non_empty(self):
        """Test 14: Verify all 8 Phase 7 visualization figures exist and are non-empty."""
        fig_dir = os.path.join(REPORTS_DIR, "figures", "phase7")
        expected_figs = [
            "taxonomic_composition.png",
            "country_richness.png",
            "sampling_distribution.png",
            "taxonomic_hierarchy.png",
            "reference_record_abundance.png",
            "geographic_station_map.png",
            "shannon_diversity_stations.png",
            "sequence_coverage_summary.png"
        ]
        for fig_name in expected_figs:
            p = os.path.join(fig_dir, fig_name)
            self.assertTrue(os.path.exists(p), f"Missing figure: {fig_name}")
            self.assertGreater(os.path.getsize(p), 1000, f"Figure file is empty or corrupted: {fig_name}")

    def test_15_raw_dataset_sha256_unmodified(self):
        """Test 15: Verify raw dataset SHA-256 remains strictly untouched."""
        with open(RAW_PATH, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(h, EXPECTED_SHA256)

    def test_16_station_aggregation_pipeline_deterministic_toy(self):
        """Test 16: Verify exact station aggregation pipeline on deterministic synthetic data.
        
        Toy scenario:
        Station at (40.0, -74.0) with 2 water samples:
        - Sample 1 (Surface, 1m):
            - Taxon A has primer replicates: AF=100, AFS=100 -> mean=100 reads
            - Taxon B has primer replicates: AF=200, AFS=200 -> mean=200 reads
        - Sample 2 (Bottom, 20m):
            - Taxon A has primer replicates: AF=100, AFS=100 -> mean=100 reads
            - Taxon B: not detected (0 reads)
            
        Deduplicated occurrence records:
        - Rec 1: Taxon A at Surface, reads=100
        - Rec 2: Taxon B at Surface, reads=200
        - Rec 3: Taxon A at Bottom, reads=100
        
        Station-level pooled abundance:
        - Taxon A total reads = 100 + 100 = 200
        - Taxon B total reads = 200
        - Total station reads = 400
        - p_A = 200/400 = 0.5, p_B = 200/400 = 0.5
        - H_reads = ln(2) = 0.693147
        - Incidence: Taxon A detected in 2 samples, Taxon B in 1 sample.
          q_A = 2/3, q_B = 1/3 -> H_inc = -(2/3*ln(2/3) + 1/3*ln(1/3)) = 0.636514
        """
        toy_records = pd.DataFrame([
            {
                "record_id": "REC_0001",
                "sequence_id": "SEQ_A",
                "species": "Species A",
                "genus": "GenusA",
                "family": "FamA",
                "sampname": "STA_01_SURF",
                "decimalLatitude": 40.0,
                "decimalLongitude": -74.0,
                "depth": 1.0,
                "locality": "Toy Station 01",
                "organismQuantity": 100.0
            },
            {
                "record_id": "REC_0002",
                "sequence_id": "SEQ_B",
                "species": "Species B",
                "genus": "GenusB",
                "family": "FamB",
                "sampname": "STA_01_SURF",
                "decimalLatitude": 40.0,
                "decimalLongitude": -74.0,
                "depth": 1.0,
                "locality": "Toy Station 01",
                "organismQuantity": 200.0
            },
            {
                "record_id": "REC_0003",
                "sequence_id": "SEQ_A",
                "species": "Species A",
                "genus": "GenusA",
                "family": "FamA",
                "sampname": "STA_01_BOTT",
                "decimalLatitude": 40.0,
                "decimalLongitude": -74.0,
                "depth": 20.0,
                "locality": "Toy Station 01",
                "organismQuantity": 100.0
            }
        ])

        station_df = compute_station_diversity_metrics(toy_records, tax_level="species")
        self.assertEqual(len(station_df), 1)
        row = station_df.iloc[0]

        # Station coordinates and pooling
        self.assertEqual(row["station_latitude"], 40.0)
        self.assertEqual(row["station_longitude"], -74.0)
        self.assertEqual(row["taxonomic_richness_S"], 2)
        self.assertEqual(row["detection_records_count"], 3)
        self.assertEqual(row["depth_meters"], round((1.0 + 1.0 + 20.0)/3, 1))  # 7.3 m mean across 3 records

        # H_reads must be exactly ln(2)
        self.assertAlmostEqual(row["shannon_diversity_H_reads"], math.log(2), places=5)
        # H_incidence must match q = [2/3, 1/3]
        expected_h_inc = -( (2/3)*math.log(2/3) + (1/3)*math.log(1/3) )
        self.assertAlmostEqual(row["shannon_diversity_H_incidence"], expected_h_inc, places=5)
        # Simpson 1 - (0.5^2 + 0.5^2) = 0.50
        self.assertAlmostEqual(row["simpson_diversity_1_minus_D"], 0.50, places=5)
        # Pielou evenness = H / ln(2) = 1.0
        self.assertAlmostEqual(row["pielou_evenness_J"], 1.0, places=5)

    def test_17_multi_rank_diversity_options(self):
        """Test 17: Verify compute_station_diversity_metrics supports named_species_only, genus, and family."""
        # 1. named_species_only excludes the 211 unlabeled records
        st_named = compute_station_diversity_metrics(self.records_df, tax_level="named_species_only")
        self.assertEqual(len(st_named), 36)
        total_named_detections = st_named["detection_records_count"].sum()
        self.assertEqual(total_named_detections, 1393)  # 1604 - 211 = 1393

        # 2. genus level
        st_genus = compute_station_diversity_metrics(self.records_df, tax_level="genus")
        self.assertEqual(len(st_genus), 36)
        self.assertEqual(st_genus.iloc[0]["taxonomic_rank_evaluated"], "genus")

        # 3. family level
        st_family = compute_station_diversity_metrics(self.records_df, tax_level="family")
        self.assertEqual(len(st_family), 36)
        self.assertEqual(st_family.iloc[0]["taxonomic_rank_evaluated"], "family")


if __name__ == "__main__":
    unittest.main()
