"""
tests/test_phase9_dashboard.py
Unit and Integration Tests for Phase 9: BioScan Interactive Dashboard

Verifies:
1. Dashboard app and component modules import cleanly
2. Prediction page dependencies and cached loaders initialize correctly
3. Phase 8 predictor initializes properly from the dashboard layer
4. Dynamic example sequences load accurately from processed reference data
5. Batch DataFrame generation and CSV export integrate with Phase 8
6. Required dashboard data files and columns exist
7. Status badge rendering and configuration completeness
8. Plotly chart generation utilities execute without errors
9. Out-of-reference result suppresses geographic projection
10. Fault tolerance: invalid sequence does not break batch processing
11. Raw dataset SHA-256 remains strictly unmodified
"""

import os
import sys
import unittest
import hashlib
import pandas as pd

# Add project root to path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "hackathon_dna_taxonomy_merged.csv")
EXPECTED_RAW_SHA256 = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"


class TestPhase9Dashboard(unittest.TestCase):
    """Test suite for Phase 9 Streamlit Dashboard architecture and components."""

    def test_01_app_and_pages_import_successfully(self):
        """Test 1: Verify all core application and page modules import cleanly."""
        import app.app as main_app
        import app.pages.identify as p_id
        import app.pages.batch_analysis as p_batch
        import app.pages.biodiversity as p_bio
        import app.pages.about as p_about
        
        self.assertTrue(callable(main_app.main))
        self.assertTrue(callable(p_id.render_identify_page))
        self.assertTrue(callable(p_batch.render_batch_page))
        self.assertTrue(callable(p_bio.render_biodiversity_page))
        self.assertTrue(callable(p_about.render_about_page))

    def test_02_components_import_successfully(self):
        """Test 2: Verify all dashboard presentation components import cleanly."""
        from app.components import (
            render_prediction_card,
            render_taxonomy_hierarchy,
            render_confidence_and_evidence,
            render_similarity_table,
            render_biodiversity_context,
            plot_status_distribution,
            plot_taxonomic_breakdown,
            plot_station_diversity_map,
            plot_diversity_vs_depth
        )
        self.assertTrue(callable(render_prediction_card))
        self.assertTrue(callable(render_taxonomy_hierarchy))
        self.assertTrue(callable(render_confidence_and_evidence))
        self.assertTrue(callable(render_similarity_table))
        self.assertTrue(callable(render_biodiversity_context))
        self.assertTrue(callable(plot_status_distribution))
        self.assertTrue(callable(plot_taxonomic_breakdown))
        self.assertTrue(callable(plot_station_diversity_map))
        self.assertTrue(callable(plot_diversity_vs_depth))

    def test_03_predictor_initialization_from_dashboard(self):
        """Test 3: Verify BioScan predictor can be loaded from dashboard caching loader."""
        from app.utils.ui_helpers import load_predictor
        # Streamlit cached function can be called directly
        predictor = load_predictor()
        self.assertIsNotNone(predictor)
        self.assertTrue(hasattr(predictor, "predict"))
        self.assertTrue(hasattr(predictor, "predict_fasta"))
        self.assertTrue(hasattr(predictor, "predict_batch"))

    def test_04_example_reference_sequence_prediction(self):
        """Test 4: Verify an example sequence from reference data passes through the prediction layer."""
        from app.utils.ui_helpers import load_predictor, load_reference_barcodes, get_example_sequences
        
        ref_df = load_reference_barcodes()
        self.assertFalse(ref_df.empty)
        examples = get_example_sequences(ref_df)
        self.assertIn("Exact Reference — Ophidion marginatum (Striped cusk-eel)", examples)
        
        ex = examples["Exact Reference — Ophidion marginatum (Striped cusk-eel)"]
        predictor = load_predictor()
        res = predictor.predict(ex["sequence"], sequence_id=ex["sequence_id"])

        self.assertEqual(res["input"]["sequence_id"], ex["sequence_id"])
        self.assertEqual(res["qc"]["status"], "passed")
        self.assertEqual(res["prediction"]["status"], "known_reference_like")
        self.assertIn(res["prediction"]["class"], ["Teleostei", "Actinopterygii"])
        self.assertEqual(res["prediction"]["order"], "Ophidiiformes")
        self.assertEqual(res["similarity"]["nearest_distance"], 0.0)

    def test_05_batch_dataframe_generation(self):
        """Test 5: Verify multi-sequence batch prediction produces a valid DataFrame."""
        from app.utils.ui_helpers import load_predictor
        from src.prediction.batch_predict import batch_results_to_dataframe

        predictor = load_predictor()
        batch_input = [
            {"sequence_id": "SEQ_A", "sequence": "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAC"},
            {"sequence_id": "SEQ_B", "sequence": "TAGCGTAATACCGTACCGTACTACGTATCGTAGCTAGCTAGCTAGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"}
        ]
        results = predictor.predict_batch(batch_input)
        self.assertEqual(len(results), 2)

        df = batch_results_to_dataframe(results)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        expected_cols = [
            "sequence_id", "sequence_length", "qc_status", "prediction_status",
            "class", "order", "family", "genus", "species", "nearest_reference",
            "nearest_distance", "nearest_similarity"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_06_dashboard_data_files_exist(self):
        """Test 6: Verify all processed datasets consumed by the dashboard exist on disk."""
        data_dir = os.path.join(BASE_DIR, "data", "processed")
        required_files = [
            "reference_barcodes.csv",
            "diversity_metrics.csv",
            "species_distribution_summary.csv",
            "biodiversity_reference_records.csv"
        ]
        for fname in required_files:
            fpath = os.path.join(data_dir, fname)
            self.assertTrue(os.path.exists(fpath), f"Missing required file: {fname}")
            self.assertGreater(os.path.getsize(fpath), 0, f"File is empty: {fname}")

    def test_07_required_dataset_columns_exist(self):
        """Test 7: Verify required columns exist in processed dashboard datasets."""
        from app.utils.ui_helpers import (
            load_reference_barcodes,
            load_diversity_metrics,
            load_species_distribution,
            load_biodiversity_records
        )
        ref_df = load_reference_barcodes()
        self.assertTrue({"sequence_id", "sequence_normalized", "class", "order", "species"}.issubset(ref_df.columns))

        div_df = load_diversity_metrics()
        self.assertTrue({"station_latitude", "station_longitude", "station_name", "shannon_diversity_H_reads"}.issubset(div_df.columns))

        dist_df = load_species_distribution()
        self.assertTrue({"sequence_id", "species", "unique_stations", "station_prevalence_pct"}.issubset(dist_df.columns))

        rec_df = load_biodiversity_records()
        self.assertTrue({"record_id", "sequence_id", "decimalLatitude", "decimalLongitude", "organismQuantity"}.issubset(rec_df.columns))

    def test_08_out_of_reference_result_suppresses_geographic_projection(self):
        """Test 8: Ensure out-of-reference prediction result strictly suppresses geographic projection."""
        from app.utils.ui_helpers import load_predictor
        predictor = load_predictor()
        # Synthetic divergent sequence (>15% Levenshtein distance)
        divergent_seq = "TAGCGTAATACCGTACCGTACTACGTATCGTAGCTAGCTAGCTAGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
        res = predictor.predict(divergent_seq, sequence_id="SYN_OUT_OF_REF")

        self.assertEqual(res["prediction"]["status"], "out_of_reference")
        self.assertGreater(res["similarity"]["nearest_distance"], 0.150)
        # Geographic distribution MUST be strictly suppressed
        self.assertEqual(res["biodiversity"]["projection_status"], "NO_PROJECTION_PERMITTED")
        self.assertEqual(res["biodiversity"]["supporting_reference_records_in_dataset"], 0)
        self.assertIsNone(res["biodiversity"].get("survey_region"))
        self.assertIn("No species or geographic distribution can be projected", res["biodiversity"]["scientific_scope_note"])

    def test_09_one_invalid_sequence_does_not_crash_batch(self):
        """Test 9: Ensure fault isolation: one invalid sequence in batch does not fail the valid sequences."""
        from app.utils.ui_helpers import load_predictor
        from src.prediction.batch_predict import batch_results_to_dataframe

        predictor = load_predictor()
        batch = [
            {"sequence_id": "VALID_1", "sequence": "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAC"},
            {"sequence_id": "INVALID_1", "sequence": "ACTGNNNNNACTG"},
            {"sequence_id": "VALID_2", "sequence": "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAA"}
        ]
        results = predictor.predict_batch(batch)
        self.assertEqual(len(results), 3)

        self.assertEqual(results[0]["qc"]["status"], "passed")
        self.assertEqual(results[1]["qc"]["status"], "failed")
        self.assertEqual(results[2]["qc"]["status"], "passed")

        df = batch_results_to_dataframe(results)
        self.assertEqual(len(df), 3)
        self.assertEqual(df.iloc[1]["prediction_status"], "qc_failed")
        self.assertEqual(df.iloc[0]["prediction_status"], "known_reference_like")

    def test_10_status_badge_rendering(self):
        """Test 10: Verify HTML badge generation for all operational status tiers."""
        from app.utils.ui_helpers import render_status_badge, STATUS_CONFIG

        for status_key in ["known_reference_like", "low_confidence", "out_of_reference", "qc_failed"]:
            badge_html = render_status_badge(status_key)
            self.assertIn("display: inline-flex", badge_html)
            self.assertIn(STATUS_CONFIG[status_key]["label"], badge_html)

    def test_11_plotly_chart_generation(self):
        """Test 11: Verify Plotly charts generate valid figure objects."""
        from app.components import (
            plot_status_distribution,
            plot_taxonomic_breakdown,
            plot_station_diversity_map,
            plot_diversity_vs_depth
        )
        from app.utils.ui_helpers import load_diversity_metrics
        import plotly.graph_objects as go

        # Test status distribution chart
        mock_df = pd.DataFrame({
            "prediction_status": ["known_reference_like", "known_reference_like", "low_confidence", "out_of_reference"],
            "class": ["Actinopterygii", "Actinopterygii", "Elasmobranchii", "Actinopterygii"]
        })
        fig_status = plot_status_distribution(mock_df)
        self.assertIsInstance(fig_status, go.Figure)

        fig_tax = plot_taxonomic_breakdown(mock_df, rank="class")
        self.assertIsInstance(fig_tax, go.Figure)

        # Test diversity map chart
        div_df = load_diversity_metrics()
        fig_map = plot_station_diversity_map(div_df)
        self.assertIsInstance(fig_map, go.Figure)

        fig_depth = plot_diversity_vs_depth(div_df)
        self.assertIsInstance(fig_depth, go.Figure)

    def test_13_dashboard_reference_library_kpis_match_phase7(self):
        """Test 13: Verify dashboard reference library KPIs match validated Phase 7 taxonomic counts."""
        from app.utils.ui_helpers import load_reference_barcodes
        from src.biodiversity.taxonomy_analysis import compute_taxonomic_richness

        ref_df = load_reference_barcodes()
        self.assertFalse(ref_df.empty)
        richness = compute_taxonomic_richness(ref_df)

        self.assertEqual(richness["unique_sequences"], 101)
        self.assertEqual(richness["species_richness"], 90)
        self.assertEqual(richness["genus_richness"], 87)
        self.assertEqual(richness["family_richness"], 66)
        self.assertEqual(richness["order_richness"], 30)
        self.assertEqual(richness["class_richness"], 4)
        self.assertEqual(richness["kingdom_richness"], 1)

    def test_12_raw_dataset_sha256_unmodified(self):
        """Test 12: Ensure raw dataset remains bit-for-bit unchanged."""
        self.assertTrue(os.path.exists(RAW_DATA_PATH), "Raw dataset file missing!")
        with open(RAW_DATA_PATH, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest().upper()
        self.assertEqual(
            actual_sha256,
            EXPECTED_RAW_SHA256,
            f"Raw dataset SHA-256 corrupted! Expected {EXPECTED_RAW_SHA256}, got {actual_sha256}"
        )


if __name__ == "__main__":
    unittest.main()

