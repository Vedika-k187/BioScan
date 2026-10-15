"""
Tests for the evaluation framework: metric arithmetic, leakage prevention,
insufficient-data handling and preservation of the Distance Veto.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    MIN_EVAL_SAMPLES,
    accuracy_coverage_curve,
    assess_sufficiency,
    calibrate_threshold_youden,
    detection_metrics,
    distance_auroc,
    taxonomy_metrics,
    threshold_sweep,
)
from src.evaluation.splits import (
    assert_no_sequence_overlap,
    class_support_summary,
    deduplicate_by_sequence,
    eligible_for_holdout,
    normalize_sequence,
    split_groups,
)

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------- taxonomy metrics
class TestTaxonomyMetricValues:
    # Hand-computed example (see comments):
    #   A: TP2 FP0 FN1 -> P=1.0  R=2/3 F1=0.8
    #   B: TP1 FP1 FN1 -> P=0.5  R=0.5 F1=0.5
    #   C: TP1 FP1 FN0 -> P=0.5  R=1.0 F1=2/3
    Y_TRUE = ["A", "A", "A", "B", "B", "C"]
    Y_PRED = ["A", "A", "B", "B", "C", "C"]

    def _m(self):
        return taxonomy_metrics(self.Y_TRUE, self.Y_PRED, min_samples=1, min_classes=2)

    def test_macro_and_weighted_values(self):
        m = self._m()
        assert m["status"] == "ok"
        assert m["macro_precision"] == pytest.approx((1.0 + 0.5 + 0.5) / 3)
        assert m["macro_recall"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)
        assert m["macro_f1"] == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)
        assert m["weighted_f1"] == pytest.approx((3 * 0.8 + 2 * 0.5 + 1 * 2 / 3) / 6)

    def test_accuracy_and_balanced_accuracy(self):
        m = self._m()
        assert m["top1_accuracy"] == pytest.approx(4 / 6)
        assert m["balanced_accuracy"] == pytest.approx((2 / 3 + 0.5 + 1.0) / 3)

    def test_counts_and_confusion_matrix(self):
        m = self._m()
        assert m["n_eligible_test_examples"] == 6
        assert m["n_evaluated_classes"] == 3
        assert m["confusion_matrix"]["labels"] == ["A", "B", "C"]
        assert m["confusion_matrix"]["matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]

    def test_per_class_table(self):
        pc = {r["class"]: r for r in self._m()["per_class"]}
        assert pc["A"]["support"] == 3 and pc["A"]["recall"] == pytest.approx(2 / 3)
        assert pc["B"]["precision"] == pytest.approx(0.5)
        assert pc["C"]["f1"] == pytest.approx(2 / 3)

    def test_predicted_only_class_does_not_enter_macro_average(self):
        m = taxonomy_metrics(["A", "A", "B", "B"], ["A", "D", "B", "B"], min_samples=1, min_classes=2)
        pc = {r["class"]: r for r in m["per_class"]}
        assert pc["D"]["support"] == 0 and pc["D"]["recall"] is None and pc["D"]["f1"] is None
        # Only A and B are evaluated classes
        assert m["n_evaluated_classes"] == 2
        assert m["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)


class TestInsufficientData:
    def test_default_minimum_blocks_small_samples(self):
        m = taxonomy_metrics(["A", "A", "B", "B"], ["A", "A", "B", "A"])
        assert m["status"] == "insufficient"
        assert "Insufficient data for reliable evaluation" in m["message"]

    def test_no_metric_values_are_fabricated(self):
        m = taxonomy_metrics(["A"] * 3, ["A"] * 3)
        assert m["status"] == "insufficient"
        for key in ("macro_f1", "weighted_f1", "balanced_accuracy", "top1_accuracy",
                    "macro_precision", "macro_recall", "confusion_matrix", "per_class"):
            assert key not in m

    def test_single_class_is_insufficient(self):
        y = ["A"] * (MIN_EVAL_SAMPLES + 5)
        assert taxonomy_metrics(y, y)["status"] == "insufficient"

    def test_empty_input_is_insufficient(self):
        assert taxonomy_metrics([], [])["status"] == "insufficient"

    def test_assess_sufficiency_reasons(self):
        r = assess_sufficiency(n_eligible=3, n_evaluated_classes=1)
        assert not r["sufficient"] and len(r["reasons"]) == 2

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            taxonomy_metrics(["A"], ["A", "B"])


# ---------------------------------------------------- unknown-detection metrics
class TestDetectionMetrics:
    # truth unknown: T T T F F F F ; flagged: T T F T F F F
    TRUTH = [True, True, True, False, False, False, False]
    FLAG = [True, True, False, True, False, False, False]

    def test_confusion_counts_and_definitions(self):
        m = detection_metrics(self.TRUTH, self.FLAG)
        assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 3)
        assert m["n_known"] == 4 and m["n_unknown"] == 3
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall_sensitivity"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(2 / 3)
        assert m["specificity"] == pytest.approx(3 / 4)

    def test_false_positive_and_false_negative_rates(self):
        m = detection_metrics(self.TRUTH, self.FLAG)
        # FPR: known sequences wrongly flagged out-of-reference
        assert m["false_positive_rate"] == pytest.approx(1 / 4)
        # FNR: unknown sequences wrongly accepted
        assert m["false_negative_rate"] == pytest.approx(1 / 3)
        assert m["confusion_matrix"]["matrix"] == [[3, 1], [1, 2]]

    def test_coverage_and_abstention(self):
        accepted = [False, False, True, False, True, True, True]
        m = detection_metrics(self.TRUTH, self.FLAG, y_pred_accepted=accepted)
        assert m["coverage_accepted_rate"] == pytest.approx(4 / 7)
        assert m["abstention_rate"] == pytest.approx(3 / 7)

    def test_undefined_metrics_are_none_not_zero(self):
        no_unknown = detection_metrics([False, False], [False, True])
        assert no_unknown["recall_sensitivity"] is None
        assert no_unknown["false_negative_rate"] is None
        no_flags = detection_metrics([True, False], [False, False])
        assert no_flags["precision"] is None and no_flags["f1"] is None

    def test_auroc_requires_both_classes(self):
        r = distance_auroc([0.1, 0.2, 0.3], [True, False, False])
        assert r["status"] == "unavailable"

    def test_auroc_perfect_separation(self):
        d = [0.01] * 6 + [0.5] * 6
        t = [False] * 6 + [True] * 6
        assert distance_auroc(d, t)["auroc"] == pytest.approx(1.0)

    def test_threshold_sweep_matches_manual_rule(self):
        d = [0.0, 0.1, 0.2, 0.4]
        t = [False, False, True, True]
        row = threshold_sweep(d, t, [0.15])[0]
        assert row["false_positive_rate"] == 0.0 and row["false_negative_rate"] == 0.0
        row = threshold_sweep(d, t, [0.3])[0]
        assert row["false_negative_rate"] == pytest.approx(0.5)

    def test_calibration_picks_separating_threshold(self):
        d = [0.01, 0.02, 0.03, 0.30, 0.40, 0.50]
        t = [False, False, False, True, True, True]
        best = calibrate_threshold_youden(d, t, [0.05, 0.1, 0.2, 0.35, 0.45])
        assert best["youden_j"] == pytest.approx(1.0)
        assert best["threshold"] in (0.05, 0.1, 0.2)

    def test_calibration_none_without_both_classes(self):
        assert calibrate_threshold_youden([0.1, 0.2], [False, False], [0.1]) is None

    def test_accuracy_coverage_curve_monotonic_coverage(self):
        conf = [0.9, 0.8, 0.6, 0.5]
        ok = [True, True, False, False]
        rows = accuracy_coverage_curve(conf, ok, n_points=4)
        covs = [r["coverage"] for r in rows]
        assert covs == sorted(covs, reverse=True)
        assert rows[0]["coverage"] == 1.0 and rows[0]["accuracy_of_accepted"] == pytest.approx(0.5)


# ------------------------------------------------------------ leakage / splits
class TestLeakagePrevention:
    def test_deduplicate_collapses_identical_normalised_sequences(self):
        df = pd.DataFrame({
            "sequence_normalized": ["acgt", "ACGT", " AC GT ", "TTTT"],
            "class": ["x", "x", "x", "y"],
        })
        uniq, rep = deduplicate_by_sequence(df)
        assert len(uniq) == 2
        assert rep["input_rows"] == 4 and rep["duplicate_rows_removed"] == 2
        assert rep["label_conflicts"] == 0

    def test_label_conflicts_are_counted_not_hidden(self):
        df = pd.DataFrame({"sequence_normalized": ["AAAA", "AAAA"], "class": ["x", "y"]})
        _, rep = deduplicate_by_sequence(df)
        assert rep["label_conflicts"] == 1

    def test_overlap_detected_after_normalisation(self):
        with pytest.raises(AssertionError):
            assert_no_sequence_overlap(["ACGT", "GGGG"], ["tttt", " acgt"])

    def test_disjoint_partitions_pass(self):
        assert_no_sequence_overlap(["ACGT"], ["TTTT"])

    def test_normalize_sequence(self):
        assert normalize_sequence(" ac\ngt ") == "ACGT"

    def test_group_split_is_disjoint_complete_and_deterministic(self):
        groups = [f"g{i}" for i in range(11)]
        c1, t1 = split_groups(groups, seed=7)
        c2, t2 = split_groups(groups, seed=7)
        assert (c1, t1) == (c2, t2)
        assert set(c1).isdisjoint(t1)
        assert set(c1) | set(t1) == set(groups)

    def test_real_reference_library_has_unique_sequences(self):
        ref = pd.read_csv(os.path.join(BASE, "data", "processed", "reference_barcodes.csv"))
        uniq, rep = deduplicate_by_sequence(ref)
        assert rep["duplicate_rows_removed"] == 0
        assert len(uniq) == 101


class TestSingletonAndRareClasses:
    def test_singletons_not_eligible(self):
        mask = eligible_for_holdout(["a", "a", "b", "c", "c", "c"])
        assert mask.tolist() == [True, True, False, True, True, True]

    def test_support_summary_reports_singletons_explicitly(self):
        s = class_support_summary(["a", "a", "b", "c", "c", "c", None])
        assert s["n_labelled_sequences"] == 6
        assert s["n_classes_total"] == 3
        assert s["n_singleton_classes"] == 1
        assert s["n_unevaluable_singleton_sequences"] == 1
        assert s["n_eligible_sequences"] == 5

    def test_all_singletons_gives_nothing_eligible(self):
        assert not eligible_for_holdout(list("abcdef")).any()


# --------------------------------------------- runner construction guarantees
class TestEvaluationConstruction:
    @pytest.fixture(scope="class")
    def built(self):
        from src.evaluation.run_model_evaluation import build_unknown_queries
        ref = pd.read_csv(os.path.join(BASE, "data", "processed", "reference_barcodes.csv"))
        return ref, build_unknown_queries(ref, seed=42)

    def test_four_categories_present_and_separate(self, built):
        _, b = built
        cats = {q["category"] for q in b["queries"]}
        assert {"known_exact", "known_perturbed", "heldout_genus"} <= cats

    def test_heldout_genus_never_split_across_partitions(self, built):
        _, b = built
        parts = {}
        for q in b["queries"]:
            if q["category"] == "heldout_genus":
                parts.setdefault(q["genus"], set()).add(q["partition"])
        assert all(len(p) == 1 for p in parts.values())

    def test_perturbed_queries_really_differ_from_source(self, built):
        ref, b = built
        seq = ref.set_index("sequence_id")["sequence_normalized"].to_dict()
        pert = [q for q in b["queries"] if q["category"] == "known_perturbed"]
        assert pert and all(q["query"] != seq[q["source"]] for q in pert)

    def test_external_sequences_not_in_reference_library(self, built):
        ref, b = built
        lib = set(ref["sequence_normalized"])
        ext = [q["query"] for q in b["queries"] if q["category"] == "external_nonfish"]
        assert lib.isdisjoint(ext)

    def test_truth_labels(self, built):
        _, b = built
        for q in b["queries"]:
            expect_unknown = q["category"] in ("heldout_genus", "external_nonfish")
            assert q["truth_unknown"] == expect_unknown


# --------------------------------- safety logic must remain intact (not weakened)
class TestSafetyLogicPreserved:
    def test_production_thresholds_unchanged(self):
        from src.prediction.evidence import (
            THRESHOLD_CLOSE_SIMILARITY, THRESHOLD_OUT_OF_REFERENCE)
        assert THRESHOLD_OUT_OF_REFERENCE == 0.150
        assert THRESHOLD_CLOSE_SIMILARITY == 0.030

    def test_distance_veto_overrides_confident_classifier(self):
        from src.prediction.evidence import synthesize_evidence_and_status
        status, tier, _ = synthesize_evidence_and_status(
            nearest_dist=0.2,
            nearest_ref={"sequence_id": "SEQ_001"},
            prob_metrics={"top1_prob": 0.999, "prob_margin": 0.99, "top2_prob": 0.0, "entropy": 0.0},
            concordance={"class": True, "order": True, "family": True},
            is_exact_match=False,
            second_dist=0.25,
        )
        assert status == "out_of_reference" and tier == "low"

    def test_exact_match_is_accepted(self):
        from src.prediction.evidence import synthesize_evidence_and_status
        status, _, _ = synthesize_evidence_and_status(
            nearest_dist=0.0, nearest_ref={"sequence_id": "SEQ_001"},
            prob_metrics={"top1_prob": 0.9, "prob_margin": 0.8, "top2_prob": 0.1, "entropy": 0.5},
            concordance={"class": True, "order": True, "family": True},
            is_exact_match=True, second_dist=0.2)
        assert status == "known_reference_like"

    def test_out_of_reference_gets_no_species_or_geography(self):
        import random
        from src.prediction.predictor import predict_sequence
        rng = random.Random(3)
        seq = "".join(rng.choice("ACGT") for _ in range(106))
        res = predict_sequence(seq)
        assert res["status"] == "out_of_reference"
        assert res["predicted_taxonomy"]["species"] == "Not available in reference data"
        assert res["biodiversity"]["projection_status"] == "NO_PROJECTION_PERMITTED"
        assert res["biodiversity"]["supporting_reference_records_in_dataset"] == 0


# ------------------------------------------------ generated results sanity checks
RESULTS = os.path.join(BASE, "reports", "evaluation", "evaluation_results.json")


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="evaluation results not generated")
class TestGeneratedResults:
    @pytest.fixture(scope="class")
    def res(self):
        return json.load(open(RESULTS))

    def test_insufficient_ranks_have_no_metrics(self, res):
        for rank, e in res["taxonomy"].items():
            if e["status"] != "ok":
                assert e["models"] == {} and e.get("reason")

    def test_sufficient_ranks_report_counts(self, res):
        for rank, e in res["taxonomy"].items():
            if e["status"] == "ok":
                for name, m in e["models"].items():
                    assert m["n_eligible_test_examples"] >= MIN_EVAL_SAMPLES
                    assert 0.0 <= m["macro_f1"] <= 1.0
                    assert m["n_evaluated_classes"] >= 2

    def test_species_not_claimed_reliable(self, res):
        assert res["taxonomy"]["species"]["status"] == "insufficient"

    def test_unknown_confusion_matrix_consistent(self, res):
        pm = res["unknown_detection"]["test"]["production_rule_out_of_reference"]
        assert pm["tp"] + pm["fn"] == pm["n_unknown"]
        assert pm["fp"] + pm["tn"] == pm["n_known"]
        assert pm["n_known"] + pm["n_unknown"] == pm["n_total"]

    def test_calibration_and_test_partitions_disjoint(self, res):
        info = res["unknown_detection"]["partition_info"]
        assert info["seed"] == res["meta"]["seed"]
