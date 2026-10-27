"""SVM classifier: construction, leakage-safe tuning, saved artefact and generated results."""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models import svm_model as sm

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(BASE, "reports", "evaluation", "evaluation_results.json")
CLASSES = {"Teleostei", "Elasmobranchii", "Chondrostei", "Petromyzonti"}


@pytest.fixture(scope="module")
def data():
    return sm.load_class_training_data()


def test_training_distribution_is_259_with_expected_classes(data):
    counts = pd.Series(data["y"]).value_counts().to_dict()
    assert counts == {"Teleostei": 82, "Elasmobranchii": 17, "Chondrostei": 80, "Petromyzonti": 80}
    assert data["is_synth"].sum() == 158
    assert data["X"].shape == (259, 64)


def test_reference_library_is_101_genuine_and_uncontaminated():
    ref = pd.read_csv(os.path.join(BASE, "data", "processed", "reference_barcodes.csv"))
    assert len(ref) == 101 and ref["sequence_normalized"].nunique() == 101
    assert not ref["sequence_id"].astype(str).str.startswith("SYN").any()
    if "basisOfRecord" in ref.columns:
        assert not (ref["basisOfRecord"] == "Synthetic augmentation").any()


def test_no_taxonomy_columns_are_features(data):
    banned = {"class", "order", "family", "genus", "species", "sequence_id"}
    assert not banned & set(data["feature_columns"])


def test_baseline_parameters_and_pipeline_structure():
    pipe = sm.make_pipeline()
    assert [n for n, _ in pipe.steps] == ["scaler", "svm"]
    svc = pipe.named_steps["svm"]
    assert (svc.C, svc.gamma, svc.kernel, svc.probability) == (1.0, "scale", "rbf", True)
    assert svc.random_state == 42


def test_pipeline_trains_predicts_valid_classes_and_probabilities(data):
    pipe = sm.make_pipeline().fit(data["X"], data["y"])
    pred = pipe.predict(data["X"][:30])
    assert set(pred) <= CLASSES
    pr = pipe.predict_proba(data["X"][:30])
    assert pr.shape == (30, 4)
    assert np.allclose(pr.sum(axis=1), 1.0)


def test_parent_groups_attach_every_synthetic_variant_to_its_genuine_parent(data):
    g, syn, y = data["groups"], data["is_synth"], data["y"]
    assert len(set(g[syn])) == 2
    genuine_ids = set(data["df"].loc[~syn, "sequence_id"])
    assert set(g[syn]) <= genuine_ids
    for parent in set(g[syn]):
        assert len(set(y[g == parent])) == 1   # a group never mixes classes


def test_tuning_splits_never_validate_on_synthetic_or_leak_siblings(data):
    splits = sm.tuning_splits(data["y"], data["groups"], data["is_synth"])
    assert len(splits) == 5
    for tr, va in splits:
        assert not data["is_synth"][va].any()
        assert set(data["y"][va]) <= {"Teleostei", "Elasmobranchii"}
        assert not set(tr) & set(va)
        assert not set(data["groups"][tr]) & set(data["groups"][va])
        assert {"Chondrostei", "Petromyzonti"} <= set(data["y"][tr])   # always kept in training


def test_grid_search_completes_and_records_best_params(data):
    t = sm.tune_svm(data["X"], data["y"], data["groups"], data["is_synth"], n_jobs=1)
    assert t["best_params"]["C"] in sm.C_GRID
    assert t["best_params"]["gamma"] in sm.GAMMA_GRID
    assert t["n_candidates"] == len(sm.C_GRID) * len(sm.GAMMA_GRID) == len(t["search_results"])
    assert 0.0 <= t["best_cv_score"] <= 1.0
    assert t["n_cv_folds"] == 5


def test_tuning_is_deterministic(data):
    a = sm.tune_svm(data["X"], data["y"], data["groups"], data["is_synth"])
    b = sm.tune_svm(data["X"], data["y"], data["groups"], data["is_synth"])
    assert a["best_params"] == b["best_params"] and a["best_cv_score"] == b["best_cv_score"]


def test_train_and_save_artefact_loads_and_predicts(tmp_path, data):
    path = str(tmp_path / "baseline_svm.joblib")
    sm.train_and_save(path)
    art = joblib.load(path)
    assert {"pipeline", "feature_set", "feature_columns", "config", "metrics", "random_state"} <= set(art)
    assert [n for n, _ in art["pipeline"].steps] == ["scaler", "svm"]   # scaler travels with the model
    pred = art["pipeline"].predict(data["X"][:10])
    assert set(pred) <= CLASSES
    assert art["metrics"]["training_samples"] == 259


def test_shipped_artefact_loads():
    path = os.path.join(BASE, "models", "baseline_svm.joblib")
    if not os.path.exists(path):
        pytest.skip("run: PYTHONPATH=. python src/models/svm_model.py")
    art = joblib.load(path)
    assert set(art["pipeline"].classes_) == CLASSES


def test_production_class_model_still_random_forest():
    art = joblib.load(os.path.join(BASE, "models", "taxonomy_class_model.joblib"))
    assert art["model_type"] == "Random Forest"


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="evaluation results not generated")
class TestSvmResults:
    @pytest.fixture(scope="class")
    def res(self):
        return json.load(open(RESULTS))

    def test_all_five_models_present_at_class_rank(self, res):
        assert set(res["taxonomy"]["class"]["models"]) >= {
            "KNN", "Random Forest", "XGBoost", "SVM Baseline", "SVM Optimized"}

    def test_existing_models_preserved_at_every_evaluated_rank(self, res):
        for rank in ("class", "order", "family"):
            assert {"KNN", "Random Forest", "XGBoost"} <= set(res["taxonomy"][rank]["models"])

    def test_svm_block_records_search_details(self, res):
        o = res["taxonomy"]["class"]["svm"]["optimization"]
        assert o["scoring"] == "balanced_accuracy"
        assert o["param_grid"]["C"] == sm.C_GRID and o["param_grid"]["gamma"] == sm.GAMMA_GRID
        assert o["best_params"]["C"] in sm.C_GRID and o["n_cv_folds"] == 5
        assert len(o["search_results"]) == 36

    def test_confusion_matrix_and_per_class_cover_all_four_classes_in_fixed_order(self, res):
        for proto in ("protocol_a",):
            for v in ("baseline", "optimized"):
                blk = res["taxonomy"]["class"]["svm"][proto][v]
                assert blk["confusion_labels"] == sm.CLASS_ORDER
                assert [r["class"] for r in blk["per_class"]] == sm.CLASS_ORDER
                assert np.array(blk["confusion_matrix"]).shape == (4, 4)

    def test_protocol_a_supports_match_training_distribution(self, res):
        rows = res["taxonomy"]["class"]["svm"]["protocol_a"]["baseline"]["per_class"]
        assert {r["class"]: r["support"] for r in rows} == {
            "Teleostei": 82, "Elasmobranchii": 17, "Chondrostei": 80, "Petromyzonti": 80}

    def test_before_after_difference_is_arithmetically_consistent(self, res):
        for proto in ("protocol_a",):
            for k, d in res["taxonomy"]["class"]["svm"][proto]["before_after"].items():
                if d["absolute_change"] is not None:
                    assert abs(d["absolute_change"] - (d["optimized"] - d["baseline"])) < 1e-9

    def test_production_model_not_silently_replaced(self, res):
        dec = res["taxonomy"]["class"]["svm"]["production_decision"]
        assert dec["production_model"] == "Random Forest"
        if dec["replace_with_svm_optimized"]:
            m = res["taxonomy"]["class"]["models"]
            assert m["SVM Optimized"]["balanced_accuracy"] > m["Random Forest"]["balanced_accuracy"]
            assert m["SVM Optimized"]["macro_f1"] > m["Random Forest"]["macro_f1"]
        # the artefact in models/ must still be a Random Forest at class rank either way
        assert res["taxonomy"]["class"]["production_model"] == "Random Forest"


def test_evaluation_page_imports_and_renders_with_svm():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    code = ("import sys; sys.path.insert(0, %r)\n"
            "from app.pages.evaluation import render_evaluation_page\n"
            "render_evaluation_page()\n" % BASE)
    at = AppTest.from_string(code, default_timeout=120).run()
    assert not at.exception, [str(e.value) for e in at.exception]
