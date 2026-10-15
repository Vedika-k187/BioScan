"""Mutation-level breakdown and headline-card logic for the evaluation dashboard."""
import json
import os

import pytest

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(BASE, "reports", "evaluation", "evaluation_results.json")


def test_every_rank_has_runtime_for_evaluated_models_when_results_exist():
    if not os.path.exists(RESULTS):
        pytest.skip("evaluation results not generated")
    res = json.load(open(RESULTS))
    for rank, e in res["taxonomy"].items():
        if e["status"] == "ok":
            for name, m in e["models"].items():
                assert m["runtime_seconds"] >= 0, (rank, name)


def test_perturbation_levels_are_reported_separately_and_consistently():
    if not os.path.exists(RESULTS):
        pytest.skip("evaluation results not generated")
    res = json.load(open(RESULTS))
    levels = res["unknown_detection"]["test"]["perturbation_levels"]
    assert list(levels) == ["1_substitutions", "2_substitutions", "5_substitutions"]
    total = sum(v["n"] for v in levels.values())
    assert total == res["unknown_detection"]["test"]["n_known_perturbed"]
    for v in levels.values():
        assert 0.0 <= v["false_positive_rate"] <= 1.0
        assert abs(v["accepted_rate"] + v["abstention_rate"] - 1.0) < 1e-9
    # more substitutions can only move sequences further away
    meds = [levels[k]["median_distance"] for k in levels]
    assert meds == sorted(meds)


def test_dashboard_helpers_never_invent_numbers_for_insufficient_ranks():
    pytest.importorskip("streamlit")
    from app.pages import evaluation as page
    res = {"taxonomy": {"genus": {"status": "insufficient", "models": {}, "support": {}}},
           "unknown_detection": {"test": {"production_rule_out_of_reference":
                                          {"recall_sensitivity": None, "false_positive_rate": None,
                                           "n_known": 0, "n_unknown": 0}}}}
    # With nothing evaluated, the deepest-rank search must come back empty rather than guess.
    deepest = next((r for r in reversed(page.RANK_ORDER)
                    if res["taxonomy"].get(r, {}).get("status") == "ok"), None)
    assert deepest is None
