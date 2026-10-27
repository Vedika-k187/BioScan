"""
app/pages/evaluation.py
Model Evaluation page.

Presentation only. Every number shown is read from
reports/evaluation/evaluation_results.json, produced by
src/evaluation/run_model_evaluation.py from held-out predictions on unique sequences.
Nothing is hardcoded and nothing is computed from training-set scores. When a metric
cannot be computed the page shows an explicit unavailable state and the reason.
"""

import json
import os
from typing import Any, Dict, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_PATH = os.path.join(BASE_DIR, "reports", "evaluation", "evaluation_results.json")
RANK_ORDER = ["class", "order", "family", "genus", "species"]

HELP = {
    "macro_f1": "Macro F1: F1 computed per class, then a plain average. Every class counts equally, so "
                "poor performance on rare classes is visible. Preferred for imbalanced taxonomy.",
    "weighted_f1": "Weighted F1: per-class F1 averaged in proportion to class size. Dominated by "
                   "common classes, so it can look good while rare classes fail.",
    "balanced_accuracy": "Mean recall over classes. Equivalent to accuracy if all classes had equal size.",
    "accuracy": "Top-1 accuracy: fraction of held-out sequences whose single predicted label is correct. "
                "Inflated by the majority class when classes are imbalanced.",
    "macro_precision": "Of the sequences predicted as a class, the fraction that truly belong to it; "
                       "averaged equally over evaluated classes.",
    "macro_recall": "Of the sequences truly in a class, the fraction found; averaged equally over "
                    "evaluated classes.",
}


@st.cache_data(show_spinner=False)
def _load_results(path: str, mtime: float) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _f(x, pct: bool = False, digits: int = 3) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.1f}%" if pct else f"{x:.{digits}f}"


def _heatmap(labels, matrix, title: str, height: int = 460) -> go.Figure:
    z = matrix
    text = [[str(v) if v else "" for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z, x=labels, y=labels, text=text, texttemplate="%{text}",
            colorscale=[[0, "#111827"], [1, "#14b8a6"]], showscale=False,
            hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Predicted", tickangle=-45, automargin=True),
        yaxis=dict(title="True", autorange="reversed", automargin=True),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def _headline_cards(res: Dict[str, Any]) -> None:
    """Four fixed headline numbers: deepest evaluated rank, plus unknown detection."""
    tax = res["taxonomy"]
    deepest = next((r for r in reversed(RANK_ORDER) if tax.get(r, {}).get("status") == "ok"), None)
    items = []
    if deepest:
        e = tax[deepest]
        m = e["models"].get(e["production_model"]) or next(iter(e["models"].values()))
        label = deepest.capitalize()
        items += [
            (f"Macro F1 ({label})", _f(m.get("macro_f1")),
             HELP["macro_f1"] + f" Deepest rank with enough data; {e['production_model']}, n={m['n_eligible_test_examples']}."),
            (f"Macro recall ({label})", _f(m.get("macro_recall")), HELP["macro_recall"]),
        ]
    else:
        items += [("Macro F1", "N/A", "No rank has enough held-out data."),
                  ("Macro recall", "N/A", "No rank has enough held-out data.")]
    pm = res["unknown_detection"]["test"]["production_rule_out_of_reference"]
    items += [
        ("Unknown recall", _f(pm["recall_sensitivity"], pct=True),
         "Of truly out-of-reference test sequences, the fraction flagged out_of_reference (positive class)."),
        ("False-positive rate", _f(pm["false_positive_rate"], pct=True),
         "Known sequences wrongly flagged out_of_reference."),
    ]
    _metric_row(items)
    st.caption(
        f"Held-out results. Taxonomy figures are for the deepest evaluable rank "
        f"({deepest or 'none'}); unknown-detection figures use {pm['n_known']} altered-known and "
        f"{pm['n_unknown']} unknown test sequences. Not the confidence of any single prediction."
    )


def _model_rank_matrix(res: Dict[str, Any]) -> None:
    tax = res["taxonomy"]
    models = ["KNN", "Random Forest", "XGBoost", "SVM Baseline", "SVM Optimized"]
    rows = []
    for name in models:
        row = {"Model": name}
        for r in RANK_ORDER:
            m = (tax.get(r, {}).get("models") or {}).get(name)
            row[r.capitalize()] = m.get("macro_f1") if m and m.get("status") == "ok" else None
        rows.append(row)
    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "Model":
            df[c] = df[c].map(lambda v: "insufficient data" if v is None or v != v else f"{v:.3f}")
    st.markdown("##### Macro F1 by model and rank")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Same held-out sequences for every model. 'Insufficient data' means the rank cannot be scored.")


def _svm_section(e: Dict[str, Any], rank: str) -> None:
    """SVM tuning, before-vs-after and per-class detail (class rank carries the full study)."""
    svm = e.get("svm")
    opt_m = e["models"].get("SVM Optimized")
    if not svm:
        if opt_m and opt_m.get("status") == "ok":
            st.caption(
                "SVM Optimized: C and gamma were tuned inside every leave-one-out fold (nested, "
                "balanced-accuracy objective). Parameters chosen across folds: "
                + "; ".join(f"{k} ({v} folds)" for k, v in opt_m.get("selected_params_across_folds", {}).items())
                + ".")
        return

    st.markdown("---")
    st.markdown("##### SVM: baseline versus tuned")
    o = svm["optimization"]
    bp = svm["baseline_params"]
    _metric_row([
        ("Baseline C / gamma", f"{bp['C']} / {bp['gamma']}", "RBF kernel, probability estimates on."),
        ("Best C / gamma", f"{o['best_params']['C']} / {o['best_params']['gamma']}",
         "Chosen by GridSearchCV; ties go to the smaller C."),
        ("Optimisation metric", o["scoring"].replace("_", " "), o["scoring_rationale"]),
        ("Best CV score", _f(o["best_cv_score"]), "Tuning score on genuine Teleostei/Elasmobranchii "
         "validation folds; not a test score."),
    ])
    st.caption(
        f"Search: {o['n_candidates']} candidates x {o['n_cv_folds']} folds = {o['n_fits']} fits in "
        f"{o['runtime_seconds']} s. C in {o['param_grid']['C']}, gamma in {o['param_grid']['gamma']}. "
        + o["cv_strategy"])

    for key in ("protocol_a",):
        with st.container():
            pr = svm[key]
            st.caption(pr["name"] + ". " + pr["caveat"])
            ba = pr["before_after"]
            rows = []
            for label, k in [("Accuracy", "top1_accuracy"), ("Balanced accuracy", "balanced_accuracy"),
                             ("Macro precision", "macro_precision"), ("Macro recall", "macro_recall"),
                             ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1"),
                             ("ROC-AUC", "macro_roc_auc"), ("Runtime (s)", "runtime_seconds")]:
                d = ba[k]
                rows.append({"Metric": label, "Baseline SVM": d["baseline"], "Optimized SVM": d["optimized"],
                             "Change": d["absolute_change"]})
            df = pd.DataFrame(rows)
            st.dataframe(df.style.format({c: "{:+.4f}" if c == "Change" else "{:.4f}"
                                          for c in ["Baseline SVM", "Optimized SVM", "Change"]}, na_rep="N/A"),
                         hide_index=True, use_container_width=True)
            best = pr["best_model"]
            names = best["best"] if isinstance(best["best"], list) else [best["best"]]
            st.caption(("Tie for best: " if best["tied"] else "Best: ") + ", ".join(names)
                       + f" (by {best['criterion']}). Test examples: {pr['baseline']['n_test_examples']}.")
            for variant, title in (("baseline", "Baseline SVM"), ("optimized", "Optimized SVM")):
                v = pr[variant]
                left, right = st.columns([3, 2])
                with left:
                    st.plotly_chart(_heatmap(v["confusion_labels"], v["confusion_matrix"],
                                             f"{title} confusion matrix", height=360),
                                    use_container_width=True, key=f"svm_cm_{key}_{variant}")
                with right:
                    pc = pd.DataFrame(v["per_class"]).rename(columns={
                        "class": "Class", "precision": "Precision", "recall": "Recall", "f1": "F1",
                        "support": "Support", "predicted_count": "Predicted"})
                    st.dataframe(pc.style.format({c: "{:.3f}" for c in ["Precision", "Recall", "F1"]},
                                                 na_rep="not evaluable"),
                                 hide_index=True, use_container_width=True)
    pdn = svm["production_decision"]
    st.info(
        f"Production model stays **{pdn['production_model']}**"
        + ("." if not pdn["replace_with_svm_optimized"] else "; the optimized SVM beat it under the rule below.")
        + " " + pdn["rule"])


# ---------------------------------------------------------------------------
def _taxonomy_tab(res: Dict[str, Any]) -> None:
    tax = res["taxonomy"]
    st.markdown("#### Evaluation coverage by taxonomic rank")
    rows = []
    for r in RANK_ORDER:
        e = tax.get(r)
        if not e:
            continue
        s = e["support"]
        rows.append({
            "Rank": r.capitalize(),
            "Status": "Evaluated" if e["status"] == "ok" else "Insufficient data",
            "Labelled sequences": s["n_labelled_sequences"],
            "Classes": s["n_classes_total"],
            "Singleton classes": s["n_singleton_classes"],
            "Evaluable test sequences": s["n_eligible_sequences"],
            "Classes evaluated": s["n_multi_example_classes"],
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "A sequence can be held out only if another sequence with the same label remains in training. "
        "Singleton-class sequences are therefore reported here but excluded from every metric."
    )
    _model_rank_matrix(res)

    st.markdown("---")
    rank = st.radio(
        "Taxonomic rank", [r for r in RANK_ORDER if r in tax],
        horizontal=True, format_func=str.capitalize, key="eval_rank",
    )
    e = tax[rank]
    s = e["support"]

    if e["status"] != "ok":
        st.warning(
            f"**{rank.capitalize()}: Insufficient data for reliable evaluation.** {e.get('reason', '')}."
        )
        st.caption(
            f"{s['n_labelled_sequences']} labelled sequences, {s['n_classes_total']} classes, "
            f"{s['n_singleton_classes']} singleton classes, {s['n_eligible_sequences']} evaluable. "
            "No metric is computed or displayed for this rank."
        )
        if rank == "species":
            st.error(
                "Species-level generalisation cannot be evaluated: almost every species has a single "
                "reference sequence. Species output is reference matching, not a validated classifier."
            )
        return

    for w in e.get("warnings", []):
        st.info(w)

    models = e["models"]
    prod = e["production_model"]
    st.caption(
        f"Evaluation: {e['evaluation_method']}  \nFeature set used by the production model: "
        f"**{e['feature_set']}**. Production model type at this rank: **{prod}**."
    )

    st.markdown("##### Model comparison (held-out)")
    comp = []
    for name, m in models.items():
        comp.append({
            "Model": name + (" (production)" if m.get("is_production_model") else ""),
            "Macro precision": m.get("macro_precision"),
            "Macro recall": m.get("macro_recall"),
            "Macro F1": m.get("macro_f1"),
            "Weighted F1": m.get("weighted_f1"),
            "Balanced acc.": m.get("balanced_accuracy"),
            "Top-1 acc.": m.get("top1_accuracy"),
            "Macro AUC": m.get("macro_roc_auc"),
            "Test examples": m["n_eligible_test_examples"],
            "Classes evaluated": m["n_evaluated_classes"],
            "Runtime (s)": m.get("runtime_seconds"),
        })
    st.dataframe(
        pd.DataFrame(comp).style.format(
            {c: "{:.3f}" for c in ["Macro precision", "Macro recall", "Macro F1", "Weighted F1",
                                   "Balanced acc.", "Top-1 acc.", "Macro AUC"]} | {"Runtime (s)": "{:.1f}"}, na_rep="N/A"),
        hide_index=True, use_container_width=True,
    )
    st.caption("Metrics are pooled leave-one-sequence-out predictions. These describe how the MODEL "
               "performs on unseen sequences; they are not the confidence of any single prediction. "
               "Runtime is the wall-clock time to run every leave-one-out fold (training included) "
               "for that model; it is not per-prediction latency.")

    _svm_section(e, rank)

    model_name = st.radio(
        "Model detail", list(models.keys()), horizontal=True,
        index=list(models.keys()).index(prod) if prod in models else 0, key="eval_model",
    )
    m = models[model_name]
    if m["status"] != "ok":
        st.warning(f"{model_name}: {m.get('message')}. {m.get('reason', '')}")
        return

    _metric_row([
        ("Macro F1", _f(m["macro_f1"]), HELP["macro_f1"]),
        ("Weighted F1", _f(m["weighted_f1"]), HELP["weighted_f1"]),
        ("Balanced accuracy", _f(m["balanced_accuracy"]), HELP["balanced_accuracy"]),
        ("Top-1 accuracy", _f(m["top1_accuracy"]), HELP["accuracy"]),
    ])
    _metric_row([
        ("Macro precision", _f(m["macro_precision"]), HELP["macro_precision"]),
        ("Macro recall", _f(m["macro_recall"]), HELP["macro_recall"]),
        ("Test examples", str(m["n_eligible_test_examples"]),
         "Held-out unique sequences actually scored at this rank."),
        ("Classes evaluated", str(m["n_evaluated_classes"]),
         "Classes with at least one held-out example."),
    ])
    for w in m.get("warnings", []):
        st.warning(w)

    left, right = st.columns([3, 2])
    with left:
        cm = m["confusion_matrix"]
        st.plotly_chart(_heatmap(cm["labels"], cm["matrix"], f"{rank.capitalize()} confusion matrix: {model_name}"),
                        use_container_width=True)
    with right:
        st.markdown("##### Per-class metrics")
        pc = pd.DataFrame(m["per_class"]).rename(columns={
            "class": "Class", "precision": "Precision", "recall": "Recall", "f1": "F1",
            "support": "Support", "predicted_count": "Predicted"})
        st.dataframe(
            pc.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}, na_rep="N/A"),
            hide_index=True, use_container_width=True, height=420,
        )

    st.markdown("##### Accepted accuracy versus coverage")
    cov = m.get("accuracy_vs_coverage")
    if cov:
        cdf = pd.DataFrame(cov)
        fig = go.Figure(go.Scatter(x=cdf["coverage"], y=cdf["accuracy_of_accepted"], mode="lines+markers",
                                   line=dict(color="#14b8a6")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Coverage (fraction of predictions accepted)",
                          yaxis_title="Accuracy of accepted", paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    st.caption(m.get("confidence_note", ""))

    nn = e.get("nearest_reference_baseline")
    if nn:
        st.markdown("##### Nearest-reference label accuracy versus distance")
        ndf = pd.DataFrame(nn["accuracy_vs_distance"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ndf["distance_threshold"], y=ndf["accuracy_of_accepted"], mode="lines+markers",
                                 name="Accuracy of accepted", line=dict(color="#14b8a6")))
        fig.add_trace(go.Scatter(x=ndf["distance_threshold"], y=ndf["coverage"], mode="lines+markers",
                                 name="Coverage", line=dict(color="#f59e0b")))
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title="Accept nearest-reference label if distance <= t",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(nn["description"] + f" Overall accuracy over all {s['n_eligible_sequences']} "
                   f"evaluable sequences: {_f(nn['accuracy_all_eligible'], pct=True)}.")


# ---------------------------------------------------------------------------
def _unknown_tab(res: Dict[str, Any]) -> None:
    u = res["unknown_detection"]
    t = u["test"]
    thr = u["production_thresholds"]

    st.markdown("#### Out-of-reference detection (separate from taxonomy accuracy)")
    st.caption(
        f"Positive class = **out-of-reference**. Production thresholds: close-match <= {thr['close_match']}, "
        f"Distance Veto > {thr['veto']}. Evaluated on the held-out TEST partition "
        f"({t['n_known_perturbed']} altered known sequences, {t['n_unknown']} unknown sequences)."
    )
    st.warning(u["threshold_independence_warning"])
    st.info("What is evaluated: " + u["decision_rule_evaluated"])

    pm = t["production_rule_out_of_reference"]
    st.markdown("##### Production rule: flagged `out_of_reference`")
    _metric_row([
        ("Precision", _f(pm["precision"]), "Of sequences flagged out-of-reference, the fraction that truly are unknown."),
        ("Recall / sensitivity", _f(pm["recall_sensitivity"]), "Of truly unknown sequences, the fraction flagged."),
        ("F1", _f(pm["f1"]), "Harmonic mean of precision and recall for the out-of-reference class."),
        ("Specificity", _f(pm["specificity"]), "Of known sequences, the fraction correctly NOT flagged."),
    ])
    _metric_row([
        ("False-positive rate", _f(pm["false_positive_rate"]), "Known sequences wrongly flagged out-of-reference."),
        ("False-negative rate", _f(pm["false_negative_rate"]), "Unknown sequences wrongly kept (missed)."),
        ("Coverage", _f(pm["coverage_accepted_rate"]), "Fraction of queries given an accepted identification (known_reference_like)."),
        ("Abstention rate", _f(pm["abstention_rate"]), "Fraction not accepted: low_confidence or out_of_reference."),
    ])
    st.caption(f"Known n={pm['n_known']}, unknown n={pm['n_unknown']}. "
               f"TP={pm['tp']}, FP={pm['fp']}, TN={pm['tn']}, FN={pm['fn']}.")

    cm = pm["confusion_matrix"]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_heatmap(["known", "out_of_reference"], cm["matrix"], "Confusion matrix (rows = truth)",
                                 height=320), use_container_width=True)
    with c2:
        ab = t["production_rule_reject_or_abstain"]
        st.markdown("##### Stricter reading: reject OR abstain")
        st.caption("Counts `low_confidence` as a flag as well as `out_of_reference`.")
        st.dataframe(pd.DataFrame([{
            "Precision": ab["precision"], "Recall": ab["recall_sensitivity"], "F1": ab["f1"],
            "Specificity": ab["specificity"], "FPR": ab["false_positive_rate"], "FNR": ab["false_negative_rate"],
        }]).style.format("{:.3f}", na_rep="N/A"), hide_index=True, use_container_width=True)

    st.markdown("##### Categories evaluated separately")
    cons = u["construction"]
    rows = []
    for key in ("heldout_genus", "external_nonfish"):
        m = t["per_unknown_category"][key]
        rows.append({
            "Category": cons[key]["label"], "True label": cons[key]["true_label"],
            "n": m["n_category_examples"], "Median distance": m["median_distance"],
            "Flagged out-of-reference": m["category_detected_as_out_of_reference"],
            "Wrongly accepted as known": m["category_accepted_as_known"],
        })
    for key in ("known_perturbed", "known_exact"):
        k = t["known_categories"][key]
        if k.get("n"):
            rows.append({
                "Category": cons[key]["label"], "True label": cons[key]["true_label"], "n": k["n"],
                "Median distance": k["median_distance"],
                "Flagged out-of-reference": k["flagged_out_of_reference_rate"],
                "Wrongly accepted as known": None,
            })
    st.dataframe(
        pd.DataFrame(rows).style.format({"Median distance": "{:.3f}", "Flagged out-of-reference": "{:.1%}",
                                         "Wrongly accepted as known": "{:.1%}"}, na_rep="n/a"),
        hide_index=True, use_container_width=True,
    )
    st.caption("Per-category precision/recall use that category as positives and the altered-known "
               "sequences as negatives; exact reference matches are excluded from pooled metrics because "
               "they are trivially accepted.")

    lv = t.get("perturbation_levels") or {}
    if lv:
        st.markdown("##### Robustness to substitution errors (altered known sequences)")
        lrows = []
        for variant, v in lv.items():
            lrows.append({
                "Substitutions": variant.split("_")[0],
                "n": v["n"],
                "Median distance": v["median_distance"],
                "Max distance": v["max_distance"],
                "Wrongly flagged out-of-reference": v["false_positive_rate"],
                "Still accepted as known": v["accepted_rate"],
                "Abstained (low confidence)": v["abstention_rate"],
            })
        st.dataframe(
            pd.DataFrame(lrows).style.format({
                "Median distance": "{:.3f}", "Max distance": "{:.3f}",
                "Wrongly flagged out-of-reference": "{:.1%}", "Still accepted as known": "{:.1%}",
                "Abstained (low confidence)": "{:.1%}"}),
            hide_index=True, use_container_width=True,
        )
        st.caption("Each level is a fixed number of random base substitutions in a ~100 bp sequence. "
                   "These are robustness checks and are reported separately from classification accuracy. "
                   "Insertions, deletions and real read errors are not tested.")

    st.markdown("##### Threshold diagnostics")
    cal = t.get("calibrated_threshold")
    if cal:
        ct = cal["test_metrics"]
        st.info(
            f"Threshold chosen on the calibration partition only (Youden's J): **{cal['threshold']:.2f}** "
            f"(calibration J={cal['calibration_youden_j']:.2f}). On the test partition: "
            f"recall {_f(ct['recall_sensitivity'])}, specificity {_f(ct['specificity'])}, "
            f"FPR {_f(ct['false_positive_rate'])}, FNR {_f(ct['false_negative_rate'])}. "
            "Diagnostic only - the production Distance Veto is unchanged."
        )
    else:
        st.info("Threshold calibration unavailable: calibration partition lacks both known and unknown examples.")

    sw = pd.DataFrame(t["threshold_sweep_descriptive"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sw["threshold"], y=sw["false_positive_rate"], name="False-positive rate",
                             line=dict(color="#f59e0b")))
    fig.add_trace(go.Scatter(x=sw["threshold"], y=sw["false_negative_rate"], name="False-negative rate",
                             line=dict(color="#ef4444")))
    fig.add_trace(go.Scatter(x=sw["threshold"], y=sw["coverage"], name="Coverage", line=dict(color="#14b8a6")))
    fig.add_vline(x=thr["veto"], line_dash="dash", line_color="#9ca3af", annotation_text="production veto")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Veto threshold (flag if distance > t)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Descriptive sweep on the test partition. Do not choose a threshold from this chart and "
               "then report its test numbers.")

    au = t["distance_auroc"]
    if au["status"] == "ok":
        st.metric("AUROC of nearest-reference distance (unknown vs altered-known)", _f(au["auroc"]),
                  help="Probability that a random unknown sequence has a larger distance than a random "
                       "known one. 0.5 = no separation, 1.0 = perfect. Uses the distance score only.")
        st.caption(f"unknown n={au['n_unknown']}, known n={au['n_known']}.")
    else:
        st.caption(f"AUROC unavailable: {au['reason']}.")


# ---------------------------------------------------------------------------
def _methods_tab(res: Dict[str, Any]) -> None:
    meta = res["meta"]
    ref = meta["reference_library"]
    st.markdown("#### Data and leakage control")
    st.markdown(
        f"- Raw survey rows: **{meta.get('raw_dataset_rows') or 'N/A'}**\n"
        f"- Unique normalised reference sequences used for evaluation: **{ref['unique_sequences']}** "
        f"(duplicate rows removed here: {ref['duplicate_rows_removed']}, label conflicts: {ref['label_conflicts']})\n"
        f"- Random seed: {meta['seed']}. Generated: {meta['generated_at_utc']}\n"
        "- Splits are made on unique sequences, never on duplicated rows. Identical sequences cannot "
        "occur in both a training and a test partition (asserted in code)."
    )
    st.markdown("#### What each unknown-detection category is, and is not")
    for key, c in res["unknown_detection"]["construction"].items():
        with st.container():
            st.markdown(f"**{c['label']}**")
            st.markdown(
                f"- Construction: {c['construction']}\n"
                f"- True label: {c['true_label']}\n"
                f"- Establishes: {c['establishes']}\n"
                f"- Does not establish: {c['does_not_establish']}"
            )
    st.markdown("#### Metric definitions")
    for k, v in res["unknown_detection"]["definitions"].items():
        st.markdown(f"- **{k}**: {v}")
    st.markdown("#### Known limitations")
    st.markdown(
        "- ~101 unique sequences; most species have a single sequence, so **species-level "
        "generalisation is not evaluable**.\n"
        "- Higher ranks are evaluated on very small per-class support; per-class values are unstable.\n"
        "- Thresholds were originally selected using these same sequences; see the warning on the "
        "unknown-detection tab.\n"
        "- Held-out genera often have a close relative in the library, so some are legitimately "
        "accepted at family level.\n"
        "- Non-fish negatives are windows of full-length NCBI records, not real amplicon reads.\n"
        "- No external benchmark (e.g. BLAST) and no real eDNA sample has been evaluated."
    )
    st.markdown("#### Planned extensions (not implemented)")
    st.markdown(
        "- More unique, independently sourced sequences per species.\n"
        "- External validation on unseen taxa from a different source.\n"
        "- Benchmark against trusted sequence search (BLAST / vsearch).\n"
        "- Better calibration of distance and confidence thresholds on larger data.\n"
        "- FASTQ-to-ASV preprocessing front end."
    )


# ---------------------------------------------------------------------------
def render_evaluation_page() -> None:
    st.markdown("### Model Evaluation")
    st.caption(
        "Held-out performance of the models and of the out-of-reference detector. "
        "This is population-level evaluation. It is NOT the confidence of an individual prediction, "
        "and nearest-reference similarity, classifier probability and overall identification confidence "
        "are different quantities."
    )

    mtime = os.path.getmtime(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else 0.0
    res = _load_results(RESULTS_PATH, mtime)
    if res is None:
        st.error("Evaluation results are not available.")
        st.markdown("Generate them (this retrains throw-away models inside cross-validation folds; "
                    "production models are not touched):")
        st.code("PYTHONPATH=. python src/evaluation/run_model_evaluation.py", language="bash")
        return

    _headline_cards(res)

    tab_tax, tab_unk, tab_methods = st.tabs(
        ["Taxonomy classification", "Unknown-DNA detection", "Methodology & limitations"]
    )
    with tab_tax:
        _taxonomy_tab(res)
    with tab_unk:
        _unknown_tab(res)
    with tab_methods:
        _methods_tab(res)
