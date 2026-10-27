"""
src/evaluation/run_model_evaluation.py
Leakage-safe evaluation of the BioScan taxonomy classifiers and the out-of-reference
(Distance Veto) detector. Writes reports/evaluation/evaluation_results.json, which is
the ONLY source for the dashboard's Model Evaluation page.

Run:  PYTHONPATH=. python src/evaluation/run_model_evaluation.py

Responsibilities are kept separate:
  * This script trains throw-away models inside cross-validation folds. It never
    touches models/*.joblib (production artefacts) and never writes training code
    paths used by the application.
  * It only READS production thresholds from src/prediction/evidence.py.
"""

import datetime as dt
import json
import os
import random
import sys
import time
import warnings
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.evaluation.metrics import (  # noqa: E402
    MIN_EVAL_CLASSES,
    MIN_EVAL_SAMPLES,
    accuracy_coverage_curve,
    assess_sufficiency,
    calibrate_threshold_youden,
    detection_metrics,
    distance_auroc,
    distance_coverage_curve,
    taxonomy_metrics,
    threshold_sweep,
)
from src.evaluation.splits import (  # noqa: E402
    assert_no_sequence_overlap,
    class_support_summary,
    deduplicate_by_sequence,
    eligible_for_holdout,
    normalize_sequence,
    split_groups,
)
from src.models import svm_model  # noqa: E402
from src.models.hierarchical_taxonomy import FEATURE_SETS, get_model  # noqa: E402
from src.prediction.evidence import (  # noqa: E402
    THRESHOLD_CLOSE_SIMILARITY,
    THRESHOLD_OUT_OF_REFERENCE,
    synthesize_evidence_and_status,
)
from src.similarity.reference_search import ReferenceSearcher  # noqa: E402

SEED = 42
RANKS = ["class", "order", "family", "genus", "species"]
MODEL_NAMES = ["KNN", "Random Forest", "XGBoost", "SVM Baseline"]
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "reports", "evaluation")
FASTA = os.path.join(BASE_DIR, "data", "external", "negative_controls_12S.fasta")
NA_SPECIES = "Not available in reference data"

PERTURB_SUBSTITUTIONS = [1, 2, 5]          # absolute base substitutions per query
AMPLICON_WINDOW = 106
WINDOW_OFFSETS = (0, 120, 240)
SWEEP_THRESHOLDS = [round(x, 3) for x in np.arange(0.02, 0.41, 0.01)]
DISTANCE_COVERAGE_THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.50]

MODEL_TYPE_TO_NAME = {
    "KNeighborsClassifier": "KNN",
    "RandomForestClassifier": "Random Forest",
    "XGBClassifier": "XGBoost",
    "SVC": "SVM Baseline",
}


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Part A - taxonomy classifiers
# ---------------------------------------------------------------------------
def _loo_predictions(X: np.ndarray, y: np.ndarray, eligible: np.ndarray, model_name: str):
    """
    Leave-one-sequence-out. For each ELIGIBLE sequence i: train on all other labelled
    sequences (scaler fitted on that training fold only), predict i.
    Returns (y_true, y_pred, confidence_or_None, y_probs_matrix).
    """
    n = len(y)
    classes = sorted(set(y.tolist()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    eligible_indices = [i for i in range(n) if eligible[i]]
    
    # Vectorized fast path for 1-NN when evaluating many samples
    if model_name == "KNN" and len(eligible_indices) > 150:
        scaler = StandardScaler().fit(X)
        X_sc = scaler.transform(X)
        le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
        knn = KNeighborsClassifier(n_neighbors=2, metric="manhattan")
        knn.fit(X_sc, y_enc)
        _, indices = knn.kneighbors(X_sc, n_neighbors=2)
        
        y_true = [str(y[i]) for i in eligible_indices]
        y_pred = [str(le.inverse_transform([y_enc[indices[i, 1]]])[0]) for i in eligible_indices]
        conf = [1.0] * len(eligible_indices)
        
        y_probs = np.zeros((len(eligible_indices), len(classes)), dtype=float)
        for row_idx, pred_label in enumerate(y_pred):
            y_probs[row_idx, class_to_idx[pred_label]] = 1.0
            
        return y_true, y_pred, conf, y_probs

    def _eval_single_fold(i: int):
        tr = np.array([j for j in range(n) if j != i])
        scaler = StandardScaler().fit(X[tr])
        le = LabelEncoder().fit(y[tr])
        clf = get_model(model_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(scaler.transform(X[tr]), le.transform(y[tr]))
        xi = scaler.transform(X[[i]])
        
        prob_vec = np.zeros(len(classes), dtype=float)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(xi)[0]
            # SVC: the label is predict(); Platt probabilities are only used for confidence / AUC.
            k = int(clf.predict(xi)[0]) if isinstance(clf, SVC) else int(np.argmax(probs))
            conf_val = float(probs[k])
            pred_cls = str(le.inverse_transform([k])[0])
            for le_idx, p_val in enumerate(probs):
                cls_name = str(le.inverse_transform([le_idx])[0])
                if cls_name in class_to_idx:
                    prob_vec[class_to_idx[cls_name]] = float(p_val)
        else:
            k = int(clf.predict(xi)[0])
            conf_val = 1.0
            pred_cls = str(le.inverse_transform([k])[0])
            prob_vec[class_to_idx[pred_cls]] = 1.0
            
        return str(y[i]), pred_cls, conf_val, prob_vec

    # Parallelize fold evaluations across available cores
    n_jobs = -1 if len(eligible_indices) > 50 else 1
    batch_size = 32 if len(eligible_indices) > 50 else "auto"
    results = joblib.Parallel(n_jobs=n_jobs, batch_size=batch_size)(
        joblib.delayed(_eval_single_fold)(i) for i in eligible_indices
    )
    
    y_true = [r[0] for r in results]
    y_pred = [r[1] for r in results]
    conf = [r[2] for r in results]
    y_probs = np.vstack([r[3] for r in results])
    
    has_meaningful_proba = (model_name != "KNN")
    return y_true, y_pred, (conf if has_meaningful_proba else None), y_probs


def _nearest_reference_by_rank(ref: pd.DataFrame, rank: str, labelled_ids: List[str],
                               searcher: ReferenceSearcher, eligible_ids: List[str],
                               lvl_df: Optional[pd.DataFrame] = None):
    """Leave-one-out nearest-reference label at `rank` and its distance."""
    ref_lab = ref.set_index("sequence_id")[rank]
    ref_seqs = ref.set_index("sequence_id")["sequence_normalized"]

    if lvl_df is not None:
        lvl_lab = lvl_df.set_index("sequence_id")[rank]
        lvl_seqs = lvl_df.set_index("sequence_id")["sequence_normalized"] if "sequence_normalized" in lvl_df.columns else lvl_df.set_index("sequence_id")["dnasequence"]
    else:
        lvl_lab = ref_lab
        lvl_seqs = ref_seqs

    dists, correct = [], []
    for sid in eligible_ids:
        seq = lvl_seqs[sid]
        true_label = lvl_lab[sid]
        exclude = sid if sid in ref_seqs.index else None
        hits = searcher.search(seq, top_k=1, exclude_id=exclude)
        if hits:
            hit = hits[0]
            pred = hit.get(rank)
            ok = pred is not None and pred != NA_SPECIES and pred == true_label
            dists.append(float(hit["distance"]))
            correct.append(bool(ok))
    return dists, correct


# ---------------------------------------------------------------------------
# Leakage-safe (parent-group) evaluation and SVM tuning helpers
# ---------------------------------------------------------------------------
SVM_METRIC_KEYS = ["top1_accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                   "macro_f1", "weighted_f1", "macro_roc_auc", "runtime_seconds"]


def _fit_predict_one(model_name: str, X: np.ndarray, y: np.ndarray, tr: np.ndarray,
                     te: int, classes: List[str]):
    """Fit one throw-away model on rows `tr` (scaler fitted on them only), predict row `te`."""
    scaler = StandardScaler().fit(X[tr])
    le = LabelEncoder().fit(y[tr])
    clf = get_model(model_name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(scaler.transform(X[tr]), le.transform(y[tr]))
    xi = scaler.transform(X[[te]])
    vec = np.zeros(len(classes), dtype=float)
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(xi)[0]
        k = int(clf.predict(xi)[0]) if isinstance(clf, SVC) else int(np.argmax(probs))
        for j, pv in enumerate(probs):
            vec[classes.index(str(le.inverse_transform([j])[0]))] = float(pv)
    else:
        k = int(clf.predict(xi)[0])
    pred = str(le.inverse_transform([k])[0])
    if not hasattr(clf, "predict_proba"):
        vec[classes.index(pred)] = 1.0
    return pred, vec


def _parallel(fn, items):
    nj = -1 if len(items) > 20 else 1
    return joblib.Parallel(n_jobs=nj, batch_size="auto")(joblib.delayed(fn)(i) for i in items)


def _optimized_loo(X, y, groups, is_synth, test_idx, classes, train_rule):
    """
    Nested evaluation of the tuned SVM. For every held-out row the C/gamma search runs on that
    fold's TRAINING rows only (train_rule(i) -> training indices), then the tuned model predicts i.
    """
    def one(i):
        tr = train_rule(i)
        lab, pr, params = svm_model.fit_predict_tuned(X, y, groups, is_synth, tr, [i], classes)
        return str(y[i]), lab[0], pr[0], params
    res = _parallel(one, list(test_idx))
    return ([r[0] for r in res], [r[1] for r in res], np.vstack([r[2] for r in res]),
            [r[3] for r in res])


def _canonical_report(y_true, y_pred, order=None):
    """Confusion matrix and per-class metrics in a fixed class order (all four classes shown)."""
    order = order or svm_model.CLASS_ORDER
    cm = confusion_matrix(y_true, y_pred, labels=order)
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=order, zero_division=0)
    pred_n = cm.sum(axis=0)
    rows = []
    for k, c in enumerate(order):
        has = int(sup[k]) > 0
        rows.append({
            "class": c,
            "precision": float(p[k]) if (has or pred_n[k] > 0) else None,
            "recall": float(r[k]) if has else None,
            "f1": float(f1[k]) if has else None,
            "support": int(sup[k]),
            "predicted_count": int(pred_n[k]),
        })
    return {"labels": order, "matrix": cm.tolist(), "per_class": rows}


def _delta(base, opt):
    if base is None or opt is None:
        return {"baseline": base, "optimized": opt, "absolute_change": None, "percent_change": None}
    d = opt - base
    return {"baseline": base, "optimized": opt, "absolute_change": d,
            "percent_change": (d / base * 100.0) if base else None}


def _best_model(table: Dict[str, Dict[str, Any]]):
    """Rank by balanced accuracy, then macro F1, then macro AUC. Ties are reported as ties."""
    def key(n):
        m = table[n]
        return (m.get("balanced_accuracy") or 0, m.get("macro_f1") or 0, m.get("macro_roc_auc") or 0)
    ranked = sorted(table, key=key, reverse=True)
    top = [n for n in ranked if key(n) == key(ranked[0])]
    return {"best": top[0] if len(top) == 1 else top, "tied": len(top) > 1,
            "criterion": "balanced accuracy, then macro F1, then macro ROC-AUC",
            "ranking": ranked}


def _svm_study(X, labels, lvl, models, preds_a, prod_name) -> Dict[str, Any]:
    """
    Class-rank SVM study under the project's leave-one-sequence-out protocol (protocol A).
    C and gamma are tuned inside every fold on that fold's training rows only (nested).
    """
    y = np.asarray(labels, dtype=object).astype(str)
    classes = sorted(set(y.tolist()))
    is_synth = lvl["is_synthetic"].astype(bool).values
    groups = svm_model.assign_parent_groups(lvl)
    n = len(y)
    all_rows = np.arange(n)

    # --- tuning on the full training set (what the saved artefact uses) ----------------
    _log("    [svm] grid search on the full training set (leakage-safe splitter)")
    tuning = svm_model.tune_svm(X, y, groups, is_synth, n_jobs=-1)
    best_params = tuning["best_params"]
    _log(f"    [svm] best C={best_params['C']} gamma={best_params['gamma']} "
         f"cv balanced acc={tuning['best_cv_score']:.4f} ({tuning['runtime_seconds']}s)")

    # --- protocol A: nested tuned SVM over all 259 rows --------------------------------
    t0 = time.time()
    _log("    [svm] protocol A: nested tuned SVM, leave-one-sequence-out (all 259)")
    yt, yp, pr, chosen_a = _optimized_loo(
        X, y, groups, is_synth, list(all_rows), classes,
        lambda i: np.array([j for j in range(n) if j != i]))
    m = taxonomy_metrics(yt, yp, pr)
    m["is_production_model"] = False
    m["runtime_seconds"] = round(time.time() - t0, 1)
    real = ~is_synth
    m["real_only_evaluation"] = taxonomy_metrics(
        [a for a, r in zip(yt, real) if r], [b for b, r in zip(yp, real) if r], pr[real])
    conf = [float(pr[k, classes.index(yp[k])]) for k in range(len(yp))]
    m["accuracy_vs_coverage"] = accuracy_coverage_curve(conf, [a == b for a, b in zip(yt, yp)])
    m["confidence_note"] = ("SVM confidence = Platt-scaled top-1 probability. Not a calibrated "
                            "probability of being correct. Tuned inside every fold (nested).")
    m["selected_params_across_folds"] = _count_params(chosen_a)
    models["SVM Optimized"] = m
    preds_a["SVM Optimized"] = (yt, yp)

    def block(table, preds):
        out = {}
        for key, name in (("baseline", "SVM Baseline"), ("optimized", "SVM Optimized")):
            t = table[name]
            out[key] = {k: t.get(k) for k in SVM_METRIC_KEYS}
            out[key].update({
                "n_test_examples": t["n_eligible_test_examples"],
                "n_evaluated_classes": t["n_evaluated_classes"],
                "confusion_matrix": _canonical_report(*preds[name])["matrix"],
                "confusion_labels": svm_model.CLASS_ORDER,
                "per_class": _canonical_report(*preds[name])["per_class"],
                "params": (svm_model.BASELINE_PARAMS if key == "baseline" else best_params),
            })
        out["before_after"] = {k: _delta(out["baseline"][k], out["optimized"][k])
                               for k in SVM_METRIC_KEYS}
        return out

    table_a = {k: v for k, v in models.items() if v.get("status") == "ok"}
    a_block = block(table_a, preds_a)

    prod_a = table_a.get(prod_name)
    opt_a = table_a.get("SVM Optimized")
    replace = bool(prod_a and opt_a
                   and (opt_a["balanced_accuracy"] or 0) > (prod_a["balanced_accuracy"] or 0)
                   and (opt_a["macro_f1"] or 0) > (prod_a["macro_f1"] or 0))
    return {
        "feature_set": "k=3 (3-mer frequencies, 64 features), standardised inside a Pipeline",
        "class_order": svm_model.CLASS_ORDER,
        "baseline_params": svm_model.BASELINE_PARAMS,
        "optimization": {
            "method": "GridSearchCV (refit=False; best chosen by mean score, ties to smaller C)",
            "scoring": svm_model.SCORING,
            "scoring_rationale": ("Balanced accuracy is the mean per-class recall, so the 82 "
                                  "Teleostei cannot dominate the objective the way plain accuracy would."),
            "param_grid": {"C": svm_model.C_GRID, "gamma": svm_model.GAMMA_GRID, "kernel": ["rbf"]},
            "cv_strategy": ("Stratified 5-fold over genuine Teleostei and Elasmobranchii sequences. "
                            "Validation uses genuine sequences only; sequences sharing a parent group "
                            "with a validation sequence are removed from that fold's training data. "
                            "Chondrostei and Petromyzonti (one independent sequence each) are always "
                            "in training and never validated."),
            "n_cv_folds": tuning["n_cv_folds"], "n_candidates": tuning["n_candidates"],
            "n_fits": tuning["n_fits"], "random_state": svm_model.SEED,
            "best_params": best_params, "best_cv_score": tuning["best_cv_score"],
            "best_cv_std": tuning["best_cv_std"], "runtime_seconds": tuning["runtime_seconds"],
            "search_results": tuning["search_results"],
            "nested_note": ("In both evaluation protocols the search is re-run inside every outer "
                            "fold on that fold's training rows only, so no test sequence influences "
                            "C or gamma."),
        },
        "protocol_a": {
            "name": "Leave-one-sequence-out (LOOCV) over all 259 training sequences",
            "caveat": ("Chondrostei and Petromyzonti each have one reference sequence plus "
                       "augmented variants of it, so their scores reflect recognition of those variants."),
            **a_block, "best_model": _best_model(table_a),
        },
        "production_decision": {
            "production_model": prod_name,
            "replace_with_svm_optimized": replace,
            "rule": ("Replace only if the optimized SVM beats the production model on BOTH balanced "
                     "accuracy and macro F1 under the evaluation protocol."),
        },
    }


def _svm_optimized_other_rank(X, labels, elig, models) -> None:
    """
    Order / family: add 'SVM Optimized' with the same nested tuning. These ranks have no
    synthetic rows, so every sequence is its own parent group.
    """
    y = np.asarray(labels, dtype=object).astype(str)
    classes = sorted(set(y.tolist()))
    n = len(y)
    groups = np.array([str(i) for i in range(n)])
    synth = np.zeros(n, dtype=bool)
    idx = [i for i in range(n) if elig[i]]
    t0 = time.time()
    yt, yp, pr, chosen = _optimized_loo(
        X, y, groups, synth, idx, classes,
        lambda i: np.array([j for j in range(n) if j != i]))
    m = taxonomy_metrics(yt, yp, pr)
    m["is_production_model"] = False
    m["runtime_seconds"] = round(time.time() - t0, 1)
    conf = [float(pr[k, classes.index(yp[k])]) for k in range(len(yp))]
    m["accuracy_vs_coverage"] = accuracy_coverage_curve(conf, [a == b for a, b in zip(yt, yp)])
    m["confidence_note"] = ("SVM confidence = Platt-scaled top-1 probability. Not a calibrated "
                            "probability of being correct. C and gamma tuned inside every fold (nested).")
    m["selected_params_across_folds"] = _count_params(chosen)
    models["SVM Optimized"] = m


def _count_params(chosen):
    out: Dict[str, int] = {}
    for p in chosen:
        k = f"C={p['C']}, gamma={p['gamma']}"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))



def evaluate_taxonomy(ref: pd.DataFrame, searcher: ReferenceSearcher) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    train_barcodes_path = os.path.join(PROCESSED, "training_barcodes.csv")
    has_augmented_class = os.path.exists(train_barcodes_path)

    for rank in RANKS:
        _log(f"[taxonomy] rank={rank}")
        art = joblib.load(os.path.join(MODELS_DIR, f"taxonomy_{rank}_model.joblib"))
        feature_set = art["feature_set"]
        cols = list(art["feature_columns"])
        production_model = MODEL_TYPE_TO_NAME.get(type(art["model"]).__name__, type(art["model"]).__name__)

        if rank == "class" and has_augmented_class:
            train_df = pd.read_csv(train_barcodes_path)
            lvl = train_df[train_df[rank].notna()].reset_index(drop=True)
            feat_clean = feature_set.lower().replace(" ", "_").replace("=", "")
            feat_file = f"training_features_{feat_clean}.csv"
            feat_path = os.path.join(PROCESSED, feat_file)
            if not os.path.exists(feat_path):
                feat_path = os.path.join(PROCESSED, "training_features_k3.csv")
            feat = pd.read_csv(feat_path)
            n_gen = int((~train_df.get('is_synthetic', pd.Series(False))).sum())
            n_syn = int(train_df.get('is_synthetic', pd.Series(False)).sum())
            eval_desc = (
                f"Leave-one-sequence-out over {len(train_df)} unique sequences ({n_gen} genuine + {n_syn} synthetic minority augmentation). "
                "All 4 classes evaluable without singletons. Scaler fitted on each training fold. Deterministic."
            )
        else:
            lvl = ref[ref[rank].notna()].reset_index(drop=True)
            feat = pd.read_csv(FEATURE_SETS[feature_set])
            eval_desc = (
                "Leave-one-sequence-out over unique sequences. Held-out sequence is "
                "evaluated only if another sequence with the same label remains in "
                "training. Scaler fitted on each training fold. Deterministic (no RNG split)."
            )

        labels = lvl[rank].astype(str).values
        support = class_support_summary(labels)
        elig = eligible_for_holdout(labels)
        suff = assess_sufficiency(int(elig.sum()), support["n_multi_example_classes"])

        entry: Dict[str, Any] = {
            "rank": rank,
            "feature_set": feature_set,
            "production_model": production_model,
            "support": support,
            "min_requirements": {"min_samples": MIN_EVAL_SAMPLES, "min_classes": MIN_EVAL_CLASSES},
            "evaluation_method": eval_desc,
            "status": "ok" if suff["sufficient"] else "insufficient",
        }
        if not suff["sufficient"]:
            entry["message"] = "Insufficient data for reliable evaluation"
            entry["reason"] = "; ".join(suff["reasons"])
            entry["models"] = {}
            out[rank] = entry
            _log(f"    insufficient: {entry['reason']}")
            continue

        merged = lvl[["sequence_id"]].merge(feat, on="sequence_id", how="left")
        if merged[cols].isna().any().any():
            raise RuntimeError(f"Missing features for some sequences at rank {rank}")
        X = merged[cols].values.astype(float)

        ref_lvl = ref[ref[rank].notna()].reset_index(drop=True)
        ref_elig = eligible_for_holdout(ref_lvl[rank].astype(str).values)
        ref_eligible_ids = ref_lvl.loc[ref_elig, "sequence_id"].tolist()
        nn_d, nn_ok = _nearest_reference_by_rank(ref, rank, ref_lvl["sequence_id"].tolist(),
                                                 searcher, ref_eligible_ids)
        entry["nearest_reference_baseline"] = {
            "description": (
                "Leave-one-out nearest-reference search (normalised edit distance), "
                "predicting the label of the closest OTHER reference. Not a trained model."
            ),
            "accuracy_all_eligible": float(np.mean(nn_ok)),
            "accuracy_vs_distance": distance_coverage_curve(nn_d, nn_ok, DISTANCE_COVERAGE_THRESHOLDS),
        }

        # Check provenance for real-only separation
        is_synthetic_col = lvl.loc[elig, "is_synthetic"].values if "is_synthetic" in lvl.columns else None
        real_mask = (~is_synthetic_col) if is_synthetic_col is not None else None

        models: Dict[str, Any] = {}
        preds_a: Dict[str, Any] = {}
        for name in MODEL_NAMES:
            t0 = time.time()
            y_true, y_pred, conf, y_probs = _loo_predictions(X, labels, elig, name)
            preds_a[name] = (y_true, y_pred)
            m = taxonomy_metrics(y_true, y_pred, y_probs)
            m["is_production_model"] = (name == production_model)
            m["runtime_seconds"] = round(time.time() - t0, 1)

            # Compute real-only held-out evaluation on genuine biological sequences
            if real_mask is not None and real_mask.sum() > 0:
                y_true_real = [yt for yt, rm in zip(y_true, real_mask) if rm]
                y_pred_real = [yp for yp, rm in zip(y_pred, real_mask) if rm]
                y_probs_real = y_probs[real_mask]
                real_metrics = taxonomy_metrics(y_true_real, y_pred_real, y_probs_real)
                m["real_only_evaluation"] = real_metrics

            if conf is not None:
                correct = [a == b for a, b in zip(y_true, y_pred)]
                m["accuracy_vs_coverage"] = accuracy_coverage_curve(conf, correct)
                m["confidence_note"] = (
                    "Confidence = the classifier's own top-1 class probability. "
                    "It is not a calibrated probability of being correct."
                )
            else:
                m["accuracy_vs_coverage"] = None
                m["confidence_note"] = (
                    "Production KNN uses k=1, so its class probabilities are 0/1 and carry "
                    "no confidence information."
                )
            models[name] = m
            _log(f"    {name:14} n={m['n_eligible_test_examples']} "
                 f"macroF1={m.get('macro_f1')} acc={m.get('top1_accuracy')} "
                 f"roc_auc={m.get('macro_roc_auc')} ({m['runtime_seconds']}s)")
        if rank == "class" and has_augmented_class and "is_synthetic" in lvl.columns:
            entry["svm"] = _svm_study(X, labels, lvl, models, preds_a, production_model)
        elif rank in ("order", "family"):
            _log(f"    [svm] nested tuned SVM at {rank}")
            _svm_optimized_other_rank(X, labels, elig, models)
        entry["models"] = models

        warn = []
        if support["n_singleton_classes"] > 0:
            warn.append(
                f"{support['n_singleton_classes']} of {support['n_classes_total']} classes have "
                f"a single sequence and cannot be held out; {support['n_unevaluable_singleton_sequences']} "
                "sequence(s) are therefore excluded from the test set (reported, not hidden)."
            )
        entry["warnings"] = warn
        out[rank] = entry
    return out


# ---------------------------------------------------------------------------
# Part B - unknown (out-of-reference) detection
# ---------------------------------------------------------------------------
def _searcher_for(library: pd.DataFrame) -> ReferenceSearcher:
    s = ReferenceSearcher()
    s.ref_df = library.reset_index(drop=True)
    s.sequences = s.ref_df["sequence_normalized"].tolist()
    s.seq_ids = s.ref_df["sequence_id"].tolist()
    s.n_references = len(s.ref_df)
    return s


def _production_status(query: str, searcher: ReferenceSearcher) -> Dict[str, Any]:
    """
    Run the production evidence rule on the nearest-reference distance.
    Class/order concordance is set to True and classifier probabilities to a neutral
    value so only the DISTANCE rules (Distance Veto and close-match) are exercised.
    """
    hits = searcher.search(query, top_k=2)
    n1 = hits[0]
    d1 = float(n1["distance"])
    d2 = float(hits[1]["distance"]) if len(hits) > 1 else d1
    status, _, _ = synthesize_evidence_and_status(
        nearest_dist=d1,
        nearest_ref=n1,
        prob_metrics={"top1_prob": 1.0, "prob_margin": 1.0, "top2_prob": 0.0, "entropy": 0.0},
        concordance={"class": True, "order": True, "family": True, "genus": None, "species": None},
        is_exact_match=bool(n1.get("is_exact_match", False)),
        second_dist=d2,
    )
    return {"distance": d1, "status": status}


def _mutate(seq: str, n_sub: int, rng: random.Random) -> str:
    s = list(seq)
    for pos in rng.sample(range(len(s)), min(n_sub, len(s))):
        s[pos] = rng.choice([c for c in "ACGT" if c != s[pos]])
    return "".join(s)


def _read_fasta(path: str) -> List[Dict[str, str]]:
    recs, header, buf = [], None, []
    if not os.path.exists(path):
        return recs
    for line in open(path):
        if line.startswith(">"):
            if header:
                recs.append({"header": header, "seq": "".join(buf).upper()})
            header, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if header:
        recs.append({"header": header, "seq": "".join(buf).upper()})
    return recs


def build_unknown_queries(ref: pd.DataFrame, seed: int = SEED) -> Dict[str, Any]:
    """Construct the four separate evaluation categories and a calibration/test split."""
    rng = random.Random(seed)
    ref_ids = ref["sequence_id"].tolist()
    calib_ids, test_ids = split_groups(ref_ids, seed)
    seq_of = ref.set_index("sequence_id")["sequence_normalized"].to_dict()

    queries: List[Dict[str, Any]] = []

    # 1. Known reference sequences (exact) - sanity check only
    for sid in ref_ids:
        queries.append(dict(category="known_exact", partition="test" if sid in test_ids else "calibration",
                            truth_unknown=False, query=seq_of[sid], source=sid, library_excludes=None))

    # 2. Altered (mutation-perturbed) known sequences. Truth: known by construction.
    for sid in ref_ids:
        for n_sub in PERTURB_SUBSTITUTIONS:
            q = _mutate(seq_of[sid], n_sub, rng)
            queries.append(dict(category="known_perturbed", partition="test" if sid in test_ids else "calibration",
                                truth_unknown=False, query=q, source=sid,
                                variant=f"{n_sub}_substitutions", library_excludes=None))

    # 3. Held-out genus. Truth: the query's genus is absent from the library.
    genera = sorted(ref["genus"].dropna().unique().tolist())
    g_calib, g_test = split_groups(genera, seed)
    for genus in genera:
        part = "test" if genus in g_test else "calibration"
        for _, r in ref[ref["genus"] == genus].iterrows():
            queries.append(dict(category="heldout_genus", partition=part, truth_unknown=True,
                                query=r["sequence_normalized"], source=r["sequence_id"],
                                library_excludes=genus, genus=genus))

    # 4. External non-fish sequences (NCBI). Truth: not a fish library member.
    skip = ("cytochrome", "nadh", "d-loop", "control region")
    recs = []
    for rec in _read_fasta(FASTA):
        h = rec["header"].lower()
        if "12s" not in h or any(k in h for k in skip):
            continue
        organism = " ".join(rec["header"].split()[1:3])
        recs.append((organism, rec["header"].split()[0], rec["seq"]))
    organisms = sorted({o for o, _, _ in recs})
    o_calib, o_test = split_groups(organisms, seed) if organisms else ([], [])
    seen = set(seq_of.values())
    seen_windows = set()
    for organism, acc, seq in recs:
        for off in WINDOW_OFFSETS:
            w = normalize_sequence(seq[off:off + AMPLICON_WINDOW])
            if len(w) != AMPLICON_WINDOW or set(w) - set("ACGT") or w in seen_windows or w in seen:
                continue
            seen_windows.add(w)
            queries.append(dict(category="external_nonfish", partition="test" if organism in o_test else "calibration",
                                truth_unknown=True, query=w, source=f"{acc}@{off}",
                                library_excludes=None, organism=organism))

    return {
        "queries": queries,
        "partition_info": {
            "seed": seed,
            "reference_sequences": {"calibration": len(calib_ids), "test": len(test_ids)},
            "genera": {"calibration": len(g_calib), "test": len(g_test)},
            "external_organisms": {"calibration": len(o_calib), "test": len(o_test)},
        },
    }


def evaluate_unknown_detection(ref: pd.DataFrame) -> Dict[str, Any]:
    _log("[unknown] building queries")
    built = build_unknown_queries(ref)
    queries = built["queries"]

    full_library = ref
    # Leakage guards: held-out queries must not be in the library they are searched against.
    searchers: Dict[Optional[str], ReferenceSearcher] = {None: _searcher_for(full_library)}
    rows = []
    t0 = time.time()
    for i, q in enumerate(queries):
        key = q["library_excludes"]
        if key not in searchers:
            lib = full_library[full_library["genus"] != key]
            assert_no_sequence_overlap(lib["sequence_normalized"], [q["query"]])
            searchers[key] = _searcher_for(lib)
        elif key is not None:
            assert_no_sequence_overlap(searchers[key].sequences, [q["query"]])
        res = _production_status(q["query"], searchers[key])
        rows.append({**{k: q[k] for k in ("category", "partition", "truth_unknown", "source")},
                     "variant": q.get("variant"), **res})
        if (i + 1) % 150 == 0:
            _log(f"    {i + 1}/{len(queries)} queries ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["flag_out_of_reference"] = df["status"].eq("out_of_reference")
    df["accepted"] = df["status"].eq("known_reference_like")
    df["not_known"] = ~df["accepted"]

    def sub(partition, cats):
        return df[(df["partition"] == partition) & (df["category"].isin(cats))]

    def pooled(partition, flag_col="flag_out_of_reference", accepted_col="accepted"):
        d = sub(partition, ["known_perturbed", "heldout_genus", "external_nonfish"])
        return detection_metrics(d["truth_unknown"], d[flag_col], d[accepted_col])

    def per_unknown_category(partition, cat):
        negs = sub(partition, ["known_perturbed"])
        poss = sub(partition, [cat])
        d = pd.concat([negs, poss])
        m = detection_metrics(d["truth_unknown"], d["flag_out_of_reference"], d["accepted"])
        m["n_category_examples"] = int(len(poss))
        m["median_distance"] = float(poss["distance"].median()) if len(poss) else None
        m["category_detected_as_out_of_reference"] = (
            float(poss["flag_out_of_reference"].mean()) if len(poss) else None
        )
        m["category_accepted_as_known"] = float(poss["accepted"].mean()) if len(poss) else None
        return m

    def known_category(partition, cat):
        d = sub(partition, [cat])
        if not len(d):
            return {"status": "unavailable", "reason": "no examples"}
        return {
            "n": int(len(d)),
            "flagged_out_of_reference_rate": float(d["flag_out_of_reference"].mean()),
            "accepted_rate": float(d["accepted"].mean()),
            "abstention_rate": float((~d["accepted"]).mean()),
            "median_distance": float(d["distance"].median()),
        }

    def perturbation_levels(partition):
        d = sub(partition, ["known_perturbed"])
        out = {}
        for variant, g in d.groupby("variant"):
            out[variant] = {
                "n": int(len(g)),
                "false_positive_rate": float(g["flag_out_of_reference"].mean()),
                "accepted_rate": float(g["accepted"].mean()),
                "abstention_rate": float((~g["accepted"]).mean()),
                "median_distance": float(g["distance"].median()),
                "max_distance": float(g["distance"].max()),
            }
        return dict(sorted(out.items(), key=lambda kv: int(kv[0].split("_")[0])))

    # Calibration of a detection threshold on the CALIBRATION partition only.
    calib = sub("calibration", ["known_perturbed", "heldout_genus", "external_nonfish"])
    cal = calibrate_threshold_youden(calib["distance"], calib["truth_unknown"], SWEEP_THRESHOLDS)

    test_sel = sub("test", ["known_perturbed", "heldout_genus", "external_nonfish"])
    prod_rule = pooled("test")
    prod_abstain_rule = pooled("test", flag_col="not_known", accepted_col="accepted")

    calibrated = None
    if cal is not None:
        flagged = test_sel["distance"] > cal["threshold"]
        calibrated = {
            "chosen_on": "calibration partition only",
            "threshold": cal["threshold"],
            "calibration_youden_j": cal["youden_j"],
            "calibration_sensitivity": cal["sensitivity"],
            "calibration_specificity": cal["specificity"],
            "test_metrics": detection_metrics(test_sel["truth_unknown"], flagged, ~flagged),
            "note": "Diagnostic only. The production Distance Veto threshold is NOT changed.",
        }

    construction = {
        "known_exact": {
            "label": "Known reference sequences (exact)",
            "construction": "Each of the 101 unique reference sequences queried against the full library.",
            "true_label": "known",
            "establishes": "Sanity check that identical sequences are accepted. Distance is 0 by construction.",
            "does_not_establish": "Any generalisation. This is trivially satisfied and is excluded from pooled metrics.",
        },
        "known_perturbed": {
            "label": "Altered (mutation-perturbed) known sequences",
            "construction": (
                f"Each reference with {PERTURB_SUBSTITUTIONS} random base substitutions "
                "(at least one, positions chosen at random, seeded). Library unchanged."
            ),
            "true_label": "known (by construction)",
            "establishes": "Tolerance to substitution-type sequencing/PCR error.",
            "does_not_establish": "Robustness to insertions/deletions, chimeras, real read errors or intraspecific variation.",
        },
        "heldout_genus": {
            "label": "Held-out taxa (genera)",
            "construction": "All sequences of a genus removed from the library, then queried. Library for each query excludes that genus.",
            "true_label": "out-of-reference at genus level (genus absent from the library)",
            "establishes": "Behaviour on a taxon with no reference of its own, often with a relative in the same family.",
            "does_not_establish": (
                "That the sequence is a different species from all library species: a relative of the "
                "same family may legitimately lie within the close-match threshold."
            ),
        },
        "external_nonfish": {
            "label": "External non-fish sequences",
            "construction": (
                f"{AMPLICON_WINDOW}-bp windows (offsets {list(WINDOW_OFFSETS)}) of NCBI mitochondrial 12S records "
                "(mammals, birds, reptiles, amphibians). Identical windows and windows identical to a reference are dropped."
            ),
            "true_label": "out-of-reference (not a fish)",
            "establishes": "Rejection of clearly distant vertebrate DNA from the same gene.",
            "does_not_establish": (
                "Performance on real amplicon reads, non-vertebrate DNA, or unknown fish. Windows come from "
                "full-length records and may not align to the MiFish amplicon boundaries."
            ),
        },
    }

    counts = df.groupby(["category", "partition"]).size().unstack(fill_value=0).to_dict(orient="index")
    return {
        "positive_class": "out_of_reference",
        "definitions": {
            "TP": "unknown sequence flagged out_of_reference",
            "FN": "unknown sequence NOT flagged (accepted or low_confidence)",
            "FP": "known sequence flagged out_of_reference",
            "TN": "known sequence not flagged",
            "false_positive_rate": "FP / (FP + TN)  - known sequences wrongly flagged",
            "false_negative_rate": "FN / (FN + TP)  - unknown sequences wrongly kept",
            "coverage": "fraction of queries with status known_reference_like (accepted identification)",
            "abstention_rate": "fraction with status low_confidence or out_of_reference",
        },
        "decision_rule_evaluated": (
            "Production evidence rule (src/prediction/evidence.py::synthesize_evidence_and_status) "
            "with class/order concordance assumed TRUE. Only the distance rules (Distance Veto and "
            "close-match) are exercised, so coverage is an upper bound and classifier probabilities "
            "are not evaluated here."
        ),
        "production_thresholds": {"close_match": THRESHOLD_CLOSE_SIMILARITY, "veto": THRESHOLD_OUT_OF_REFERENCE},
        "threshold_independence_warning": (
            "The production thresholds (0.03 / 0.15) were selected in an earlier experiment that used "
            "all of these sequences. Test-partition results for the PRODUCTION thresholds are therefore "
            "optimistic. The calibrated-threshold result is the leakage-free estimate for a threshold "
            "chosen without seeing the test partition."
        ),
        "construction": construction,
        "partition_info": built["partition_info"],
        "counts_by_category_and_partition": counts,
        "calibration": {
            "n_known_perturbed": int(len(sub("calibration", ["known_perturbed"]))),
            "n_unknown": int(len(sub("calibration", ["heldout_genus", "external_nonfish"]))),
            "chosen": cal,
        },
        "test": {
            "n_known_perturbed": int(len(sub("test", ["known_perturbed"]))),
            "n_unknown": int(len(sub("test", ["heldout_genus", "external_nonfish"]))),
            "production_rule_out_of_reference": prod_rule,
            "production_rule_reject_or_abstain": prod_abstain_rule,
            "per_unknown_category": {
                "heldout_genus": per_unknown_category("test", "heldout_genus"),
                "external_nonfish": per_unknown_category("test", "external_nonfish"),
            },
            "known_categories": {
                "known_perturbed": known_category("test", "known_perturbed"),
                "known_exact": known_category("test", "known_exact"),
            },
            "perturbation_levels": perturbation_levels("test"),
            "distance_auroc": distance_auroc(test_sel["distance"], test_sel["truth_unknown"]),
            "threshold_sweep_descriptive": threshold_sweep(
                test_sel["distance"], test_sel["truth_unknown"], SWEEP_THRESHOLDS
            ),
            "calibrated_threshold": calibrated,
        },
    }


# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = pd.read_csv(os.path.join(PROCESSED, "reference_barcodes.csv"))
    ref, dedupe_report = deduplicate_by_sequence(raw)
    _log(f"[data] {dedupe_report}")

    searcher = _searcher_for(ref)
    taxonomy = evaluate_taxonomy(ref, searcher)
    unknown = evaluate_unknown_detection(ref)

    aug_path = os.path.join(BASE_DIR, "data", "raw", "hackathon_dna_taxonomy_merged_with_minority_synthetic.csv")
    if os.path.exists(aug_path):
        raw_rows = len(pd.read_csv(aug_path, low_memory=False))
    else:
        audit_path = os.path.join(BASE_DIR, "reports", "audit_metrics.json")
        raw_rows = None
        if os.path.exists(audit_path):
            raw_rows = json.load(open(audit_path)).get("dataset_size", {}).get("rows")

    result = {
        "meta": {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "seed": SEED,
            "raw_dataset_rows": raw_rows,
            "reference_library": dedupe_report,
            "note": (
                "All metrics are computed from held-out predictions on unique sequences. "
                "Training-set scores are never reported as evidence of generalisation."
            ),
        },
        "taxonomy": taxonomy,
        "unknown_detection": unknown,
    }
    path = os.path.join(OUT_DIR, "evaluation_results.json")
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2)
    _log(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
