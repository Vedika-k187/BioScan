"""
BioScan: SVM taxonomy classifier (baseline, leakage-safe tuning, saved artefact)
Module: src/models/svm_model.py

The SVM is a fourth classifier beside KNN, Random Forest and XGBoost. It is trained
on exactly the same DNA features as the production CLASS model (3-mer frequencies from
training_features_k3.csv, standardised) and is evaluated by src/evaluation/run_model_evaluation.py.

Leakage handling (important)
----------------------------
The training table holds 101 genuine sequences + 158 synthetic variants. Every synthetic
variant is a 1-3 substitution copy of ONE genuine "parent" sequence. Hyperparameters are
therefore tuned with a custom splitter that

  * validates ONLY on genuine sequences (never on synthetic variants), and
  * only on classes that have at least two independent genuine sequences
    (Teleostei, Elasmobranchii), and
  * removes every sequence that shares a parent group with a validation sequence from
    that fold's training data.

Chondrostei and Petromyzonti each have a single independent biological sequence, so no
cross-validation can measure how well they generalise; they stay in every training fold
but never in a validation fold.

The scaler lives inside the sklearn Pipeline, so it is fitted on the training part of each
fold only.

Run:  PYTHONPATH=. python src/models/svm_model.py
"""

import json
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.similarity.reference_search import levenshtein_distance  # noqa: E402

SEED = 42
PROCESSED = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_BARCODES = os.path.join(PROCESSED, "training_barcodes.csv")
TRAINING_FEATURES = os.path.join(PROCESSED, "training_features_k3.csv")
SVM_ARTIFACT = os.path.join(MODELS_DIR, "baseline_svm.joblib")

FEATURE_SET = "k=3"                      # same as the production class model
CLASS_ORDER = ["Teleostei", "Elasmobranchii", "Chondrostei", "Petromyzonti"]

BASELINE_PARAMS: Dict[str, Any] = {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
C_GRID: List[float] = [0.01, 0.1, 1, 10, 100, 1000]
GAMMA_GRID: List[Any] = ["scale", "auto", 0.001, 0.01, 0.1, 1]
SCORING = "balanced_accuracy"
N_SPLITS = 5


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def assign_parent_groups(df: pd.DataFrame) -> np.ndarray:
    """
    Group id per row. A genuine sequence is its own group. A synthetic variant joins the
    group of its nearest genuine sequence of the same species (same class as fallback).
    """
    syn = df["is_synthetic"].astype(bool).values
    ids = df["sequence_id"].astype(str).values
    seqs = df["sequence_normalized"].astype(str).values
    groups = ids.copy()
    genuine_idx = np.where(~syn)[0]
    for i in np.where(syn)[0]:
        cand = [j for j in genuine_idx if df["species"].iloc[j] == df["species"].iloc[i]]
        if not cand:
            cand = [j for j in genuine_idx if df["class"].iloc[j] == df["class"].iloc[i]]
        if not cand:
            continue
        best = min(cand, key=lambda j: levenshtein_distance(seqs[i], seqs[j]))
        groups[i] = ids[best]
    return groups


def load_class_training_data() -> Dict[str, Any]:
    """The 259 unique training sequences, k=3 features and class labels (no metadata features)."""
    df = pd.read_csv(TRAINING_BARCODES)
    df = df[df["class"].notna()].reset_index(drop=True)
    feat = pd.read_csv(TRAINING_FEATURES)
    cols = [c for c in feat.columns if c != "sequence_id"]
    merged = df[["sequence_id"]].merge(feat, on="sequence_id", how="left")
    if merged[cols].isna().any().any():
        raise RuntimeError("Missing k=3 features for some training sequences")
    return {
        "df": df,
        "X": merged[cols].values.astype(float),
        "y": df["class"].astype(str).values,
        "feature_columns": cols,
        "is_synth": df["is_synthetic"].astype(bool).values,
        "groups": assign_parent_groups(df),
    }


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def make_svm(params: Optional[Dict[str, Any]] = None, probability: bool = True) -> SVC:
    p = {**BASELINE_PARAMS, **(params or {})}
    return SVC(C=p["C"], gamma=p["gamma"], kernel=p["kernel"],
               probability=probability, random_state=SEED)


def make_pipeline(params: Optional[Dict[str, Any]] = None, probability: bool = True) -> Pipeline:
    """StandardScaler -> SVC. The scaler is therefore always fitted on training data only."""
    return Pipeline([("scaler", StandardScaler()), ("svm", make_svm(params, probability))])


# ---------------------------------------------------------------------------
# Leakage-safe validation splitter
# ---------------------------------------------------------------------------
def tuning_splits(y: np.ndarray, groups: np.ndarray, is_synth: np.ndarray,
                  n_splits: int = N_SPLITS, seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    (train_idx, val_idx) pairs over the rows of the pool passed in. See module docstring.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    is_synth = np.asarray(is_synth, dtype=bool)
    n_groups = {c: len(set(groups[y == c])) for c in set(y.tolist())}
    eligible = np.array([i for i in range(len(y)) if (not is_synth[i]) and n_groups[y[i]] >= 2])
    if len(eligible) == 0 or len(set(y[eligible].tolist())) < 2:
        raise ValueError("Need >= 2 classes with >= 2 independent genuine sequences to tune")
    counts = pd.Series(y[eligible]).value_counts()
    k = int(min(n_splits, counts.min()))
    if k < 2:
        raise ValueError("Not enough genuine sequences per class for cross-validated tuning")
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    splits = []
    all_idx = np.arange(len(y))
    for _, v in skf.split(eligible, y[eligible]):
        val = eligible[v]
        banned = set(groups[val].tolist())
        tr = np.array([i for i in all_idx if i not in set(val.tolist()) and groups[i] not in banned])
        splits.append((tr, val))
    return splits


def _select_best(cv_results: Dict[str, Any]) -> int:
    """Highest mean score; ties go to the smaller C, then to the earlier gamma in the grid."""
    mean = np.asarray(cv_results["mean_test_score"], dtype=float)
    best = np.nanmax(mean)
    tied = [i for i in range(len(mean)) if abs(mean[i] - best) < 1e-12]
    def key(i):
        p = cv_results["params"][i]
        return (float(p["svm__C"]), GAMMA_GRID.index(p["svm__gamma"]))
    return min(tied, key=key)


def tune_svm(X: np.ndarray, y: np.ndarray, groups: np.ndarray, is_synth: np.ndarray,
             n_jobs: int = 1) -> Dict[str, Any]:
    """GridSearchCV over C x gamma with the leakage-safe splitter, scored by balanced accuracy."""
    splits = tuning_splits(y, groups, is_synth)
    grid = {"svm__C": C_GRID, "svm__gamma": GAMMA_GRID}
    t0 = time.time()
    gs = GridSearchCV(
        make_pipeline(probability=False), grid, scoring=SCORING, cv=splits,
        refit=False, n_jobs=n_jobs, error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    runtime = time.time() - t0
    res = gs.cv_results_
    bi = _select_best(res)
    best = res["params"][bi]
    return {
        "best_params": {"C": float(best["svm__C"]), "gamma": best["svm__gamma"], "kernel": "rbf"},
        "best_cv_score": float(res["mean_test_score"][bi]),
        "best_cv_std": float(res["std_test_score"][bi]),
        "n_cv_folds": len(splits),
        "n_candidates": len(res["params"]),
        "n_fits": len(res["params"]) * len(splits),
        "runtime_seconds": round(runtime, 2),
        "search_results": [
            {"C": float(p["svm__C"]), "gamma": p["svm__gamma"],
             "mean_balanced_accuracy": float(res["mean_test_score"][i]),
             "std": float(res["std_test_score"][i]), "rank": int(res["rank_test_score"][i])}
            for i, p in enumerate(res["params"])
        ],
    }


# ---------------------------------------------------------------------------
# Fit / predict helpers shared with the evaluation harness
# ---------------------------------------------------------------------------
def fit_predict(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: Sequence[int],
                classes: List[str], params: Optional[Dict[str, Any]] = None
                ) -> Tuple[List[str], np.ndarray]:
    """Fit StandardScaler+SVC on train_idx, return (labels, probability matrix over `classes`)."""
    pipe = make_pipeline(params, probability=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X[train_idx], y[train_idx])
    Xt = X[list(test_idx)]
    labels = [str(v) for v in pipe.predict(Xt)]
    pr = pipe.predict_proba(Xt)
    out = np.zeros((len(Xt), len(classes)), dtype=float)
    for j, c in enumerate(pipe.classes_):
        out[:, classes.index(str(c))] = pr[:, j]
    return labels, out


def fit_predict_tuned(X: np.ndarray, y: np.ndarray, groups: np.ndarray, is_synth: np.ndarray,
                      train_idx: np.ndarray, test_idx: Sequence[int], classes: List[str]
                      ) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """Nested step: tune on the training rows only, then fit and predict the held-out rows."""
    t = tune_svm(X[train_idx], y[train_idx], groups[train_idx], is_synth[train_idx], n_jobs=1)
    labels, probs = fit_predict(X, y, train_idx, test_idx, classes, t["best_params"])
    return labels, probs, t["best_params"]


# ---------------------------------------------------------------------------
# Train and save the final artefact
# ---------------------------------------------------------------------------
def train_and_save(path: str = SVM_ARTIFACT) -> Dict[str, Any]:
    data = load_class_training_data()
    X, y = data["X"], data["y"]
    t = tune_svm(X, y, data["groups"], data["is_synth"], n_jobs=-1)
    pipe = make_pipeline(t["best_params"], probability=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    artefact = {
        "pipeline": pipe,
        "feature_set": FEATURE_SET,
        "feature_columns": data["feature_columns"],
        "target_classes": [str(c) for c in pipe.classes_],
        "config": {
            "model": "SVM (StandardScaler + SVC)", "kernel": "rbf",
            "baseline_params": BASELINE_PARAMS, "best_params": t["best_params"],
            "scoring": SCORING, "param_grid": {"C": C_GRID, "gamma": GAMMA_GRID},
            "n_cv_folds": t["n_cv_folds"],
            "cv_strategy": "stratified 5-fold over genuine Teleostei/Elasmobranchii sequences; "
                           "parent-group siblings removed from each training fold",
        },
        "metrics": {
            "training_samples": int(len(y)),
            "n_genuine": int((~data["is_synth"]).sum()),
            "n_synthetic": int(data["is_synth"].sum()),
            "best_cv_balanced_accuracy": t["best_cv_score"],
            "tuning_runtime_seconds": t["runtime_seconds"],
            "note": "Held-out generalisation metrics are in reports/evaluation/evaluation_results.json "
                    "(taxonomy.class.svm). The CV score above is a tuning score, not a test score.",
        },
        "random_state": SEED,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(artefact, path)
    return {"artefact_path": path, "tuning": {k: v for k, v in t.items() if k != "search_results"}}


if __name__ == "__main__":
    print(json.dumps(train_and_save(), indent=2, default=str))
