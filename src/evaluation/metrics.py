"""
src/evaluation/metrics.py
Pure, side-effect-free metric calculations for the BioScan evaluation dashboard.

Nothing here reads files, trains models or imports the production engine, so every
function can be unit-tested against hand-computed values. All functions return
`None` / an explicit "unavailable" status instead of a number when a metric is
mathematically undefined or the data cannot support it. Nothing is fabricated.

Conventions
-----------
* Taxonomy metrics: macro averages are taken over the classes that are present in
  the ground truth (the *evaluated* classes). A class that only appears in
  predictions is counted as an error for the true class it displaced but does not
  add a row of zeros to the macro average.
* Unknown detection: the POSITIVE class is "out-of-reference".
      TP = unknown sequence flagged out-of-reference
      FN = unknown sequence accepted/kept as known   (missed unknown)
      FP = known sequence flagged out-of-reference   (false alarm)
      TN = known sequence not flagged
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# A rank is only evaluated when at least this much held-out data exists.
# These are conservative rules of thumb, not statistical guarantees.
MIN_EVAL_SAMPLES = 20
MIN_EVAL_CLASSES = 2
LOW_SUPPORT_PER_CLASS = 5


def _safe_div(num: float, den: float) -> Optional[float]:
    return None if den == 0 else float(num) / float(den)


# ---------------------------------------------------------------------------
# Taxonomy classification
# ---------------------------------------------------------------------------
def assess_sufficiency(
    n_eligible: int,
    n_evaluated_classes: int,
    min_samples: int = MIN_EVAL_SAMPLES,
    min_classes: int = MIN_EVAL_CLASSES,
) -> Dict[str, Any]:
    """Decide whether a rank has enough held-out data for meaningful metrics."""
    reasons: List[str] = []
    if n_eligible < min_samples:
        reasons.append(
            f"only {n_eligible} evaluable held-out sequences (minimum {min_samples})"
        )
    if n_evaluated_classes < min_classes:
        reasons.append(
            f"only {n_evaluated_classes} evaluable class(es) (minimum {min_classes})"
        )
    return {"sufficient": not reasons, "reasons": reasons}


def taxonomy_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    y_prob: Optional[Sequence[Any]] = None,
    min_samples: int = MIN_EVAL_SAMPLES,
    min_classes: int = MIN_EVAL_CLASSES,
) -> Dict[str, Any]:
    """
    Compute classification metrics from pooled held-out predictions.

    Returns {"status": "insufficient", ...counts, "reason": ...} with NO metric values
    when the data cannot support a meaningful evaluation.
    """
    y_true = np.asarray(list(y_true), dtype=object)
    y_pred = np.asarray(list(y_pred), dtype=object)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    n = int(len(y_true))
    true_classes = sorted(set(y_true.tolist())) if n else []
    suff = assess_sufficiency(n, len(true_classes), min_samples, min_classes)
    base = {
        "n_eligible_test_examples": n,
        "n_evaluated_classes": len(true_classes),
    }
    if not suff["sufficient"]:
        return {
            **base,
            "status": "insufficient",
            "message": "Insufficient data for reliable evaluation",
            "reason": "; ".join(suff["reasons"]),
        }

    # Macro averages over evaluated (ground-truth) classes only.
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=true_classes, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=true_classes, average="weighted", zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    accuracy = float(np.mean(y_true == y_pred))

    # Multiclass One-vs-Rest Macro ROC-AUC
    macro_roc_auc = None
    if y_prob is not None and len(true_classes) >= 2:
        try:
            prob_arr = np.asarray(y_prob, dtype=float)
            if prob_arr.ndim == 2 and prob_arr.shape[1] == len(true_classes):
                macro_roc_auc = float(roc_auc_score(
                    y_true, prob_arr, multi_class="ovr", average="macro", labels=true_classes
                ))
        except Exception:
            macro_roc_auc = None

    all_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, average=None, zero_division=0
    )
    per_class = []
    for lab, pi, ri, fi, si in zip(all_labels, p, r, f1, support):
        per_class.append(
            {
                "class": str(lab),
                "precision": float(pi),
                # recall is undefined for a class never present in the ground truth
                "recall": float(ri) if si > 0 else None,
                "f1": float(fi) if si > 0 else None,
                "support": int(si),
                "predicted_count": int(np.sum(y_pred == lab)),
            }
        )

    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    mean_support = n / len(true_classes)
    warnings = []
    if mean_support < LOW_SUPPORT_PER_CLASS:
        warnings.append(
            f"Low per-class support (mean {mean_support:.1f} test examples per class); "
            "per-class values are highly unstable."
        )

    return {
        **base,
        "status": "ok",
        "top1_accuracy": accuracy,
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_roc_auc": macro_roc_auc,
        "mean_support_per_class": float(mean_support),
        "per_class": per_class,
        "confusion_matrix": {"labels": [str(x) for x in all_labels], "matrix": cm.tolist()},
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Unknown (out-of-reference) detection
# ---------------------------------------------------------------------------
def detection_metrics(
    y_true_unknown: Sequence[bool],
    y_pred_unknown: Sequence[bool],
    y_pred_accepted: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """
    Binary known-vs-out-of-reference metrics. Positive class = out-of-reference.

    y_true_unknown : verified label, True = sequence is NOT represented in the library.
    y_pred_unknown : system flagged the sequence as out_of_reference.
    y_pred_accepted: (optional) system issued an accepted identification
                     (status known_reference_like). Used for coverage/abstention.
    """
    t = np.asarray(list(y_true_unknown), dtype=bool)
    p = np.asarray(list(y_pred_unknown), dtype=bool)
    if len(t) != len(p):
        raise ValueError("label and prediction arrays must have the same length")

    tp = int(np.sum(t & p))
    fn = int(np.sum(t & ~p))
    fp = int(np.sum(~t & p))
    tn = int(np.sum(~t & ~p))
    n_unknown = tp + fn
    n_known = fp + tn
    n = len(t)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = None if precision is None or recall is None or (precision + recall) == 0 else (
        2 * precision * recall / (precision + recall)
    )
    out = {
        "n_total": n,
        "n_known": n_known,
        "n_unknown": n_unknown,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall_sensitivity": recall,
        "f1": f1,
        "specificity": _safe_div(tn, tn + fp),
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "confusion_matrix": {
            "labels": ["known_reference", "out_of_reference"],
            "matrix": [[tn, fp], [fn, tp]],  # rows = truth, cols = prediction
        },
        "flagged_out_of_reference_rate": _safe_div(int(p.sum()), n),
    }
    if y_pred_accepted is not None:
        a = np.asarray(list(y_pred_accepted), dtype=bool)
        if len(a) != n:
            raise ValueError("accepted array length mismatch")
        out["coverage_accepted_rate"] = _safe_div(int(a.sum()), n)
        out["abstention_rate"] = _safe_div(int((~a).sum()), n)
    return out


def distance_auroc(
    distances: Sequence[float], y_true_unknown: Sequence[bool], min_per_class: int = 5
) -> Dict[str, Any]:
    """
    AUROC of nearest-reference distance as a score for 'out-of-reference'.
    Only computed when both classes have at least `min_per_class` examples.
    """
    d = np.asarray(list(distances), dtype=float)
    t = np.asarray(list(y_true_unknown), dtype=bool)
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos < min_per_class or n_neg < min_per_class:
        return {
            "status": "unavailable",
            "reason": f"needs >= {min_per_class} examples of each class "
                      f"(unknown={n_pos}, known={n_neg})",
        }
    return {
        "status": "ok",
        "auroc": float(roc_auc_score(t, d)),
        "n_unknown": n_pos,
        "n_known": n_neg,
    }


def threshold_sweep(
    distances: Sequence[float],
    y_true_unknown: Sequence[bool],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """
    For each candidate veto threshold t, flag distance > t as out-of-reference and
    report FPR / FNR / coverage. Purely descriptive: it must not be used to pick a
    threshold on the same data it is reported on.
    """
    d = np.asarray(list(distances), dtype=float)
    t = np.asarray(list(y_true_unknown), dtype=bool)
    rows = []
    for thr in thresholds:
        flagged = d > thr
        m = detection_metrics(t, flagged, y_pred_accepted=~flagged)
        rows.append(
            {
                "threshold": float(thr),
                "false_positive_rate": m["false_positive_rate"],
                "false_negative_rate": m["false_negative_rate"],
                "recall_sensitivity": m["recall_sensitivity"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "coverage": m["coverage_accepted_rate"],
            }
        )
    return rows


def calibrate_threshold_youden(
    distances: Sequence[float],
    y_true_unknown: Sequence[bool],
    candidates: Sequence[float],
) -> Optional[Dict[str, Any]]:
    """
    Choose the threshold maximising Youden's J = sensitivity + specificity - 1 on a
    CALIBRATION set. Ties are broken toward the lower false-positive rate.
    Returns None when the calibration set lacks both classes.
    """
    d = np.asarray(list(distances), dtype=float)
    t = np.asarray(list(y_true_unknown), dtype=bool)
    if t.sum() == 0 or (~t).sum() == 0:
        return None
    best = None
    for thr in candidates:
        flagged = d > thr
        sens = np.sum(t & flagged) / t.sum()
        spec = np.sum(~t & ~flagged) / (~t).sum()
        j = sens + spec - 1.0
        key = (j, spec)
        if best is None or key > best["_key"]:
            best = {"threshold": float(thr), "youden_j": float(j),
                    "sensitivity": float(sens), "specificity": float(spec), "_key": key}
    best.pop("_key")
    return best


# ---------------------------------------------------------------------------
# Coverage / selective prediction
# ---------------------------------------------------------------------------
def accuracy_coverage_curve(
    confidence: Sequence[float],
    correct: Sequence[bool],
    n_points: int = 10,
) -> List[Dict[str, Any]]:
    """
    Selective-prediction curve: accept predictions with confidence >= t and report
    coverage (fraction accepted) and accuracy among the accepted.
    """
    c = np.asarray(list(confidence), dtype=float)
    ok = np.asarray(list(correct), dtype=bool)
    if len(c) == 0:
        return []
    qs = np.unique(np.quantile(c, np.linspace(0, 1, n_points, endpoint=False)))
    rows = []
    for thr in qs:
        acc_mask = c >= thr
        n_acc = int(acc_mask.sum())
        rows.append(
            {
                "confidence_threshold": float(thr),
                "coverage": n_acc / len(c),
                "accepted": n_acc,
                "accuracy_of_accepted": float(ok[acc_mask].mean()) if n_acc else None,
            }
        )
    return rows


def distance_coverage_curve(
    distances: Sequence[float],
    correct: Sequence[bool],
    thresholds: Sequence[float],
) -> List[Dict[str, Any]]:
    """Accept nearest-reference label only when distance <= t."""
    d = np.asarray(list(distances), dtype=float)
    ok = np.asarray(list(correct), dtype=bool)
    rows = []
    for thr in thresholds:
        m = d <= thr
        n_acc = int(m.sum())
        rows.append(
            {
                "distance_threshold": float(thr),
                "coverage": n_acc / len(d) if len(d) else None,
                "accepted": n_acc,
                "accuracy_of_accepted": float(ok[m].mean()) if n_acc else None,
            }
        )
    return rows
