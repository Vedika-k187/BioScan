"""
src/evaluation/splits.py
Leakage-safe data handling for BioScan evaluation.

The raw survey file repeats every unique DNA sequence 136 times (sampling events x
primer replicates). Evaluating on those rows would put identical sequences in both
training and test data. Everything here works on UNIQUE NORMALISED SEQUENCES.
"""

import random
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

_WS = re.compile(r"\s+")


def normalize_sequence(seq: str) -> str:
    """Uppercase, remove all whitespace. Does not alter bases."""
    return _WS.sub("", str(seq)).upper()


def deduplicate_by_sequence(
    df: pd.DataFrame,
    seq_col: str = "sequence_normalized",
    label_cols: Sequence[str] = ("class", "order", "family", "genus", "species"),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Collapse rows with identical normalised sequences to one row.

    Returns (unique_df, report). `report` counts input rows, unique sequences, and
    label conflicts (identical sequence carrying different labels). On conflict the
    first record is kept and the conflict is counted, never silently hidden.
    """
    work = df.copy()
    work["_norm_seq"] = work[seq_col].map(normalize_sequence)
    n_rows = len(work)
    conflicts = 0
    for col in label_cols:
        if col in work.columns:
            per_seq = work.groupby("_norm_seq")[col].nunique(dropna=True)
            conflicts += int((per_seq > 1).sum())
    unique = work.drop_duplicates("_norm_seq", keep="first").reset_index(drop=True)
    unique[seq_col] = unique["_norm_seq"]
    unique = unique.drop(columns="_norm_seq")
    return unique, {
        "input_rows": int(n_rows),
        "unique_sequences": int(len(unique)),
        "duplicate_rows_removed": int(n_rows - len(unique)),
        "label_conflicts": int(conflicts),
    }


def assert_no_sequence_overlap(
    train_seqs: Iterable[str], test_seqs: Iterable[str]
) -> None:
    """Raise AssertionError when any normalised sequence is in both partitions."""
    tr: Set[str] = {normalize_sequence(s) for s in train_seqs}
    te: Set[str] = {normalize_sequence(s) for s in test_seqs}
    overlap = tr & te
    if overlap:
        raise AssertionError(
            f"Leakage: {len(overlap)} identical sequence(s) appear in both train and test"
        )


def eligible_for_holdout(labels: Sequence) -> np.ndarray:
    """
    Boolean mask: a sequence can be evaluated by leave-one-out only if at least one
    OTHER sequence shares its label (otherwise the true class is absent from training
    and a closed-set classifier cannot possibly predict it). Singleton-class
    sequences are returned as False and must be REPORTED, not silently dropped.
    """
    s = pd.Series(list(labels))
    counts = s.map(s.value_counts())
    return (counts >= 2).to_numpy()


def class_support_summary(labels: Sequence) -> Dict[str, int]:
    """Counts describing how much of a rank is evaluable."""
    s = pd.Series([x for x in labels if pd.notna(x)])
    vc = s.value_counts()
    n_singleton_classes = int((vc == 1).sum())
    n_eval = int(vc[vc >= 2].sum())
    return {
        "n_labelled_sequences": int(len(s)),
        "n_classes_total": int(len(vc)),
        "n_singleton_classes": n_singleton_classes,
        "n_multi_example_classes": int((vc >= 2).sum()),
        "n_eligible_sequences": n_eval,
        "n_unevaluable_singleton_sequences": int(len(s) - n_eval),
    }


def split_groups(groups: Iterable[str], seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Deterministically split group names (genera / organisms / sequence ids) into a
    calibration half and a test half. Groups never appear in both halves.
    """
    uniq = sorted(set(groups))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    half = len(uniq) // 2
    calibration, test = uniq[:half], uniq[half:]
    assert not (set(calibration) & set(test))
    return sorted(calibration), sorted(test)
