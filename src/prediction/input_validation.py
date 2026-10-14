"""
src/prediction/input_validation.py
Phase 8: BioScan Biological DNA Sequence Input Validation & Quality Control

Performs strict, scientifically justified QC on input DNA sequences:
- Validates strictly {A, C, G, T} nucleotide alphabet.
- Rejects ambiguous IUPAC codes (e.g. N, R, Y, S, W, K, M, B, D, H, V).
- Rejects non-alphabetic noise, whitespace, linebreaks, digits, and punctuation.
- Evaluates 12S rDNA V5 amplicon length bounds (87-112 bp reference baseline).
- Produces structured QC reports without crashing on malformed inputs.
"""

import re
from typing import Dict, List, Optional, Tuple, Any

VALID_BASES_SET = set("ACGT")
IUPAC_AMBIGUOUS_SET = set("NRYWSKMBDHV")
REF_AMPLICON_RANGE = (87, 112)
UNUSUAL_SHORT_THRESHOLD = 50
UNUSUAL_LONG_THRESHOLD = 300


def validate_dna_sequence(
    sequence: Any,
    sequence_id: Optional[str] = None
) -> Tuple[bool, Optional[str], List[str], List[str], Dict[str, Any]]:
    """
    Perform rigorous biological Quality Control (QC) on an input DNA sequence.

    Parameters:
    - sequence: DNA sequence as string or object convertible to string
    - sequence_id: optional user-provided sequence identifier

    Returns:
    - is_valid: bool (True if passes all strict biological criteria)
    - clean_sequence: Optional[str] (uppercase stripped A/C/G/T sequence if valid)
    - warnings: List[str] (biological observations, e.g. unusual amplicon length)
    - errors: List[str] (fatal QC failure reasons)
    - metrics: Dict[str, Any] (length, GC content, base counts)
    """
    warnings: List[str] = []
    errors: List[str] = []

    # 1. Null / None Check
    if sequence is None:
        errors.append("Input sequence is None.")
        return False, None, warnings, errors, {}

    # 2. Type Check
    if not isinstance(sequence, str):
        try:
            sequence = str(sequence)
        except Exception:
            errors.append("Input could not be converted to string.")
            return False, None, warnings, errors, {}

    # 3. Clean whitespace, tabs, and line breaks
    clean_seq = re.sub(r"\s+", "", sequence).upper().strip()

    # 4. Empty Sequence Check
    if len(clean_seq) == 0:
        errors.append("Input sequence is empty.")
        return False, None, warnings, errors, {}

    # 5. Strictly Valid Nucleotide Alphabet {A, C, G, T}
    invalid_chars = set(clean_seq) - VALID_BASES_SET
    if invalid_chars:
        sorted_invalid = sorted(list(invalid_chars))
        has_ambiguous = any(c in IUPAC_AMBIGUOUS_SET for c in invalid_chars)
        msg = f"Invalid characters detected in sequence: {sorted_invalid}."
        if has_ambiguous:
            ambig_chars = sorted(list(set(clean_seq) & IUPAC_AMBIGUOUS_SET))
            msg += f" Ambiguous IUPAC nucleotide codes {ambig_chars} are not permitted in reference-grade DNA barcodes."
        errors.append(msg)
        return False, None, warnings, errors, {}

    # 6. Biological Amplicon Length Checks
    seq_len = len(clean_seq)
    if seq_len < UNUSUAL_SHORT_THRESHOLD:
        warnings.append(
            f"Sequence length ({seq_len} bp) is unusually short for 12S rDNA V5 amplicons "
            f"(reference baseline: {REF_AMPLICON_RANGE[0]}-{REF_AMPLICON_RANGE[1]} bp)."
        )
    elif seq_len > UNUSUAL_LONG_THRESHOLD:
        warnings.append(
            f"Sequence length ({seq_len} bp) is unusually long for 12S rDNA V5 amplicons "
            f"(reference baseline: {REF_AMPLICON_RANGE[0]}-{REF_AMPLICON_RANGE[1]} bp)."
        )
    elif seq_len < REF_AMPLICON_RANGE[0] or seq_len > REF_AMPLICON_RANGE[1]:
        warnings.append(
            f"Sequence length ({seq_len} bp) deviates slightly from the curated reference range "
            f"({REF_AMPLICON_RANGE[0]}-{REF_AMPLICON_RANGE[1]} bp), but is within permissible bounds."
        )

    # 7. Compositional Metrics
    g_count = clean_seq.count("G")
    c_count = clean_seq.count("C")
    a_count = clean_seq.count("A")
    t_count = clean_seq.count("T")
    gc_content = round((g_count + c_count) / seq_len, 4) if seq_len > 0 else 0.0

    qc_metrics = {
        "sequence_id": sequence_id,
        "length": seq_len,
        "gc_content": gc_content,
        "A_count": a_count,
        "C_count": c_count,
        "G_count": g_count,
        "T_count": t_count,
        "is_reference_length": (REF_AMPLICON_RANGE[0] <= seq_len <= REF_AMPLICON_RANGE[1])
    }

    return True, clean_seq, warnings, errors, qc_metrics


def validate_and_clean_sequence(sequence: Any) -> Tuple[bool, Optional[str], List[str], Dict[str, Any]]:
    """
    Backward-compatibility wrapper matching the Phase 6 signature:
    returns (is_valid, clean_seq, messages, qc_metrics).
    """
    ok, clean, warnings, errors, metrics = validate_dna_sequence(sequence)
    messages = errors if not ok else warnings
    metrics_with_warnings = dict(metrics)
    metrics_with_warnings["warnings"] = warnings
    return ok, clean, messages, metrics_with_warnings
