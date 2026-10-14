"""
src/similarity/reference_search.py
Phase 6: BioScan Reference Library Similarity Search Module

Provides fast, robust similarity searching of query DNA sequences against
the 101 curated reference barcodes (12S rDNA V5 amplicons, 87-112 bp).

Handles:
- Exact sequence matching
- Hamming distance (for equal-length amplicons)
- Normalized Levenshtein edit distance (for unequal-length amplicons)
- Sequence identity percentage
- Top-k ranked retrieval with complete taxonomic metadata
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REF_BARCODES_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
TAXONOMY_REF_PATH = os.path.join(PROCESSED_DIR, "taxonomy_reference.csv")


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings using 2-row dynamic programming.
    Time complexity: O(len(s1) * len(s2)), Space complexity: O(min(len(s1), len(s2))).
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1] + [0] * len(s2)
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row[j + 1] = min(insertions, deletions, substitutions)
        previous_row = current_row
    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Compute normalized edit distance: Levenshtein(s1, s2) / max(len(s1), len(s2)).
    Returns a float in [0.0, 1.0].
    """
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    return levenshtein_distance(s1, s2) / max_len


def hamming_distance(s1: str, s2: str) -> Optional[int]:
    """
    Compute Hamming distance for equal-length sequences.
    Returns None if sequence lengths differ.
    """
    if len(s1) != len(s2):
        return None
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


class ReferenceSearcher:
    """
    Index and search engine for the BioScan reference barcode library.
    """
    def __init__(self, ref_path: str = REF_BARCODES_PATH):
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference barcodes file not found at: {ref_path}")
        
        self.ref_df = pd.read_csv(ref_path)
        self.sequences = self.ref_df["sequence_normalized"].tolist()
        self.seq_ids = self.ref_df["sequence_id"].tolist()
        self.n_references = len(self.ref_df)

    def search(
        self,
        query_sequence: str,
        top_k: int = 5,
        exclude_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Search reference library for the top-k most similar barcodes to query_sequence.

        Parameters:
        - query_sequence: DNA string (normalized uppercase A/C/G/T)
        - top_k: number of nearest references to return
        - exclude_id: optional sequence_id to exclude (for leave-one-out validation)

        Returns:
        Ranked list of top-k match dictionaries.
        """
        q_clean = str(query_sequence).upper().strip()
        q_len = len(q_clean)
        if q_len == 0:
            raise ValueError("Query sequence cannot be empty.")

        matches = []
        for idx, row in self.ref_df.iterrows():
            ref_id = row["sequence_id"]
            if exclude_id and ref_id == exclude_id:
                continue

            ref_seq = row["sequence_normalized"]
            ref_len = len(ref_seq)

            # Compute distances
            edit_dist = levenshtein_distance(q_clean, ref_seq)
            norm_dist = edit_dist / max(q_len, ref_len)
            seq_ident = 1.0 - norm_dist
            ham_dist = hamming_distance(q_clean, ref_seq)

            matches.append({
                "sequence_id": ref_id,
                "distance": round(norm_dist, 4),
                "similarity": round(seq_ident, 4),
                "edit_distance": edit_dist,
                "hamming_distance": ham_dist,
                "length": ref_len,
                "length_diff": abs(q_len - ref_len),
                "is_exact_match": (edit_dist == 0),
                "class": row["class"] if pd.notnull(row["class"]) else None,
                "order": row["order"] if pd.notnull(row["order"]) else None,
                "family": row["family"] if pd.notnull(row["family"]) else None,
                "genus": row["genus"] if pd.notnull(row["genus"]) else None,
                "species": row["species"] if pd.notnull(row["species"]) else "Not available in reference data",
                "scientificName": row["scientificName"] if "scientificName" in row and pd.notnull(row["scientificName"]) else None,
            })

        # Sort by distance ascending, then by edit_distance
        matches.sort(key=lambda m: (m["distance"], m["edit_distance"]))

        # Assign ranks
        results = []
        for rank, m in enumerate(matches[:top_k], start=1):
            m["rank"] = rank
            results.append(m)

        return results


# Global singleton instance for easy import and reuse
_SEARCHER_INSTANCE: Optional[ReferenceSearcher] = None

def get_reference_searcher() -> ReferenceSearcher:
    """Obtain or initialize the global ReferenceSearcher singleton."""
    global _SEARCHER_INSTANCE
    if _SEARCHER_INSTANCE is None:
        _SEARCHER_INSTANCE = ReferenceSearcher()
    return _SEARCHER_INSTANCE

def find_similar_sequences(
    sequence: str,
    top_k: int = 5,
    exclude_id: Optional[str] = None
) -> List[Dict]:
    """
    Convenience function: search reference library for nearest barcodes.
    """
    searcher = get_reference_searcher()
    return searcher.search(sequence, top_k=top_k, exclude_id=exclude_id)
