"""
BioScan Phase 2: DNA Feature Engineering Pipeline
Module: src/features/dna_features.py

Extracts numerical representations from normalized DNA barcode sequences:
1. Basic sequence & nucleotide composition metrics
2. Sequence complexity and entropy metrics
3. Normalized k-mer frequency representations (k = 2, 3, 4, 5)
4. Exports isolated numerical feature tables and metadata.
"""

import os
import json
import math
import itertools
from collections import Counter
import numpy as np
import pandas as pd

REFERENCE_BARCODES_PATH = os.path.join("data", "processed", "reference_barcodes.csv")
PROCESSED_DIR = os.path.join("data", "processed")
METADATA_PATH = os.path.join(PROCESSED_DIR, "feature_metadata.json")
BASES = ["A", "C", "G", "T"]

def generate_all_kmers(k):
    """Generate all possible DNA k-mers for alphabet {A, C, G, T} in lexicographical order."""
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]

def extract_basic_and_complexity_features(sequence):
    """
    Compute basic length, nucleotide counts, frequencies, GC/AT ratios,
    and sequence complexity metrics for a single DNA sequence.
    """
    seq = str(sequence).upper().strip()
    length = len(seq)
    if length == 0:
        raise ValueError("Sequence cannot be empty.")

    a_cnt = seq.count("A")
    c_cnt = seq.count("C")
    g_cnt = seq.count("G")
    t_cnt = seq.count("T")

    # Frequencies
    a_freq = a_cnt / length
    c_freq = c_cnt / length
    g_freq = g_cnt / length
    t_freq = t_cnt / length

    # Composition content
    gc_content = (g_cnt + c_cnt) / length
    at_content = (a_cnt + t_cnt) / length

    # Complexity features
    unique_bases = len(set(seq))
    
    # Shannon entropy of nucleotide composition (log base 2)
    entropy = 0.0
    for freq in [a_freq, c_freq, g_freq, t_freq]:
        if freq > 0:
            entropy -= freq * math.log2(freq)
            
    max_freq = max(a_freq, c_freq, g_freq, t_freq)
    min_freq = min(a_freq, c_freq, g_freq, t_freq)

    # Unique observed k-mers for k=2 and k=3
    k2_observed = len(set(seq[i:i+2] for i in range(length - 1))) if length >= 2 else 0
    k3_observed = len(set(seq[i:i+3] for i in range(length - 2))) if length >= 3 else 0

    return {
        # Basic sequence features (11 features)
        "sequence_length": length,
        "A_count": a_cnt,
        "T_count": t_cnt,
        "G_count": g_cnt,
        "C_count": c_cnt,
        "A_frequency": a_freq,
        "T_frequency": t_freq,
        "G_frequency": g_freq,
        "C_frequency": c_freq,
        "GC_content": gc_content,
        "AT_content": at_content,
        # Complexity features (6 features)
        "unique_base_count": unique_bases,
        "unique_k2_count": k2_observed,
        "unique_k3_count": k3_observed,
        "shannon_entropy": entropy,
        "max_base_frequency": max_freq,
        "min_base_frequency": min_freq,
    }

class KMerEncoder:
    """Reusable k-mer frequency encoder for DNA sequences."""
    def __init__(self, k):
        self.k = k
        self.kmers = generate_all_kmers(k)
        self.kmer_to_idx = {kmer: idx for idx, kmer in enumerate(self.kmers)}
        self.vocab_size = len(self.kmers)

    def encode_sequence(self, sequence):
        """
        Extract normalized k-mer frequency vector for a sequence.
        Sum of frequencies equals 1.0 when len(seq) >= k.
        """
        seq = str(sequence).upper().strip()
        num_windows = len(seq) - self.k + 1
        
        if num_windows <= 0:
            # Sequence shorter than k
            return {kmer: 0.0 for kmer in self.kmers}

        counts = Counter(seq[i:i + self.k] for i in range(num_windows))
        freqs = {kmer: counts.get(kmer, 0) / num_windows for kmer in self.kmers}
        return freqs

    def transform_dataframe(self, df, sequence_col="sequence_normalized", id_col="sequence_id"):
        """Transform a dataframe of sequences into a k-mer frequency DataFrame."""
        rows = []
        for _, row in df.iterrows():
            seq_id = row[id_col]
            seq = row[sequence_col]
            kmer_dict = self.encode_sequence(seq)
            kmer_dict[id_col] = seq_id
            rows.append(kmer_dict)

        # Place sequence_id as the first column
        cols = [id_col] + self.kmers
        res_df = pd.DataFrame(rows)[cols]
        return res_df

def run_feature_engineering_pipeline():
    """
    Execute Phase 2 feature extraction and save feature matrices.
    """
    print(f"[Phase 2] Loading reference barcodes from {REFERENCE_BARCODES_PATH}...")
    if not os.path.exists(REFERENCE_BARCODES_PATH):
        raise FileNotFoundError(f"Missing {REFERENCE_BARCODES_PATH}. Run Phase 1 first.")

    ref_df = pd.read_csv(REFERENCE_BARCODES_PATH)
    n_seqs = len(ref_df)
    print(f"          Loaded {n_seqs} independent reference sequences.")

    # 1. Basic & Complexity features
    print("[Step 1 & 4] Extracting basic composition and complexity features...")
    basic_records = []
    for _, row in ref_df.iterrows():
        feats = extract_basic_and_complexity_features(row["sequence_normalized"])
        feats["sequence_id"] = row["sequence_id"]
        basic_records.append(feats)

    # Reorder columns with sequence_id first
    basic_cols = ["sequence_id"] + [k for k in basic_records[0].keys() if k != "sequence_id"]
    features_basic = pd.DataFrame(basic_records)[basic_cols]
    
    basic_csv_path = os.path.join(PROCESSED_DIR, "features_basic.csv")
    features_basic.to_csv(basic_csv_path, index=False)
    print(f"          Saved: {basic_csv_path} ({features_basic.shape[0]} rows, {features_basic.shape[1] - 1} features)")

    # 2. k-mer features for k = 2, 3, 4, 5
    kmer_dfs = {}
    for k in [2, 3, 4, 5]:
        print(f"[Step 2] Extracting k-mer features for k={k} (vocab={4**k})...")
        encoder = KMerEncoder(k=k)
        k_df = encoder.transform_dataframe(ref_df, sequence_col="sequence_normalized", id_col="sequence_id")
        
        # Validation checks
        assert k_df.shape[0] == n_seqs, f"Row count mismatch for k={k}"
        assert k_df.shape[1] == (4**k + 1), f"Feature dimension mismatch for k={k}: expected {4**k + 1}, got {k_df.shape[1]}"
        
        # Check non-negative and sum to 1.0
        kmer_cols = encoder.kmers
        sums = k_df[kmer_cols].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-5), f"k-mer frequencies do not sum to 1.0 for k={k}"
        assert (k_df[kmer_cols] >= 0).all().all(), f"Negative k-mer frequency detected for k={k}"
        
        k_csv_path = os.path.join(PROCESSED_DIR, f"features_k{k}.csv")
        k_df.to_csv(k_csv_path, index=False)
        print(f"          Saved: {k_csv_path} ({k_df.shape[0]} rows, {len(kmer_cols)} k-mer features)")
        kmer_dfs[k] = k_df

    # 3. Combined matrices (Basic + k)
    combined_dfs = {}
    for k in [2, 3, 4, 5]:
        # Merge on sequence_id
        comb_df = pd.merge(features_basic, kmer_dfs[k], on="sequence_id")
        comb_csv_path = os.path.join(PROCESSED_DIR, f"features_basic_k{k}.csv")
        comb_df.to_csv(comb_csv_path, index=False)
        print(f"          Saved: {comb_csv_path} ({comb_df.shape[0]} rows, {comb_df.shape[1] - 1} combined features)")
        combined_dfs[k] = comb_df

    # 4. Redundancy & Correlation Analysis on Basic Features
    numeric_basic = features_basic.drop(columns=["sequence_id"])
    corr_matrix = numeric_basic.corr()
    
    # Identify high correlations (|r| >= 0.95)
    redundant_pairs = []
    cols = numeric_basic.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= 0.95:
                redundant_pairs.append({
                    "feature_1": cols[i],
                    "feature_2": cols[j],
                    "correlation": float(r)
                })

    # 5. Metadata JSON
    metadata = {
        "dataset_name": "BioScan Reference DNA Feature Library",
        "sample_count": n_seqs,
        "sequence_id_column": "sequence_id",
        "scaling_applied": False,
        "labels_included": False,
        "excluded_columns": [
            "class", "order", "family", "genus", "species", "scientificName",
            "taxonRank", "kingdom", "phylum", "gbifid", "occurrenceID",
            "eventID", "sampname", "latitude", "longitude", "iucnRedListCategory"
        ],
        "feature_sets": {
            "basic": {
                "file": "features_basic.csv",
                "sample_count": n_seqs,
                "feature_count": features_basic.shape[1] - 1,
                "categories": {
                    "composition": ["sequence_length", "A_count", "T_count", "G_count", "C_count",
                                    "A_frequency", "T_frequency", "G_frequency", "C_frequency",
                                    "GC_content", "AT_content"],
                    "complexity": ["unique_base_count", "unique_k2_count", "unique_k3_count",
                                   "shannon_entropy", "max_base_frequency", "min_base_frequency"]
                }
            },
            "k2": {
                "file": "features_k2.csv",
                "sample_count": n_seqs,
                "feature_count": 16,
                "k": 2
            },
            "k3": {
                "file": "features_k3.csv",
                "sample_count": n_seqs,
                "feature_count": 64,
                "k": 3
            },
            "k4": {
                "file": "features_k4.csv",
                "sample_count": n_seqs,
                "feature_count": 256,
                "k": 4
            },
            "k5": {
                "file": "features_k5.csv",
                "sample_count": n_seqs,
                "feature_count": 1024,
                "k": 5
            },
            "basic_k2": {
                "file": "features_basic_k2.csv",
                "sample_count": n_seqs,
                "feature_count": combined_dfs[2].shape[1] - 1
            },
            "basic_k3": {
                "file": "features_basic_k3.csv",
                "sample_count": n_seqs,
                "feature_count": combined_dfs[3].shape[1] - 1
            },
            "basic_k4": {
                "file": "features_basic_k4.csv",
                "sample_count": n_seqs,
                "feature_count": combined_dfs[4].shape[1] - 1
            },
            "basic_k5": {
                "file": "features_basic_k5.csv",
                "sample_count": n_seqs,
                "feature_count": combined_dfs[5].shape[1] - 1
            }
        },
        "redundancy_analysis": {
            "mathematical_identities": [
                "A_frequency + T_frequency + G_frequency + C_frequency == 1.0 (Sum-to-one constraint)",
                "GC_content + AT_content == 1.0 (Perfect collinearity, r = -1.0)",
                "GC_content == G_frequency + C_frequency",
                "AT_content == A_frequency + T_frequency"
            ],
            "high_correlation_pairs": redundant_pairs
        }
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"          Saved metadata to: {METADATA_PATH}")

    return {
        "n_samples": n_seqs,
        "features_basic_shape": features_basic.shape,
        "k2_shape": kmer_dfs[2].shape,
        "k3_shape": kmer_dfs[3].shape,
        "k4_shape": kmer_dfs[4].shape,
        "k5_shape": kmer_dfs[5].shape,
        "redundant_pairs_count": len(redundant_pairs)
    }

if __name__ == "__main__":
    res = run_feature_engineering_pipeline()
    print("\nPhase 2 Feature Engineering Pipeline execution complete.")
