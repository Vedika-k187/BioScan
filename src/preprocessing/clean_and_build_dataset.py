"""
BioScan Phase 1: Data Cleaning & Reference Barcode Construction Pipeline
Module: src/preprocessing/clean_and_build_dataset.py

Loads data/raw/hackathon_dna_taxonomy_merged.csv without modifying it,
validates integrity, normalizes DNA sequences, checks taxonomic consistency,
and constructs the reference barcode and taxonomy tables in data/processed/.
"""

import os
import re
import hashlib
import json
import pandas as pd
import numpy as np

RAW_DATA_PATH = os.path.join("data", "raw", "hackathon_dna_taxonomy_merged.csv")
PROCESSED_DIR = os.path.join("data", "processed")
REF_BARCODES_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
TAXONOMY_REF_PATH = os.path.join(PROCESSED_DIR, "taxonomy_reference.csv")
RAW_SHA256_EXPECTED = "5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65"

def compute_sha256(filepath):
    """Compute SHA256 checksum of a file to verify raw integrity."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest().upper()

def load_and_validate_raw(raw_path=RAW_DATA_PATH):
    """
    Step 1: Load and validate raw dataset schema, data types, and sequence integrity.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    current_hash = compute_sha256(raw_path)
    print(f"[Step 1] Loading raw dataset: {raw_path}")
    print(f"         Raw dataset SHA256: {current_hash}")

    df = pd.read_csv(raw_path, low_memory=False)
    n_rows, n_cols = df.shape
    print(f"         Dimensions: {n_rows} rows, {n_cols} columns")

    # Required column checks
    required_cols = ["dnasequence", "class", "order", "family", "genus", "species"]
    missing_req = [c for c in required_cols if c not in df.columns]
    if missing_req:
        raise ValueError(f"Missing required columns in raw dataset: {missing_req}")

    # Inspect DNA sequences
    dna_series = df["dnasequence"]
    null_dna_count = dna_series.isnull().sum()
    if null_dna_count > 0:
        raise ValueError(f"Found {null_dna_count} null DNA sequences in raw dataset!")

    # Check for formatting and character noise in raw sequences
    has_lowercase = dna_series.apply(lambda s: any(c.islower() for c in str(s))).sum()
    has_leading_trailing_ws = dna_series.apply(lambda s: str(s) != str(s).strip()).sum()
    has_internal_ws = dna_series.apply(lambda s: bool(re.search(r'\s', str(s)))).sum()
    has_gaps = dna_series.apply(lambda s: bool(re.search(r'[-.]', str(s)))).sum()
    
    # Check for non-ACGT or ambiguous IUPAC bases
    acgt_pattern = re.compile(r'^[ACGTacgt]+$')
    non_acgt_count = dna_series.apply(lambda s: not bool(acgt_pattern.match(str(s).strip()))).sum()

    validation_report = {
        "raw_path": raw_path,
        "sha256": current_hash,
        "rows": n_rows,
        "columns": n_cols,
        "null_dna_count": int(null_dna_count),
        "has_lowercase": int(has_lowercase),
        "has_leading_trailing_ws": int(has_leading_trailing_ws),
        "has_internal_ws": int(has_internal_ws),
        "has_gaps": int(has_gaps),
        "non_acgt_count": int(non_acgt_count)
    }

    print(f"         Validation summary:")
    print(f"           - Null sequences: {null_dna_count}")
    print(f"           - Sequences with lowercase characters: {has_lowercase}")
    print(f"           - Sequences with leading/trailing whitespace: {has_leading_trailing_ws}")
    print(f"           - Sequences with internal whitespace: {has_internal_ws}")
    print(f"           - Sequences with gap characters: {has_gaps}")
    print(f"           - Sequences with non-ACGT / ambiguous bases: {non_acgt_count}")

    return df, validation_report

def normalize_sequences(df):
    """
    Step 2: Normalize DNA sequences, preserving raw column and computing compositional stats.
    """
    print("[Step 2] Normalizing DNA sequences and computing compositional metrics...")
    df = df.copy()
    
    # Retain raw sequence as sequence_raw
    df["sequence_raw"] = df["dnasequence"]
    
    # Normalize: uppercase and stripped
    df["sequence_normalized"] = df["dnasequence"].astype(str).str.strip().str.upper()
    
    # Strict validation: all characters must be in {A, C, G, T}
    valid_acgt_mask = df["sequence_normalized"].str.match(r'^[ACGT]+$')
    if not valid_acgt_mask.all():
        invalid_examples = df.loc[~valid_acgt_mask, "sequence_normalized"].head()
        raise ValueError(f"Found invalid sequences post-normalization:\n{invalid_examples}")
        
    # Compositional statistics (Phase 1 scope only, no k-mers)
    df["sequence_length"] = df["sequence_normalized"].str.len()
    df["A_count"] = df["sequence_normalized"].apply(lambda s: s.count('A'))
    df["T_count"] = df["sequence_normalized"].apply(lambda s: s.count('T'))
    df["G_count"] = df["sequence_normalized"].apply(lambda s: s.count('G'))
    df["C_count"] = df["sequence_normalized"].apply(lambda s: s.count('C'))
    
    # GC content calculated as (G + C) / length
    df["GC_content"] = (df["G_count"] + df["C_count"]) / df["sequence_length"]

    print(f"         Normalized {len(df)} rows.")
    print(f"         Sequence lengths range: {df['sequence_length'].min()} - {df['sequence_length'].max()} bp")
    print(f"         Mean GC content: {df['GC_content'].mean():.4f}")
    return df

def verify_taxonomic_consistency(df):
    """
    Step 3 (pre-deduplication check): Verify that repeated sequences map to consistent taxonomy.
    """
    print("[Step 3] Verifying taxonomic consistency across all sequence occurrences...")
    tax_levels = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    
    conflicts = []
    grouped = df.groupby("sequence_normalized")
    
    for seq, group in grouped:
        for lvl in tax_levels:
            if lvl in group.columns:
                # nunique(dropna=False) checks whether values vary, including NaN vs non-NaN
                distinct_vals = group[lvl].drop_duplicates().tolist()
                # If there are multiple distinct values, check if it's a conflict
                if len(distinct_vals) > 1:
                    conflicts.append({
                        "sequence": seq[:25] + "...",
                        "level": lvl,
                        "distinct_values": distinct_vals,
                        "occurrence_count": len(group)
                    })
                    
    if conflicts:
        print(f"CRITICAL ERROR: Found {len(conflicts)} taxonomic conflicts across identical sequences!")
        for c in conflicts[:5]:
            print(f"  Conflict at {c['level']}: {c['distinct_values']}")
        raise ValueError(f"Taxonomic inconsistency detected in raw data: {conflicts}")
        
    print(f"         Consistency check PASSED: All {df['sequence_normalized'].nunique()} unique sequences have 100% deterministic taxonomy.")
    return True

def build_reference_tables(df):
    """
    Step 3, 4, 6: Build reference_barcodes.csv and taxonomy_reference.csv.
    """
    print("[Step 4 & 6] Constructing reference barcode and taxonomy tables...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Calculate occurrence frequency in raw table before deduplication
    occurrence_counts = df["sequence_normalized"].value_counts().to_dict()
    
    # Deduplicate to one row per unique sequence
    # Preserve first row's clean attributes
    ref_df = df.drop_duplicates(subset=["sequence_normalized"]).copy()
    ref_df = ref_df.sort_values(by=["class", "order", "family", "genus", "species", "sequence_normalized"]).reset_index(drop=True)
    
    # Assign a standard stable identifier SEQ_001 to SEQ_101
    ref_df["sequence_id"] = [f"SEQ_{i+1:03d}" for i in range(len(ref_df))]
    ref_df["raw_occurrence_count"] = ref_df["sequence_normalized"].map(occurrence_counts)
    
    # Define exact columns for reference_barcodes.csv
    ref_columns = [
        "sequence_id",
        "sequence_raw",
        "sequence_normalized",
        "sequence_length",
        "A_count",
        "T_count",
        "G_count",
        "C_count",
        "GC_content",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "scientificName",
        "taxonRank",
        "kingdom",
        "phylum",
        "iucnRedListCategory",
        "raw_occurrence_count"
    ]
    
    # Retain only existing relevant columns
    available_ref_cols = [c for c in ref_columns if c in ref_df.columns]
    reference_barcodes = ref_df[available_ref_cols]
    
    # Define exact columns for taxonomy_reference.csv (Step 6)
    taxonomy_columns = [
        "sequence_id",
        "sequence_normalized",
        "class",
        "order",
        "family",
        "genus",
        "species"
    ]
    taxonomy_reference = ref_df[taxonomy_columns]
    
    # Save CSVs
    reference_barcodes.to_csv(REF_BARCODES_PATH, index=False)
    taxonomy_reference.to_csv(TAXONOMY_REF_PATH, index=False)
    
    print(f"         Saved: {REF_BARCODES_PATH} ({len(reference_barcodes)} rows, {len(reference_barcodes.columns)} cols)")
    print(f"         Saved: {TAXONOMY_REF_PATH} ({len(taxonomy_reference)} rows, {len(taxonomy_reference.columns)} cols)")
    
    # Taxonomic completeness audit
    total_seqs = len(ref_df)
    has_sp = ref_df["species"].notnull().sum()
    missing_sp = ref_df["species"].isnull().sum()
    missing_gen = ref_df["genus"].isnull().sum()
    missing_fam = ref_df["family"].isnull().sum()
    missing_ord = ref_df["order"].isnull().sum()
    missing_cls = ref_df["class"].isnull().sum()
    
    completeness_stats = {
        "total_unique_sequences": int(total_seqs),
        "with_species_label": int(has_sp),
        "missing_species": int(missing_sp),
        "missing_genus": int(missing_gen),
        "missing_family": int(missing_fam),
        "missing_order": int(missing_ord),
        "missing_class": int(missing_cls)
    }
    
    print(f"         Taxonomy Completeness Breakdown:")
    print(f"           - With species label: {has_sp} ({has_sp/total_seqs*100:.1f}%)")
    print(f"           - Missing species: {missing_sp} ({missing_sp/total_seqs*100:.1f}%)")
    print(f"           - Missing genus: {missing_gen} ({missing_gen/total_seqs*100:.1f}%)")
    print(f"           - Missing family: {missing_fam} ({missing_fam/total_seqs*100:.1f}%)")
    print(f"           - Missing order: {missing_ord} ({missing_ord/total_seqs*100:.1f}%)")
    print(f"           - Missing class: {missing_cls} ({missing_cls/total_seqs*100:.1f}%)")
    
    return reference_barcodes, taxonomy_reference, completeness_stats

def run_phase1_pipeline():
    """
    Orchestrate full Phase 1 data cleaning pipeline.
    """
    # 1. Load and validate
    raw_df, val_report = load_and_validate_raw()
    
    # 2. Normalize and compute compositional features
    norm_df = normalize_sequences(raw_df)
    
    # 3. Verify consistency
    verify_taxonomic_consistency(norm_df)
    
    # 4. Build output tables
    ref_barcodes, tax_ref, comp_stats = build_reference_tables(norm_df)
    
    # Verify raw file unchanged
    post_hash = compute_sha256(RAW_DATA_PATH)
    assert post_hash == val_report["sha256"], "CRITICAL: Raw dataset was modified during processing!"
    print(f"[Verification] Raw dataset SHA256 integrity confirmed: {post_hash}")
    
    return {
        "validation_report": val_report,
        "completeness_stats": comp_stats,
        "ref_barcodes_shape": ref_barcodes.shape,
        "tax_ref_shape": tax_ref.shape,
        "raw_sha256": post_hash
    }

if __name__ == "__main__":
    summary = run_phase1_pipeline()
    print("\nPhase 1 Data Cleaning Pipeline completed successfully.")
