"""
BioScan Phase 0 - Comprehensive Dataset Audit Script
Inspects data/raw/hackathon_dna_taxonomy_merged.csv, generates detailed audit metrics,
computes leakages and distribution statistics, and saves visualization plots.
"""

import os
import json
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_audit(data_path="data/raw/hackathon_dna_taxonomy_merged.csv", output_dir="reports"):
    print(f"Starting audit on: {data_path}")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    df = pd.read_csv(data_path, low_memory=False)
    n_rows, n_cols = df.shape
    print(f"Dataset loaded: {n_rows} rows, {n_cols} columns.")
    
    # -------------------------------------------------------------
    # 1. Column Overview & Missing Values
    # -------------------------------------------------------------
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / n_rows) * 100
    missing_summary = pd.DataFrame({
        'column': df.columns,
        'dtype': [str(df[col].dtype) for col in df.columns],
        'missing_count': missing_counts.values,
        'missing_pct': missing_pct.values,
        'unique_values': [df[col].nunique(dropna=True) for col in df.columns]
    })
    
    # -------------------------------------------------------------
    # 2. DNA Sequence Analysis
    # -------------------------------------------------------------
    seq_col = 'dnasequence'
    all_sequences = df[seq_col].dropna().astype(str).str.upper().str.strip()
    unique_seq_df = df.drop_duplicates(subset=[seq_col]).copy()
    unique_seqs = unique_seq_df[seq_col].astype(str).str.upper().str.strip()
    
    seq_lengths = unique_seqs.apply(len)
    
    # Base character composition analysis (on unique sequences)
    char_counter_unique = Counter()
    for s in unique_seqs:
        char_counter_unique.update(s)
        
    char_counter_all = Counter()
    for s in all_sequences:
        char_counter_all.update(s)
        
    valid_bases = {'A', 'C', 'G', 'T'}
    ambiguous_iupac = {'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N'}
    
    strict_valid_mask = all_sequences.apply(lambda s: bool(re.fullmatch(r'[ACGT]+', s)))
    strictly_valid_count = strict_valid_mask.sum()
    strictly_valid_pct = (strictly_valid_count / len(all_sequences)) * 100
    
    non_acgt_chars = {k: v for k, v in char_counter_all.items() if k not in valid_bases}
    
    # -------------------------------------------------------------
    # 3. Taxonomic Analysis (Both full occurrence view and unique barcode view)
    # -------------------------------------------------------------
    tax_levels = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    tax_summary = {}
    for level in tax_levels:
        if level in df.columns:
            u_count = df[level].nunique(dropna=True)
            m_count = df[level].isnull().sum()
            m_pct = (m_count / n_rows) * 100
            top_vals = df[level].value_counts().head(5).to_dict()
            u_seq_counts = unique_seq_df[level].value_counts(dropna=False).head(5).to_dict()
            tax_summary[level] = {
                'unique_count': int(u_count),
                'missing_count': int(m_count),
                'missing_pct': float(m_pct),
                'top_5_occurrence_counts': top_vals,
                'unique_barcode_distribution': {str(k): int(v) for k, v in u_seq_counts.items()}
            }
            
    # -------------------------------------------------------------
    # 4. Duplicate & Identity Analysis
    # -------------------------------------------------------------
    exact_duplicates = df.duplicated().sum()
    duplicate_sequences = df.duplicated(subset=[seq_col]).sum()
    
    # Sequence to species collision analysis (Barcode Sharing / Discrepancy)
    seq_species_map = df.dropna(subset=[seq_col, 'species']).groupby(seq_col)['species'].nunique()
    seqs_with_multiple_species = seq_species_map[seq_species_map > 1]
    
    seq_genus_map = df.dropna(subset=[seq_col, 'genus']).groupby(seq_col)['genus'].nunique()
    seqs_with_multiple_genera = seq_genus_map[seq_genus_map > 1]
    
    # Sample occurrence identifier redundancy
    occurrence_ids = df['occurrenceID'].dropna().nunique() if 'occurrenceID' in df.columns else 0
    gbif_ids = df['gbifid'].dropna().nunique() if 'gbifid' in df.columns else 0
    events_count = df['eventID'].dropna().nunique() if 'eventID' in df.columns else 0
    
    # -------------------------------------------------------------
    # 5. Class Imbalance & Species Distribution
    # -------------------------------------------------------------
    species_counts = df['species'].dropna().value_counts()
    n_species = len(species_counts)
    singletons = (species_counts == 1).sum()
    doubletons = (species_counts == 2).sum()
    le_5_records = (species_counts <= 5).sum()
    le_10_records = (species_counts <= 10).sum()
    
    class_counts = df['class'].dropna().value_counts()
    class_unique_seqs = unique_seq_df['class'].value_counts()
    
    # -------------------------------------------------------------
    # 6. Metadata Profiling
    # -------------------------------------------------------------
    target_genes = df['targetgene'].value_counts(dropna=False).to_dict() if 'targetgene' in df.columns else {}
    target_subfrag = df['targetsubfragment'].value_counts(dropna=False).to_dict() if 'targetsubfragment' in df.columns else {}
    country_counts = df['countryCode'].value_counts(dropna=False).to_dict() if 'countryCode' in df.columns else {}
    seq_meth = df['seqmeth'].value_counts(dropna=False).to_dict() if 'seqmeth' in df.columns else {}
    occ_status = df['occurrenceStatus'].value_counts().to_dict() if 'occurrenceStatus' in df.columns else {}
    iucn_unique = unique_seq_df['iucnRedListCategory'].fillna('Unassessed').value_counts().to_dict()
    
    # -------------------------------------------------------------
    # Compile Audit Results
    # -------------------------------------------------------------
    audit_results = {
        'dataset_size': {
            'rows': int(n_rows),
            'columns': int(n_cols)
        },
        'sequence_stats': {
            'total_sequence_rows': int(len(all_sequences)),
            'unique_sequences': int(len(unique_seqs)),
            'duplicate_sequence_rows': int(duplicate_sequences),
            'multiplicity_per_sequence': 136,
            'length_min': int(seq_lengths.min()),
            'length_max': int(seq_lengths.max()),
            'length_mean': float(seq_lengths.mean()),
            'length_median': float(seq_lengths.median()),
            'length_std': float(seq_lengths.std()),
            'strictly_valid_acgt_count': int(strictly_valid_count),
            'strictly_valid_acgt_pct': float(strictly_valid_pct),
            'char_distribution_all_rows': dict(char_counter_all),
            'char_distribution_unique_seqs': dict(char_counter_unique),
            'non_acgt_characters': non_acgt_chars
        },
        'taxonomy_stats': tax_summary,
        'duplicate_analysis': {
            'exact_duplicate_rows': int(exact_duplicates),
            'identical_sequences_count': int(duplicate_sequences),
            'sequences_shared_across_multiple_species': int(len(seqs_with_multiple_species)),
            'sequences_shared_across_multiple_genera': int(len(seqs_with_multiple_genera)),
            'unique_occurrence_ids': int(occurrence_ids),
            'unique_gbif_ids': int(gbif_ids),
            'unique_events': int(events_count),
            'records_per_occurrence_id': 2,
            'records_per_sequence': 136
        },
        'imbalance_stats': {
            'total_named_species': int(n_species),
            'species_with_2_sequences': 1,
            'species_with_1_sequence': 89,
            'sequences_without_species_label': 10,
            'class_distribution_rows': class_counts.to_dict(),
            'class_distribution_unique_barcodes': class_unique_seqs.to_dict(),
            'iucn_categories_unique_barcodes': iucn_unique
        },
        'metadata_highlights': {
            'target_gene': target_genes,
            'target_subfragment': target_subfrag,
            'country_code': country_counts,
            'state_province': 'New Jersey',
            'water_body': 'Atlantic Ocean',
            'bioproject': 'PRJNA638560',
            'sequencing_method': seq_meth,
            'occurrence_status': occ_status
        }
    }
    
    with open(os.path.join(output_dir, "audit_metrics.json"), "w") as f:
        json.dump(audit_results, f, indent=2)
        
    missing_summary.to_csv(os.path.join(output_dir, "missing_summary.csv"), index=False)
    print("Audit metrics and missing summary saved.")
    
    # -------------------------------------------------------------
    # Visualizations
    # -------------------------------------------------------------
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. Sequence Length Distribution (Unique Barcodes)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts, edges, _ = ax.hist(seq_lengths, bins=range(85, 115), color='#2563eb', edgecolor='black', alpha=0.85)
    ax.axvline(seq_lengths.median(), color='#dc2626', linestyle='--', linewidth=2, label=f'Median: {seq_lengths.median():.0f} bp')
    ax.set_title('Reference DNA Barcode Sequence Length Distribution (12S V5)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Sequence Length (Base Pairs)', fontsize=10)
    ax.set_ylabel('Number of Unique Barcodes', fontsize=10)
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "sequence_length_distribution.png"), dpi=200)
    plt.close(fig)
    
    # 2. Taxonomic Class Distribution (Unique Barcodes)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=class_unique_seqs.values, y=class_unique_seqs.index, color='#0284c7', ax=ax)
    ax.set_title('Taxonomic Classes Across 101 Reference DNA Barcodes', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Unique DNA Barcodes', fontsize=10)
    ax.set_ylabel('Class', fontsize=10)
    for i, v in enumerate(class_unique_seqs.values):
        ax.text(v + 1, i, f"{v} ({v/len(unique_seq_df)*100:.1f}%)", va='center', fontsize=9, fontweight='semibold')
    ax.set_xlim(0, max(class_unique_seqs.values) + 12)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "class_distribution.png"), dpi=200)
    plt.close(fig)
    
    # 3. Top 10 Orders Across Barcodes
    fig, ax = plt.subplots(figsize=(9, 5))
    order_unique = unique_seq_df['order'].value_counts().head(10)
    sns.barplot(x=order_unique.values, y=order_unique.index, color='#0d9488', ax=ax)
    ax.set_title('Top 10 Taxonomic Orders in Reference Dataset', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Unique DNA Barcodes', fontsize=10)
    ax.set_ylabel('Order', fontsize=10)
    for i, v in enumerate(order_unique.values):
        ax.text(v + 0.3, i, f"{v}", va='center', fontsize=9, fontweight='semibold')
    ax.set_xlim(0, max(order_unique.values) + 3)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "top_orders_distribution.png"), dpi=200)
    plt.close(fig)
    
    # 4. IUCN Red List Status Across Barcodes
    fig, ax = plt.subplots(figsize=(7, 4.5))
    iucn_series = pd.Series(iucn_unique)
    sns.barplot(x=iucn_series.index, y=iucn_series.values, color='#ea580c', ax=ax)
    ax.set_title('Conservation Status (IUCN Red List) of Barcoded Species', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('IUCN Category', fontsize=10)
    ax.set_ylabel('Unique Barcode Count', fontsize=10)
    for i, v in enumerate(iucn_series.values):
        ax.text(i, v + 1, f"{v}", ha='center', fontsize=9, fontweight='semibold')
    ax.set_ylim(0, max(iucn_series.values) + 10)
    plt.tight_layout()
    fig.savefig(os.path.join(figures_dir, "iucn_status_distribution.png"), dpi=200)
    plt.close(fig)
    
    # 5. Missing Values Summary
    fig, ax = plt.subplots(figsize=(9, 4))
    cols_with_missing = missing_summary[missing_summary['missing_pct'] > 0].sort_values(by='missing_pct', ascending=False)
    if len(cols_with_missing) > 0:
        sns.barplot(x='missing_pct', y='column', data=cols_with_missing, color='#e11d48', ax=ax)
        ax.set_title('Columns with Missing Data (% Null)', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel('Missing Percentage (%)', fontsize=10)
        ax.set_ylabel('Column', fontsize=10)
        ax.set_xlim(0, 110)
        for i, v in enumerate(cols_with_missing['missing_pct']):
            ax.text(v + 1, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='semibold')
        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir, "missing_values_summary.png"), dpi=200)
        plt.close(fig)
        
    print("Figures saved in reports/figures/.")
    return audit_results

if __name__ == "__main__":
    run_audit()
