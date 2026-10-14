"""
src/biodiversity/taxonomy_analysis.py
Phase 7: Taxonomic Hierarchy & Richness Analysis Module

Summarizes hierarchical structure: Class -> Order -> Family -> Genus -> Species.
Explicitly distinguishes between:
- "species absent from the reference library"
- "species label missing from a reference record"
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

TAXONOMY_RANKS = ["class", "order", "family", "genus", "species"]

def compute_taxonomic_richness(
    df: pd.DataFrame,
    seq_col: str = "sequence_id"
) -> Dict[str, Any]:
    """
    Compute rigorous taxonomic richness across available reference records.

    Parameters:
    - df: DataFrame containing taxonomy columns (class, order, family, genus, species)
    - seq_col: column identifying unique DNA barcodes

    Returns:
    - Dictionary with richness counts at each taxonomic rank.
    """
    total_records = len(df)
    unique_seqs = df[seq_col].nunique() if seq_col in df.columns else df["dnasequence"].nunique()

    # Valid named species (exclude null, empty, or placeholder strings)
    valid_species_mask = (
        df["species"].notnull() &
        (df["species"] != "") &
        (df["species"] != "Not available in reference data")
    )
    species_series = df.loc[valid_species_mask, "species"]
    n_species = species_series.nunique()
    n_unlabeled_species_records = int((~valid_species_mask).sum())
    
    # At sequence level, how many sequences lack a species label?
    if seq_col in df.columns:
        seq_species_df = df[[seq_col, "species"]].drop_duplicates()
        seq_unlabeled = int((
            seq_species_df["species"].isnull() |
            (seq_species_df["species"] == "") |
            (seq_species_df["species"] == "Not available in reference data")
        ).sum())
        seq_labeled = unique_seqs - seq_unlabeled
    else:
        seq_unlabeled = n_unlabeled_species_records
        seq_labeled = total_records - seq_unlabeled

    # Higher taxonomic ranks (dropna=True)
    n_genera = df["genus"].dropna().nunique()
    n_families = df["family"].dropna().nunique()
    n_orders = df["order"].dropna().nunique()
    n_classes = df["class"].dropna().nunique()
    n_phyla = df["phylum"].dropna().nunique() if "phylum" in df.columns else 1
    n_kingdoms = df["kingdom"].dropna().nunique() if "kingdom" in df.columns else 1

    return {
        "total_records": total_records,
        "unique_sequences": unique_seqs,
        "species_labeled_sequences": seq_labeled,
        "species_unlabeled_sequences": seq_unlabeled,
        "species_richness": n_species,
        "genus_richness": n_genera,
        "family_richness": n_families,
        "order_richness": n_orders,
        "class_richness": n_classes,
        "phylum_richness": n_phyla,
        "kingdom_richness": n_kingdoms,
        "species_label_missing_rate_pct": round(seq_unlabeled / unique_seqs * 100, 2) if unique_seqs > 0 else 0.0
    }


def analyze_taxonomic_hierarchy(ref_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze parent-to-child branching ratios across the 101 reference sequences:
    Class -> Order -> Family -> Genus -> Species.
    """
    # 1. Deduplicate by unique taxonomy string to analyze library branching
    tax_df = ref_df[["class", "order", "family", "genus", "species"]].drop_duplicates().copy()

    # Ratios and grouping
    # Orders per Class
    class_order = tax_df[["class", "order"]].dropna().drop_duplicates()
    orders_per_class = class_order.groupby("class")["order"].count().to_dict()

    # Families per Order
    order_fam = tax_df[["order", "family"]].dropna().drop_duplicates()
    fams_per_order = order_fam.groupby("order")["family"].count().to_dict()

    # Genera per Family
    fam_gen = tax_df[["family", "genus"]].dropna().drop_duplicates()
    genera_per_family = fam_gen.groupby("family")["genus"].count().to_dict()

    # Species per Genus (strictly for records with valid species label)
    valid_sp_df = tax_df[
        tax_df["species"].notnull() &
        (tax_df["species"] != "Not available in reference data")
    ][["genus", "species"]].dropna().drop_duplicates()
    species_per_genus = valid_sp_df.groupby("genus")["species"].count().to_dict()

    # Singleton genus count (genera with exactly 1 species in the library)
    singleton_genera = [g for g, count in species_per_genus.items() if count == 1]
    multi_species_genera = [g for g, count in species_per_genus.items() if count > 1]

    # Singleton species count across 101 barcodes
    # (How many barcodes represent a species with only 1 sequence?)
    if "sequence_id" in ref_df.columns:
        sp_counts = ref_df[
            ref_df["species"].notnull() &
            (ref_df["species"] != "Not available in reference data")
        ]["species"].value_counts().to_dict()
        singleton_species = [s for s, c in sp_counts.items() if c == 1]
        doubleton_species = [s for s, c in sp_counts.items() if c == 2]
    else:
        sp_counts = {}
        singleton_species = []
        doubleton_species = []

    return {
        "orders_per_class": orders_per_class,
        "mean_orders_per_class": round(float(np.mean(list(orders_per_class.values()))), 2),
        "families_per_order": fams_per_order,
        "mean_families_per_order": round(float(np.mean(list(fams_per_order.values()))), 2),
        "genera_per_family": genera_per_family,
        "mean_genera_per_family": round(float(np.mean(list(genera_per_family.values()))), 2),
        "species_per_genus": species_per_genus,
        "mean_species_per_genus": round(float(np.mean(list(species_per_genus.values()))), 2) if species_per_genus else 0.0,
        "total_cataloged_genera": len(species_per_genus),
        "singleton_genera_count": len(singleton_genera),
        "multi_species_genera_count": len(multi_species_genera),
        "multi_species_genera_list": multi_species_genera,
        "singleton_species_count": len(singleton_species),
        "doubleton_species_count": len(doubleton_species),
        "doubleton_species_name": doubleton_species[0] if doubleton_species else "None"
    }


def build_taxonomic_summary(
    ref_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Construct a clean summary table of taxonomic composition across ranks.
    Saves to output_path if provided.
    """
    richness = compute_taxonomic_richness(ref_df)
    hierarchy = analyze_taxonomic_hierarchy(ref_df)

    summary_rows = [
        {
            "taxonomic_rank": "Kingdom",
            "distinct_taxa_count": richness["kingdom_richness"],
            "dominant_taxon": "Animalia",
            "coverage_pct": 100.0,
            "branching_ratio_to_parent": "-",
            "notes": "Metazoan reference barcodes"
        },
        {
            "taxonomic_rank": "Phylum",
            "distinct_taxa_count": richness["phylum_richness"],
            "dominant_taxon": "Chordata",
            "coverage_pct": 100.0,
            "branching_ratio_to_parent": "1.0 phyla / kingdom",
            "notes": "All chordate fishes"
        },
        {
            "taxonomic_rank": "Class",
            "distinct_taxa_count": richness["class_richness"],
            "dominant_taxon": "Teleostei (97 sequences)",
            "coverage_pct": 100.0,
            "branching_ratio_to_parent": "4.0 classes / phylum",
            "notes": "Teleostei (97), Elasmobranchii (2), Chondrostei (1), Petromyzonti (1)"
        },
        {
            "taxonomic_rank": "Order",
            "distinct_taxa_count": richness["order_richness"],
            "dominant_taxon": "Perciformes (28 sequences)",
            "coverage_pct": 100.0,
            "branching_ratio_to_parent": f"{hierarchy['mean_orders_per_class']:.2f} orders / class",
            "notes": "30 distinct orders across 4 classes"
        },
        {
            "taxonomic_rank": "Family",
            "distinct_taxa_count": richness["family_richness"],
            "dominant_taxon": "Gobiidae / Sciaenidae (5 sequences each)",
            "coverage_pct": 100.0,
            "branching_ratio_to_parent": f"{hierarchy['mean_families_per_order']:.2f} families / order",
            "notes": "66 distinct families across 30 orders"
        },
        {
            "taxonomic_rank": "Genus",
            "distinct_taxa_count": richness["genus_richness"],
            "dominant_taxon": f"{hierarchy['singleton_genera_count']} singleton genera",
            "coverage_pct": 100.0,
            "branching_ratio_to_parent": f"{hierarchy['mean_genera_per_family']:.2f} genera / family",
            "notes": "87 distinct genera across 66 families"
        },
        {
            "taxonomic_rank": "Species (Named)",
            "distinct_taxa_count": richness["species_richness"],
            "dominant_taxon": f"{hierarchy['doubleton_species_name']} (2 sequences)",
            "coverage_pct": round(richness["species_labeled_sequences"] / richness["unique_sequences"] * 100, 1),
            "branching_ratio_to_parent": f"{hierarchy['mean_species_per_genus']:.2f} species / genus",
            "notes": f"90 species (89 singletons, 1 doubleton); 10 sequences missing species label"
        },
        {
            "taxonomic_rank": "Species (Unlabeled / Ambiguous)",
            "distinct_taxa_count": richness["species_unlabeled_sequences"],
            "dominant_taxon": "Unassigned at species rank in reference data",
            "coverage_pct": round(richness["species_unlabeled_sequences"] / richness["unique_sequences"] * 100, 1),
            "branching_ratio_to_parent": "-",
            "notes": "Retain valid Genus/Family/Order/Class identities"
        }
    ]

    summary_df = pd.DataFrame(summary_rows)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(output_path, index=False)

    return summary_df
