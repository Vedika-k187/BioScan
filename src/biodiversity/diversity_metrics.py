"""
src/biodiversity/diversity_metrics.py
Phase 7: Scientific Diversity Indices Module (Shannon, Simpson, Pielou)

Defines explicit observational units and avoids artificial inflation from duplicate survey rows:
- Shannon Diversity: H = -sum(p_i * ln(p_i))
- Gini-Simpson Index: 1 - sum(p_i^2)
- Pielou's Evenness: J = H / ln(S)

Includes deterministic mathematical functions tested against known theoretical values.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

def calculate_shannon_diversity(
    counts_or_props: Union[List[float], np.ndarray, pd.Series],
    base: float = math.e
) -> float:
    """
    Calculate Shannon Diversity Index (H) from observed taxon counts or proportions.

    H = - sum_{i} (p_i * log_b(p_i))

    Edge cases:
    - Empty input or all zeros -> returns 0.0
    - Single taxon -> returns 0.0 (zero uncertainty / zero diversity)
    - Equal proportions of S taxa -> returns log_b(S)
    """
    if counts_or_props is None:
        return 0.0

    arr = np.array(counts_or_props, dtype=float)
    if len(arr) == 0:
        return 0.0

    # Filter out non-positive entries
    arr = arr[arr > 0]
    if len(arr) == 0 or len(arr) == 1:
        return 0.0

    total = np.sum(arr)
    if total <= 0:
        return 0.0

    p = arr / total
    
    if base == math.e:
        h = -np.sum(p * np.log(p))
    elif base == 2.0:
        h = -np.sum(p * np.log2(p))
    elif base == 10.0:
        h = -np.sum(p * np.log10(p))
    else:
        h = -np.sum(p * (np.log(p) / math.log(base)))

    return float(round(h, 6))


def calculate_simpson_diversity(
    counts_or_props: Union[List[float], np.ndarray, pd.Series]
) -> float:
    """
    Calculate Gini-Simpson Index: 1 - sum(p_i^2).
    Represents probability that two randomly sampled observations belong to different taxa.
    """
    if counts_or_props is None:
        return 0.0

    arr = np.array(counts_or_props, dtype=float)
    arr = arr[arr > 0]
    if len(arr) <= 1:
        return 0.0

    total = np.sum(arr)
    if total <= 0:
        return 0.0

    p = arr / total
    simpson = 1.0 - np.sum(p ** 2)
    return float(round(simpson, 6))


def calculate_pielou_evenness(
    shannon_h: float,
    richness_s: int,
    base: float = math.e
) -> float:
    """
    Calculate Pielou's Evenness (J):
    J = H / H_max = H / log_b(S)

    Range: [0.0, 1.0]
    """
    if richness_s <= 1 or shannon_h <= 0.0:
        return 0.0

    h_max = math.log(richness_s) if base == math.e else (math.log(richness_s) / math.log(base))
    if h_max <= 0:
        return 0.0

    j = shannon_h / h_max
    return float(round(min(1.0, max(0.0, j)), 6))


def compute_station_diversity_metrics(
    records_df: pd.DataFrame,
    tax_level: str = "species",
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Compute station-level ecological diversity metrics across the 36 sampling stations.

    Methodological Pipeline:
    1. Input: Deduplicated field occurrence records (biodiversity_reference_records.csv, N=1,604),
       where technical primer replicates (AF vs AFS) were averaged for each occurrence.
    2. Station Definition: A unique geographic sampling station defined by its (decimalLatitude, decimalLongitude)
       coordinate pair (36 distinct stations).
    3. Multi-Sample Aggregation within Station:
       At 32 of 36 stations, two depth strata were sampled: Surface (S, depth=1.0 m) and Bottom (B, seafloor).
       Occurrence records from both depth samples at the same station are pooled together.
    4. Taxon Abundance Calculation:
       For each taxon i present at that station, its abundance A_i is calculated as the sum of mean read counts
       across occurrences at that station:
           A_i = sum_{j in station occurrences} organismQuantity_{i, j}
    5. Relative Abundance (p_i):
           p_i = A_i / sum_k A_k
    6. Shannon Diversity Indices:
       - Abundance-weighted: H_reads = - sum_{i=1}^S p_i * ln(p_i)
       - Incidence-based:    H_incidence = - sum_{i=1}^S q_i * ln(q_i)
         where q_i = c_i / sum_k c_k (c_i is detection count across samples at that station).

    Taxonomic Levels Supported (tax_level):
    - 'species' (default): Evaluates all detected lineages; named species use their binomen, while
      unlabeled barcodes are assigned to their distinct lineage ('{genus} sp.' or '{family} sp.').
    - 'named_species_only': Excludes species-unlabeled records; calculates diversity strictly on named species.
    - 'genus': Groups all detections by Genus rank.
    - 'family': Groups all detections by Family rank.

    Zero Detections:
    Taxa with 0 positive detections at that station have A_i = 0, contributing 0 to the summation,
    and are not included in richness S.
    """
    df = records_df.copy()

    # Define taxon identity column
    if tax_level == "species":
        # All detected lineages: named species keep their binomen;
        # records lacking species name are assigned to their distinct lineage '{genus} sp.' or '{family} sp.'
        def get_tax_id(row):
            sp = row.get("species")
            if pd.notnull(sp) and sp != "" and sp != "Not available in reference data":
                return sp
            gen = row.get("genus")
            if pd.notnull(gen) and gen != "" and gen != "Not available in reference data":
                return f"{gen} sp."
            fam = row.get("family")
            return f"{fam} sp." if pd.notnull(fam) and fam != "" else "Unassigned Teleostei"

        df["taxon_eval"] = df.apply(get_tax_id, axis=1)
    elif tax_level == "named_species_only":
        valid_sp = df["species"].notnull() & (df["species"] != "") & (df["species"] != "Not available in reference data")
        df = df[valid_sp].copy()
        df["taxon_eval"] = df["species"]
    else:
        df["taxon_eval"] = df[tax_level].fillna(f"Unassigned_{tax_level}")

    station_metrics = []
    # Group by geographic coordinates (36 unique stations)
    grouped = df.groupby(["decimalLatitude", "decimalLongitude"])

    for (lat, lon), group in grouped:
        station_name = group["sampname"].iloc[0]
        locality = group["locality"].iloc[0]
        depth = float(group["depth"].mean())
        n_occurrences = len(group)

        # Taxon abundance (sum of reads per taxon at this station)
        taxon_reads = group.groupby("taxon_eval")["organismQuantity"].sum()
        richness = len(taxon_reads)
        
        # Diversity using relative read counts
        h_reads = calculate_shannon_diversity(taxon_reads.values)
        simpson_reads = calculate_simpson_diversity(taxon_reads.values)
        evenness_reads = calculate_pielou_evenness(h_reads, richness)

        # Diversity using incidence / occurrence count
        taxon_inc = group.groupby("taxon_eval")["record_id"].count()
        h_inc = calculate_shannon_diversity(taxon_inc.values)

        dominant_taxon = taxon_reads.idxmax() if not taxon_reads.empty else "None"
        dominant_share_pct = round(float(taxon_reads.max() / taxon_reads.sum() * 100), 1) if taxon_reads.sum() > 0 else 0.0

        station_metrics.append({
            "station_latitude": lat,
            "station_longitude": lon,
            "station_name": station_name,
            "depth_meters": round(depth, 1),
            "taxonomic_rank_evaluated": tax_level,
            "detection_records_count": n_occurrences,
            "taxonomic_richness_S": richness,
            "shannon_diversity_H_reads": h_reads,
            "shannon_diversity_H_incidence": h_inc,
            "simpson_diversity_1_minus_D": simpson_reads,
            "pielou_evenness_J": evenness_reads,
            "dominant_taxon": dominant_taxon,
            "dominant_taxon_share_pct": dominant_share_pct
        })

    metrics_df = pd.DataFrame(station_metrics).sort_values(
        by="shannon_diversity_H_reads",
        ascending=False
    ).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        metrics_df.to_csv(output_path, index=False)

    return metrics_df
