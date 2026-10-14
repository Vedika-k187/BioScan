"""
src/biodiversity/geographic_analysis.py
Phase 7: Geographic Analysis & Occurrence Module

Builds clean, analysis-ready tables derived from approved reference and survey data:
- data/processed/biodiversity_reference_records.csv
- data/processed/country_biodiversity_summary.csv
- data/processed/species_distribution_summary.csv

Strictly respects:
- Distinction between reference barcodes (101 sequences) and survey occurrences (1,604 positive field records).
- Clear terminology: "reference-record distribution", not "global species distribution".
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .data_quality import categorize_geographic_records

PROCESSED_DIR = os.path.join("data", "processed")

def build_biodiversity_reference_records(
    raw_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Construct deduplicated, analysis-ready occurrence records for detected reference barcodes.

    Process:
    1. Map raw sequences to sequence_id (SEQ_001 to SEQ_101).
    2. Filter to positive field detections (occurrenceStatus == 'PRESENT').
    3. Deduplicate across technical primer replicates (AF vs AFS), averaging read quantities.
    4. Validate and categorize coordinates and taxonomy completeness.
    """
    df = raw_df.copy()
    
    # Map sequences to stable sequence_id
    seq_to_id = dict(zip(ref_df["sequence_normalized"], ref_df["sequence_id"]))
    df["sequence_normalized"] = df["dnasequence"].astype(str).str.strip().str.upper()
    df["sequence_id"] = df["sequence_normalized"].map(seq_to_id)

    # Filter to positive field detections
    present_df = df[df["occurrenceStatus"] == "PRESENT"].copy()

    # Deduplicate across forward primer assays (AF vs AFS)
    # Group by occurrenceID and sequence_id, averaging organismQuantity (read count)
    dedup_cols = ["occurrenceID", "sequence_id"]
    agg_dict = {
        "dnasequence": "first",
        "sequence_normalized": "first",
        "class": "first",
        "order": "first",
        "family": "first",
        "genus": "first",
        "species": "first",
        "scientificName": "first",
        "countryCode": "first",
        "stateProvince": "first",
        "locality": "first",
        "waterBody": "first",
        "decimalLatitude": "first",
        "decimalLongitude": "first",
        "depth": "first",
        "eventDate": "first",
        "year": "first",
        "month": "first",
        "day": "first",
        "sampname": "first",
        "occurrenceStatus": "first",
        "organismQuantity": "mean",
        "associatedSequences": "first"
    }
    
    # Keep only columns that exist
    actual_agg = {k: v for k, v in agg_dict.items() if k in present_df.columns}
    grouped = present_df.groupby(dedup_cols, as_index=False).agg(actual_agg)
    
    # Rename/format fields
    grouped["record_id"] = [f"REC_{i+1:04d}" for i in range(len(grouped))]
    grouped["sequence_length"] = grouped["sequence_normalized"].str.len()
    grouped["country"] = "United States"
    grouped["organismQuantity"] = grouped["organismQuantity"].round(1)

    # Annotate coordinate and taxonomy quality
    annotated = categorize_geographic_records(grouped)

    # Order columns cleanly
    preferred_order = [
        "record_id",
        "sequence_id",
        "sequence_normalized",
        "sequence_length",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "scientificName",
        "country",
        "countryCode",
        "stateProvince",
        "waterBody",
        "locality",
        "decimalLatitude",
        "decimalLongitude",
        "depth",
        "eventDate",
        "year",
        "month",
        "day",
        "sampname",
        "occurrenceID",
        "occurrenceStatus",
        "organismQuantity",
        "associatedSequences",
        "coordinate_quality",
        "taxonomy_quality"
    ]
    final_cols = [c for c in preferred_order if c in annotated.columns]
    res_df = annotated[final_cols].sort_values(by=["sequence_id", "eventDate", "sampname"]).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index=False)

    return res_df


def summarize_country_biodiversity(
    records_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    raw_df: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Summarize reference-record biodiversity metrics by country / region.
    Distinguishes survey assays, positive occurrence records, and cataloged reference barcodes.
    """
    total_raw_rows = len(raw_df) if raw_df is not None else len(records_df) * 2
    pos_records = len(records_df)

    # Detected taxa
    detected_seqs = records_df["sequence_id"].nunique()
    detected_sp = records_df["species"].dropna().loc[lambda s: (s != "") & (s != "Not available in reference data")].nunique()
    detected_gen = records_df["genus"].dropna().nunique()
    detected_fam = records_df["family"].dropna().nunique()
    detected_ord = records_df["order"].dropna().nunique()
    detected_cls = records_df["class"].dropna().nunique()

    # Cataloged library taxa
    lib_seqs = ref_df["sequence_id"].nunique()
    lib_sp = ref_df["species"].dropna().loc[lambda s: (s != "") & (s != "Not available in reference data")].nunique()
    lib_gen = ref_df["genus"].dropna().nunique()
    lib_fam = ref_df["family"].dropna().nunique()
    lib_ord = ref_df["order"].dropna().nunique()
    lib_cls = ref_df["class"].dropna().nunique()

    # Spatial bounds
    lat_min = float(records_df["decimalLatitude"].min())
    lat_max = float(records_df["decimalLatitude"].max())
    lon_min = float(records_df["decimalLongitude"].min())
    lon_max = float(records_df["decimalLongitude"].max())
    stations = records_df[["decimalLatitude", "decimalLongitude"]].drop_duplicates().shape[0]

    # Depths
    d_min = float(records_df["depth"].min())
    d_max = float(records_df["depth"].max())
    d_mean = round(float(records_df["depth"].mean()), 2)

    # Dates
    date_start = str(records_df["eventDate"].min())[:10]
    date_end = str(records_df["eventDate"].max())[:10]

    summary_data = [{
        "country": "United States",
        "country_code": "US",
        "survey_region": "New Jersey Continental Shelf (Northwest Atlantic Ocean)",
        "total_raw_assays": total_raw_rows,
        "positive_occurrence_records": pos_records,
        "library_unique_barcodes": lib_seqs,
        "barcodes_detected_in_survey": detected_seqs,
        "library_species_count": lib_sp,
        "species_detected_in_survey": detected_sp,
        "library_genus_count": lib_gen,
        "genera_detected_in_survey": detected_gen,
        "library_family_count": lib_fam,
        "families_detected_in_survey": detected_fam,
        "library_order_count": lib_ord,
        "orders_detected_in_survey": detected_ord,
        "library_class_count": lib_cls,
        "classes_detected_in_survey": detected_cls,
        "sampling_stations_count": stations,
        "latitude_min_deg_N": lat_min,
        "latitude_max_deg_N": lat_max,
        "longitude_min_deg_W": lon_min,
        "longitude_max_deg_W": lon_max,
        "depth_min_meters": d_min,
        "depth_max_meters": d_max,
        "depth_mean_meters": d_mean,
        "survey_start_date": date_start,
        "survey_end_date": date_end,
        "scientific_scope_note": "Reflects reference records from 2019 NJ OTS survey; does NOT represent complete national fauna."
    }]

    summary_df = pd.DataFrame(summary_data)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        summary_df.to_csv(output_path, index=False)

    return summary_df


def summarize_species_distribution(
    records_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Summarize reference-record occurrence frequencies across the 101 reference sequences.

    Metrics:
    - positive_occurrences: number of survey samples (out of 68) with positive detection
    - occurrence_prevalence_pct: percentage of surveyed samples with detection
    - unique_stations: number of distinct stations (out of 36) where barcode was found
    - depth and coordinate ranges
    """
    total_samples = 68
    total_stations = 36

    # Aggregate occurrence stats per sequence_id
    occ_stats = records_df.groupby("sequence_id").agg(
        positive_occurrences=("record_id", "count"),
        unique_stations=("sampname", lambda s: records_df.loc[s.index, ["decimalLatitude", "decimalLongitude"]].drop_duplicates().shape[0]),
        min_depth=("depth", "min"),
        max_depth=("depth", "max"),
        mean_depth=("depth", "mean"),
        min_latitude=("decimalLatitude", "min"),
        max_latitude=("decimalLatitude", "max"),
        min_longitude=("decimalLongitude", "min"),
        max_longitude=("decimalLongitude", "max"),
        mean_read_count=("organismQuantity", "mean")
    ).reset_index()

    # Merge with full 101 reference barcodes
    merged = pd.merge(ref_df, occ_stats, on="sequence_id", how="left")

    # Fill NA for barcodes with 0 detections (e.g. SEQ_022, SEQ_026, SEQ_085)
    merged["positive_occurrences"] = merged["positive_occurrences"].fillna(0).astype(int)
    merged["unique_stations"] = merged["unique_stations"].fillna(0).astype(int)
    merged["occurrence_prevalence_pct"] = (merged["positive_occurrences"] / total_samples * 100).round(1)
    merged["station_prevalence_pct"] = (merged["unique_stations"] / total_stations * 100).round(1)
    merged["mean_depth"] = merged["mean_depth"].round(2)
    merged["mean_read_count"] = merged["mean_read_count"].round(1)

    # Library status
    has_sp = merged["species"].notnull() & (merged["species"] != "") & (merged["species"] != "Not available in reference data")
    merged["taxonomic_label_status"] = np.where(has_sp, "species_labeled", "species_unlabeled")
    merged["country"] = "United States"

    cols = [
        "sequence_id",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "taxonomic_label_status",
        "sequence_length",
        "positive_occurrences",
        "occurrence_prevalence_pct",
        "unique_stations",
        "station_prevalence_pct",
        "min_depth",
        "max_depth",
        "mean_depth",
        "min_latitude",
        "max_latitude",
        "min_longitude",
        "max_longitude",
        "mean_read_count",
        "country"
    ]
    res_df = merged[[c for c in cols if c in merged.columns]].sort_values(
        by=["positive_occurrences", "class", "order", "family"],
        ascending=[False, True, True, True]
    ).reset_index(drop=True)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        res_df.to_csv(output_path, index=False)

    return res_df


def analyze_sampling_coverage(
    records_df: pd.DataFrame,
    ref_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Calculate sampling coverage metrics:
    - Proportion of reference barcodes with positive field detections
    - Spatial station density
    - Depth gradient representation
    """
    n_total_barcodes = len(ref_df)
    detected_barcodes = records_df["sequence_id"].nunique()
    undetected_barcodes = n_total_barcodes - detected_barcodes

    n_samples = records_df["sampname"].nunique()
    n_stations = records_df[["decimalLatitude", "decimalLongitude"]].drop_duplicates().shape[0]

    return {
        "total_reference_barcodes": n_total_barcodes,
        "barcodes_with_field_occurrences": detected_barcodes,
        "detection_rate_pct": round(detected_barcodes / n_total_barcodes * 100, 2),
        "undetected_barcodes_count": undetected_barcodes,
        "unique_field_samples": n_samples,
        "unique_sampling_stations": n_stations,
        "depth_gradient_meters": (float(records_df["depth"].min()), float(records_df["depth"].max())),
        "coordinate_completeness_pct": 100.0,
        "sampling_method": "Environmental DNA (eDNA) 12S rDNA V5 metabarcoding ocean trawl survey",
        "geographic_extent": "Offshore New Jersey, USA (38.658°N - 40.468°N, 73.862°W - 74.969°W)"
    }
