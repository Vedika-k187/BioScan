"""
app/utils/ui_helpers.py
Phase 9: BioScan Interactive Biodiversity Intelligence Dashboard UI Helpers

Provides:
1. Streamlit resource caching for prediction engine and static processed tables
2. Dynamic reference sequence example loader
3. Centralized status configuration and badge rendering
4. Scientific CSS styling injection
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple

# Ensure workspace root is in sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.prediction.predictor import get_predictor, BioScanPredictor

DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REF_BARCODES_PATH = os.path.join(DATA_PROCESSED_DIR, "reference_barcodes.csv")
DIVERSITY_METRICS_PATH = os.path.join(DATA_PROCESSED_DIR, "diversity_metrics.csv")
SPECIES_DIST_PATH = os.path.join(DATA_PROCESSED_DIR, "species_distribution_summary.csv")
BIO_RECORDS_PATH = os.path.join(DATA_PROCESSED_DIR, "biodiversity_reference_records.csv")

# -------------------------------------------------------------
# 1. Cached Loaders
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading BioScan Prediction Engine (models & reference library)...")
def load_predictor() -> BioScanPredictor:
    """Load and cache the singleton BioScanPredictor engine."""
    return get_predictor()


@st.cache_data(show_spinner=False)
def load_reference_barcodes() -> pd.DataFrame:
    """Load reference barcode catalog."""
    if os.path.exists(REF_BARCODES_PATH):
        return pd.read_csv(REF_BARCODES_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_diversity_metrics() -> pd.DataFrame:
    """Load station-level diversity metrics."""
    if os.path.exists(DIVERSITY_METRICS_PATH):
        return pd.read_csv(DIVERSITY_METRICS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_species_distribution() -> pd.DataFrame:
    """Load species survey prevalence and geographic summary."""
    if os.path.exists(SPECIES_DIST_PATH):
        return pd.read_csv(SPECIES_DIST_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_biodiversity_records() -> pd.DataFrame:
    """Load individual survey detection records."""
    if os.path.exists(BIO_RECORDS_PATH):
        return pd.read_csv(BIO_RECORDS_PATH)
    return pd.DataFrame()


# -------------------------------------------------------------
# 2. Status Configurations & Visual Styles
# -------------------------------------------------------------
STATUS_CONFIG = {
    "known_reference_like": {
        "label": "Reference-Library Match",
        "short_label": "Known Reference-Like",
        "icon": "🟢",
        "badge_class": "status-known",
        "color": "#10b981", # Emerald
        "bg_color": "rgba(16, 185, 129, 0.12)",
        "border_color": "#10b981",
        "description": "Sequence exhibits close sequence identity (distance ≤ 0.050 or exact match) and taxonomic agreement with the cataloged reference library."
    },
    "low_confidence": {
        "label": "Low Confidence Match",
        "short_label": "Low Confidence",
        "icon": "🟡",
        "badge_class": "status-low",
        "color": "#f59e0b", # Amber
        "bg_color": "rgba(245, 158, 11, 0.12)",
        "border_color": "#f59e0b",
        "description": "Sequence falls within broad reference boundary (distance ≤ 0.150) but exhibits low probability margin (< 0.150) or taxonomic disagreement."
    },
    "out_of_reference": {
        "label": "Out of Reference Library",
        "short_label": "Out of Reference",
        "icon": "🔴",
        "badge_class": "status-out",
        "color": "#ef4444", # Red
        "bg_color": "rgba(239, 68, 68, 0.12)",
        "border_color": "#ef4444",
        "description": "Sequence is distant from all cataloged reference library barcodes (distance > 0.150). Distance veto enforced."
    },
    "qc_failed": {
        "label": "QC Validation Failed",
        "short_label": "QC Failed",
        "icon": "⚪",
        "badge_class": "status-qc",
        "color": "#94a3b8", # Slate
        "bg_color": "rgba(148, 163, 184, 0.12)",
        "border_color": "#94a3b8",
        "description": "Sequence contains invalid DNA characters, unsupported ambiguity codes, or is empty."
    }
}


def render_status_badge(status: str) -> str:
    """Generate HTML snippet for a status badge."""
    cfg = STATUS_CONFIG.get(status, STATUS_CONFIG["qc_failed"])
    return f"""
    <div style="display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px; 
                background: {cfg['bg_color']}; border: 1px solid {cfg['border_color']}; 
                border-radius: 9999px; font-weight: 600; font-size: 0.95rem; color: {cfg['color']};">
        <span>{cfg['icon']}</span>
        <span>{cfg['label']}</span>
    </div>
    """


# -------------------------------------------------------------
# 3. Dynamic Example Sequences Loader
# -------------------------------------------------------------
def get_example_sequences(ref_df: Optional[pd.DataFrame] = None) -> Dict[str, Dict[str, Any]]:
    """
    Dynamically extract reference sequences from processed dataset to provide
    real, synchronized examples for testing and demonstration.
    """
    if ref_df is None or ref_df.empty:
        ref_df = load_reference_barcodes()

    examples = {}

    if not ref_df.empty:
        # Example 1: Ophidion marginatum (Striped cusk-eel)
        ophidion_rows = ref_df[ref_df["species"].fillna("").str.contains("Ophidion marginatum", case=False)]
        if not ophidion_rows.empty:
            row1 = ophidion_rows.iloc[0]
            seq1 = str(row1["sequence_normalized"])
            examples["Exact Reference — Ophidion marginatum (Striped cusk-eel)"] = {
                "sequence_id": str(row1["sequence_id"]),
                "sequence": seq1,
                "expected_status": "known_reference_like",
                "notes": "Exact 12S V5 reference barcode from library (distance 0.000)."
            }

            # Example 2: 1-bp synthetic mutation of Ophidion
            mut_seq = list(seq1)
            mid_idx = len(mut_seq) // 2
            mut_seq[mid_idx] = "C" if mut_seq[mid_idx] != "C" else "A"
            examples["Known Reference-Like (1-bp Natural/Sequencing Variant)"] = {
                "sequence_id": f"{row1['sequence_id']}_var1",
                "sequence": "".join(mut_seq),
                "expected_status": "known_reference_like",
                "notes": "Single nucleotide variant (distance ≈ 0.010, within close boundary ≤ 0.050)."
            }

        # Example 3: Prionotus carolinus (Northern searobin)
        prionotus_rows = ref_df[ref_df["species"].fillna("").str.contains("Prionotus carolinus", case=False)]
        if not prionotus_rows.empty:
            row2 = prionotus_rows.iloc[0]
            examples["Exact Reference — Prionotus carolinus (Northern searobin)"] = {
                "sequence_id": str(row2["sequence_id"]),
                "sequence": str(row2["sequence_normalized"]),
                "expected_status": "known_reference_like",
                "notes": "Exact reference barcode demonstrating elasmobranch/teleost classification."
            }

        # Example 4: Low-confidence case (synthetic ~7-bp divergence)
        if ophidion_rows.empty and not ref_df.empty:
            base_seq = str(ref_df.iloc[0]["sequence_normalized"])
        else:
            base_seq = str(ophidion_rows.iloc[0]["sequence_normalized"]) if not ophidion_rows.empty else "ACTG"*25
        
        low_conf_list = list(base_seq)
        for idx in range(10, 18):
            if idx < len(low_conf_list):
                curr = low_conf_list[idx]
                low_conf_list[idx] = "T" if curr != "T" else "G"
        examples["Low-Confidence Query (Intermediate Divergence, ~7% edit distance)"] = {
            "sequence_id": "SYN_LOW_CONF_01",
            "sequence": "".join(low_conf_list),
            "expected_status": "low_confidence",
            "notes": "Diverged variant (0.05 < distance ≤ 0.150) exhibiting classification ambiguity."
        }

    # Example 5: Out-of-reference sequence (synthetic highly diverged sequence)
    # Guaranteed distance > 0.150 from any reference
    out_seq = "TAGCGTAATACCGTACCGTACTACGTATCGTAGCTAGCTAGCTAGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    examples["Out-of-Reference Sequence (Divergent Non-Library Barcode)"] = {
        "sequence_id": "SYN_OUT_REF_01",
        "sequence": out_seq,
        "expected_status": "out_of_reference",
        "notes": "Synthetic divergent sequence (>15% distance) demonstrating Distance Veto enforcement."
    }

    # Example 6: Invalid QC sequence (contains ambiguous IUPAC code 'N')
    examples["Invalid QC Sequence (Contains IUPAC Degenerate 'N' Bases)"] = {
        "sequence_id": "MALFORMED_01",
        "sequence": "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGNTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAG",
        "expected_status": "qc_failed",
        "notes": "Rejected by biological QC because BioScan strictly disallows ambiguous imputation."
    }

    return examples


# -------------------------------------------------------------
# 4. Custom CSS Injection
# -------------------------------------------------------------
def inject_custom_css():
    """Inject polished, high-tech scientific styling for BioScan."""
    st.markdown(
        """
        <style>
        /* Base typography & containers */
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #0284c7 0%, #0d9488 50%, #10b981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }
        .bio-card {
            background-color: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 18px 22px;
            margin-bottom: 16px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        }
        .tax-step {
            display: inline-block;
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            padding: 6px 12px;
            margin: 4px;
            font-size: 0.88rem;
        }
        .tax-rank {
            font-weight: 700;
            color: #475569;
            text-transform: uppercase;
            font-size: 0.72rem;
            letter-spacing: 0.05em;
        }
        .tax-value {
            font-size: 0.95rem;
            color: #0f172a;
            font-weight: 600;
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 12px 16px;
            text-align: center;
        }
        .metric-label {
            font-size: 0.78rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0f172a;
            margin-top: 4px;
        }
        .warning-box {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 4px;
            color: #92400e;
            font-size: 0.92rem;
            margin: 10px 0;
        }
        .alert-box {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 12px 16px;
            border-radius: 4px;
            color: #991b1b;
            font-size: 0.92rem;
            margin: 10px 0;
        }
        .success-box {
            background-color: #ecfdf5;
            border-left: 4px solid #10b981;
            padding: 12px 16px;
            border-radius: 4px;
            color: #065f46;
            font-size: 0.92rem;
            margin: 10px 0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
