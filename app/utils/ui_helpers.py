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
        "icon": "✓",
        "badge_class": "status-known",
        "color": "#14b8a6", # Teal
        "bg_color": "#0f2a28",
        "border_color": "#115e59",
        "description": "Sequence exhibits close sequence identity (distance ≤ 0.030 or exact match) and taxonomic agreement with the cataloged reference library."
    },
    "low_confidence": {
        "label": "Low Confidence Match",
        "short_label": "Low Confidence",
        "icon": "⚠",
        "badge_class": "status-low",
        "color": "#fbbf24", # Amber
        "bg_color": "#2b2412",
        "border_color": "#6b5310",
        "description": "Sequence falls within broad reference boundary (distance ≤ 0.150) but exhibits low probability margin (< 0.150) or taxonomic disagreement."
    },
    "out_of_reference": {
        "label": "Out of Reference Library",
        "short_label": "Out of Reference",
        "icon": "⚠",
        "badge_class": "status-out",
        "color": "#e11d48", # Rose
        "bg_color": "#2d1519",
        "border_color": "#7f1d2d",
        "description": "Sequence is distant from all cataloged reference library barcodes (distance > 0.150). Distance veto enforced."
    },
    "qc_failed": {
        "label": "QC Validation Failed",
        "short_label": "QC Failed",
        "icon": "✕",
        "badge_class": "status-qc",
        "color": "#9ca3af", # Slate
        "bg_color": "#1a2332",
        "border_color": "#334155",
        "description": "Sequence contains invalid DNA characters, unsupported ambiguity codes, or is empty."
    }
}


def render_status_badge(status: str) -> str:
    """Generate HTML snippet for a status badge."""
    cfg = STATUS_CONFIG.get(status, STATUS_CONFIG["qc_failed"])
    # Single line: blank lines inside an HTML block make Markdown render it as code.
    return (
        f'<div style="display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; '
        f'background: {cfg["bg_color"]}; border: 1px solid {cfg["border_color"]}; '
        f'border-radius: 9999px; font-weight: 600; font-size: 0.85rem; color: {cfg["color"]};">'
        f'<span style="font-size: 0.9rem;">{cfg["icon"]}</span><span>{cfg["label"]}</span></div>'
    )


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
                "notes": "Single nucleotide variant (distance ≈ 0.010, within close boundary ≤ 0.030)."
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
        examples["Low-Confidence Query (Intermediate Divergence, ~8% edit distance)"] = {
            "sequence_id": "SYN_LOW_CONF_01",
            "sequence": "".join(low_conf_list),
            "expected_status": "low_confidence",
            "notes": "Diverged variant (0.03 < distance ≤ 0.150) exhibiting classification ambiguity."
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

    # Example 5b: real non-fish 12S sequence (NCBI negative control)
    try:
        import os
        fasta = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             "data", "external", "negative_controls_12S.fasta")
        name, parts, recs = None, [], []
        for line in open(fasta):
            if line.startswith(">"):
                if name: recs.append((name, "".join(parts)))
                name, parts = line[1:].strip(), []
            else:
                parts.append(line.strip().upper())
        if name: recs.append((name, "".join(parts)))
        for nm, sq in recs:
            if "Mus musculus" in nm and "12S" in nm:
                win = sq[:106]
                if set(win) <= set("ACGT") and len(win) == 106:
                    examples["Real Non-Fish Sequence (House mouse 12S, NCBI)"] = {
                        "sequence_id": nm.split()[0],
                        "sequence": win,
                        "expected_status": "out_of_reference",
                        "notes": "Real mouse 12S fragment from NCBI. Not a fish, so the Distance Veto should reject it."
                    }
                break
    except OSError:
        pass

    # Example 6: Invalid QC sequence (contains ambiguous IUPAC code 'N')
    examples["Invalid QC Sequence (Contains IUPAC Degenerate 'N' Bases)"] = {
        "sequence_id": "MALFORMED_01",
        "sequence": "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGNTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAG",
        "expected_status": "qc_failed",
        "notes": "Rejected by biological QC because BioScan strictly disallows ambiguous imputation."
    }

    return examples


# -------------------------------------------------------------
# Brand mark: "Helix fish" (two DNA strands forming a fish)
# -------------------------------------------------------------
def bioscan_logo_svg(width: int = 46) -> str:
    """Inline SVG of the BioScan helix-fish mark. Pure presentation."""
    import math
    strand_a, strand_b, rungs = [], [], []
    for i in range(91):
        t_ = i / 90
        x = 18 + t_ * 132
        amp = 30 * math.pow(math.sin(math.pi * t_), 0.8)
        w = math.sin(t_ * math.pi * 2 * 2.25)
        ya, yb = 62 + amp * w, 62 - amp * w
        strand_a.append(f"{x:.1f},{ya:.1f}")
        strand_b.append(f"{x:.1f},{yb:.1f}")
        if i % 5 == 2:
            rungs.append(
                f'<line x1="{x:.1f}" y1="{ya:.1f}" x2="{x:.1f}" y2="{yb:.1f}" '
                'stroke="#94a3b8" stroke-opacity=".45" stroke-width="2"/>'
            )
    height = round(width * 0.62)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 200 124" role="img" aria-label="BioScan logo">'
        + "".join(rungs)
        + f'<polyline points="{" ".join(strand_a)}" fill="none" stroke="#2dd4bf" stroke-width="4.5" '
          'stroke-linecap="round" stroke-linejoin="round"/>'
        + f'<polyline points="{" ".join(strand_b)}" fill="none" stroke="#60a5fa" stroke-width="4.5" '
          'stroke-linecap="round" stroke-linejoin="round"/>'
        + '<path d="M150 62 L186 30 Q176 62 186 94 Z" fill="#2dd4bf"/>'
        + '<circle cx="40" cy="55" r="3.4" fill="#0b1220"/><circle cx="40" cy="55" r="1.4" fill="#fbbf24"/>'
        + "</svg>"
    )


# -------------------------------------------------------------
# 4. Custom CSS Injection
# -------------------------------------------------------------
def inject_custom_css():
    """Inject polished, high-tech scientific styling for BioScan."""
    st.markdown(
        """
        <style>
        /* 1. HIDE DEFAULT STREAMLIT MULTIPAGE NAVIGATION */
        [data-testid="stSidebarNav"],
        [data-testid="stSidebarNavItems"],
        [data-testid="stSidebarNavSeparator"],
        ul[data-testid="stSidebarNavItems"] {
            display: none !important;
        }

        /* 2. BASE TYPOGRAPHY & LAYOUT */
        html, body, [class*="css"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #e5e7eb;
        }

        /* Main hero styling */
        .hero-title {
            font-size: 1.85rem;
            font-weight: 700;
            color: #e5e7eb;
            letter-spacing: -0.025em;
            margin: 0;
            padding: 0;
            line-height: 1.2;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            color: #9ca3af;
            margin-top: 0.35rem;
            margin-bottom: 0.75rem;
            line-height: 1.5;
        }
        .hero-chip {
            display: inline-flex;
            align-items: center;
            padding: 2px 9px;
            background: #1f2937;
            color: #cbd5e1;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        /* Clean Card Container */
        .bioscan-card {
            background-color: #111827;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.03);
        }

        /* Prediction Result Hero Card */
        .prediction-hero-card {
            background: #111827;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04);
            margin-bottom: 20px;
        }
        .prediction-species {
            font-size: 1.75rem;
            font-weight: 700;
            font-style: italic;
            color: #e5e7eb;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        .prediction-lineage {
            font-size: 0.92rem;
            color: #9ca3af;
            font-weight: 500;
            margin-top: 4px;
        }

        /* KPI Metric Block */
        .kpi-row {
            display: flex;
            gap: 12px;
            margin: 12px 0;
        }
        .kpi-metric {
            background: #1a2332;
            border: 1px solid #1f2937;
            border-radius: 8px;
            padding: 10px 14px;
            flex: 1;
            text-align: center;
        }
        .kpi-label {
            font-size: 0.7rem;
            font-weight: 600;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .kpi-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #e5e7eb;
            margin-top: 2px;
        }

        /* Taxonomy Trail */
        .taxonomy-trail {
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 6px;
            padding: 12px 16px;
            background: #1a2332;
            border: 1px solid #334155;
            border-radius: 8px;
            margin: 12px 0;
        }
        .tax-crumb {
            display: inline-flex;
            align-items: baseline;
            gap: 4px;
            font-size: 0.85rem;
        }
        .tax-crumb-rank {
            font-size: 0.68rem;
            font-weight: 600;
            color: #9ca3af;
            text-transform: uppercase;
        }
        .tax-crumb-val {
            font-weight: 600;
            color: #e5e7eb;
        }
        .tax-arrow {
            color: #6b7280;
            font-size: 0.75rem;
        }

        /* Subtle alerts */
        .clean-alert-warning {
            background-color: #2b2412;
            border: 1px solid #5b4510;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 6px;
            color: #fcd34d;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 12px 0;
        }
        .clean-alert-error {
            background-color: #2d1519;
            border: 1px solid #4c1d2a;
            border-left: 4px solid #f43f5e;
            padding: 12px 16px;
            border-radius: 6px;
            color: #fda4af;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 12px 0;
        }
        .clean-alert-success {
            background-color: #0f2a1c;
            border: 1px solid #14532d;
            border-left: 4px solid #10b981;
            padding: 12px 16px;
            border-radius: 6px;
            color: #86efac;
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 12px 0;
        }

        /* Streamlit Input & Button Polish */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.92rem;
            padding: 0.4rem 1.2rem;
            transition: all 0.15s ease-in-out;
        }
        .stTextArea textarea {
            border-radius: 8px;
            font-family: monospace;
            font-size: 0.88rem;
        }
        .stTextInput input {
            border-radius: 8px;
        }
        /* Hide deploy button & bottom footer */
        .stDeployButton {
            display: none !important;
        }
        footer {
            visibility: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Visual polish layer: typography, spacing, controls. No behaviour or layout changes.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;500;600;700&display=swap');

        html, body, [class*="css"], .stMarkdown, .stButton button, input, textarea {
            font-family: 'Figtree', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }
        .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }

        /* top toolbar: keep the page clean */
        [data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu { display: none !important; }
        header[data-testid="stHeader"] { background: transparent; height: 0; }

        /* headings */
        h1, h2, h3, h4, h5 { letter-spacing: -0.015em; font-weight: 600; }
        h3 { font-size: 1.45rem; margin-top: 0.4rem; }
        h4 { font-size: 1.1rem; margin-top: 1.2rem; color: #dbe5ee; }
        hr { border-color: #1b2535 !important; margin: 1.4rem 0 !important; }

        /* sidebar */
        [data-testid="stSidebar"] { background: #0a0f1a; border-right: 1px solid #172033; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
        [data-testid="stSidebar"] [role="radiogroup"] { gap: 4px; }
        [data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 9px 12px; border-radius: 10px; border: 1px solid transparent;
            width: 100%; transition: background .15s ease, border-color .15s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover { background: #111a2b; }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div:first-child { display: none; }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"] p { font-size: 0.97rem; font-weight: 500; color: #9fb0c3; }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) {
            background: rgba(45, 212, 191, 0.10); border-color: rgba(45, 212, 191, 0.35);
        }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input:checked) p { color: #e8fffb; font-weight: 600; }

        /* buttons */
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px; font-weight: 600; padding: 0.55rem 1.2rem;
            border: 1px solid #243449; background: #101a2b; color: #e5e7eb;
        }
        .stButton > button:hover, .stDownloadButton > button:hover { border-color: #2dd4bf; color: #ffffff; }
        button[data-testid="stBaseButton-primary"] { background: #14b8a6 !important; border-color: #14b8a6 !important; }
        button[data-testid="stBaseButton-primary"] p { color: #03201b !important; font-weight: 700; }
        button[data-testid="stBaseButton-primary"]:hover { background: #2dd4bf !important; border-color: #2dd4bf !important; }
        button[data-testid="stBaseButton-secondary"] { background: #101a2b; border: 1px solid #243449; border-radius: 12px; }
        button[data-testid="stBaseButton-secondary"]:hover { border-color: #2dd4bf; }

        /* inputs */
        .stTextArea textarea, .stTextInput input, [data-baseweb="select"] > div {
            border-radius: 12px !important; border-color: #1f2c40 !important; background: #0d1422 !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus { border-color: #2dd4bf !important; box-shadow: 0 0 0 1px #2dd4bf !important; }
        [data-testid="stFileUploader"] section { border-radius: 14px; border: 1px dashed #2a3b52; background: #0d1422; }

        /* tabs */
        .stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid #172033; }
        .stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 10px 10px 0 0; font-weight: 500; }
        .stTabs [aria-selected="true"] { color: #5eead4 !important; }

        /* metric cards and tables */
        [data-testid="stMetric"] {
            background: #0e1626; border: 1px solid #1b2840; border-radius: 14px; padding: 14px 16px;
        }
        [data-testid="stMetricLabel"] p { color: #8ea2b8; font-size: 0.82rem; font-weight: 500; }
        [data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; }
        [data-testid="stDataFrame"] { border: 1px solid #1b2840; border-radius: 12px; overflow: hidden; }

        /* result cards */
        .prediction-hero-card, .bioscan-card {
            background: #0e1626 !important; border: 1px solid #1b2840 !important; border-radius: 18px !important;
            box-shadow: none !important;
        }
        .kpi-metric { background: #0b1322 !important; border: 1px solid #1b2840 !important; border-radius: 12px !important; }
        .hero-chip { background: rgba(45,212,191,.12) !important; color: #7ff0dd !important; border: 1px solid rgba(45,212,191,.28); }
        .clean-alert-warning, .clean-alert-error, .clean-alert-success { border-radius: 12px !important; }
        [data-testid="stAlert"] { border-radius: 12px; }
        [data-testid="stCaptionContainer"] { color: #8ea2b8; }


        /* sidebar: icons, pinned footer, info cards */
        [data-testid="stSidebar"] label[data-testid="stRadioOption"] { display: flex; align-items: center; gap: 10px; }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]::before { content: ""; width: 18px; height: 18px; flex: 0 0 18px; background-size: contain; background-repeat: no-repeat; order: -1; }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="0"])::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238ea2b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3'/></svg>"); }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="0"]:checked)::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235eead4' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='7'/><path d='M21 21l-4.3-4.3'/></svg>"); }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="1"])::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238ea2b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 3l9 5-9 5-9-5 9-5z'/><path d='M3 13l9 5 9-5'/></svg>"); }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="1"]:checked)::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235eead4' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 3l9 5-9 5-9-5 9-5z'/><path d='M3 13l9 5 9-5'/></svg>"); }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="2"])::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238ea2b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 21s-7-6.2-7-11a7 7 0 0114 0c0 4.8-7 11-7 11z'/><circle cx='12' cy='10' r='2.5'/></svg>"); }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="2"]:checked)::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235eead4' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M12 21s-7-6.2-7-11a7 7 0 0114 0c0 4.8-7 11-7 11z'/><circle cx='12' cy='10' r='2.5'/></svg>"); }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="3"])::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238ea2b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 20V10M10 20V4M16 20v-7M22 20H2'/></svg>"); }
        [data-testid="stSidebar"] label[data-testid="stRadioOption"]:has(input[value="3"]:checked)::before { background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%235eead4' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M4 20V10M10 20V4M16 20v-7M22 20H2'/></svg>"); }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] { min-height: calc(100vh - 7.5rem); }
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.bs-foot) { margin-top: auto; }
        .bs-foot { display: flex; flex-direction: column; gap: 10px; padding-top: 18px; }
        .bs-card { background: #0e1626; border: 1px solid #1b2840; border-radius: 12px; padding: 12px 14px; }
        .bs-h { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; color: #6f87a0; margin-bottom: 8px; }
        .bs-row { display: flex; justify-content: space-between; font-size: 0.86rem; color: #9fb0c3; padding: 2px 0; }
        .bs-row b { color: #e8f1f8; font-weight: 600; font-variant-numeric: tabular-nums; }
        .bs-leg { display: flex; align-items: flex-start; gap: 9px; font-size: 0.84rem; color: #b9c8d8; padding: 3px 0; line-height: 1.3; }
        .bs-leg i { width: 9px; height: 9px; border-radius: 50%; margin-top: 5px; flex: 0 0 9px; }
        .bs-leg small { display: block; color: #6f87a0; font-size: 0.76rem; }
        .bs-small { font-size: 0.74rem; color: #5f758c; padding: 2px 4px; }

        /* brand lockup */
        .bs-brand { display: flex; align-items: center; gap: 10px; }
        .bs-brand .bs-name { font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; color: #eef6fb; line-height: 1; }
        .bs-brand .bs-name span { color: #2dd4bf; }
        .bs-tag { font-size: 0.78rem; color: #8ea2b8; margin-top: 6px; letter-spacing: 0.01em; }
        </style>
        """,
        unsafe_allow_html=True
    )
