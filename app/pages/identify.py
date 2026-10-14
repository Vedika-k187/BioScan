"""
app/pages/identify.py
Phase 9: DNA Identification Page (Core Interactive Demonstration)

Allows users to input a single DNA sequence (via paste or FASTA upload),
runs biological QC, executes the end-to-end Phase 8 prediction engine,
and presents complete, transparent evidence.
"""

import streamlit as st
import pandas as pd
from typing import Optional

from app.utils.ui_helpers import (
    load_predictor,
    load_reference_barcodes,
    load_biodiversity_records,
    get_example_sequences
)
from app.components import (
    render_prediction_card,
    render_taxonomy_hierarchy,
    render_confidence_and_evidence,
    render_similarity_table,
    render_biodiversity_context
)


def render_identify_page():
    """Render the main DNA Sequence Identification Page."""
    # 1. Page Header & Scope Banner
    st.markdown('<div class="main-header">🧬 BioScan DNA Identification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">From DNA sequence to taxonomic identification, confidence, '
        'similarity evidence, and biodiversity context.</div>',
        unsafe_allow_html=True
    )

    # Capability tags
    st.markdown(
        """
        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;">
            <span class="tax-step">📌 <b>Locus:</b> 12S rDNA V5</span>
            <span class="tax-step">🐟 <b>Domain:</b> Teleost & Elasmobranch Fish</span>
            <span class="tax-step">🌲 <b>Model:</b> Hierarchical ML (Class &rarr; Species)</span>
            <span class="tax-step">🔎 <b>Search:</b> Normalized Levenshtein Top-5</span>
            <span class="tax-step">🛡️ <b>Safety:</b> Distance Veto Unknown Detection</span>
            <span class="tax-step">🌍 <b>Context:</b> 2019 NJ Ocean Trawl Survey</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Strict Scope Limitation Notice
    st.caption(
        "**Reference Library Scope**: 101 unique 12S barcode sequences representing 90 named species (89 singletons). "
        "Predictions reflect reference-library matching and are bounded by the cataloged barcode domain."
    )
    st.markdown("---")

    # Load resources
    predictor = load_predictor()
    ref_df = load_reference_barcodes()
    bio_records_df = load_biodiversity_records()
    examples = get_example_sequences(ref_df)

    # 2. Dynamic Example Selector
    st.markdown("##### 💡 Try an Example Sequence from the Validated Dataset")
    selected_example_name = st.selectbox(
        "Select a pre-loaded reference sequence or synthetic test case:",
        options=["(Custom Input)"] + list(examples.keys()),
        index=0
    )

    default_seq_id = "SAMPLE_QUERY_01"
    default_seq = ""

    if selected_example_name != "(Custom Input)":
        ex_info = examples[selected_example_name]
        default_seq_id = ex_info["sequence_id"]
        default_seq = ex_info["sequence"]
        st.info(f"**Loaded Example:** {ex_info['notes']}")

    # 3. Input Tabs (Paste vs Upload FASTA)
    tab_paste, tab_fasta = st.tabs(["✍️ Paste DNA Sequence", "📁 Upload Single-Sequence FASTA"])
    
    seq_to_predict = ""
    seq_id_to_predict = default_seq_id
    trigger_prediction = False

    with tab_paste:
        col_id, col_btn = st.columns([3, 1])
        with col_id:
            input_seq_id = st.text_input("Sequence Identifier (Optional):", value=default_seq_id, key="input_id_paste")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn_paste = st.button("🚀 Identify Sequence", type="primary", use_container_width=True, key="btn_paste")

        input_sequence = st.text_area(
            "DNA Sequence (A, C, G, T):",
            value=default_seq,
            height=120,
            placeholder="e.g. ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAC",
            key="input_seq_paste"
        )
        if predict_btn_paste:
            seq_to_predict = input_sequence
            seq_id_to_predict = input_seq_id
            trigger_prediction = True

    with tab_fasta:
        st.caption("Upload a `.fasta`, `.fa`, or `.txt` file containing a single DNA sequence.")
        fasta_file = st.file_uploader("Choose a FASTA file", type=["fasta", "fa", "txt", "fna"], key="fasta_upload_single")
        predict_btn_fasta = st.button("🚀 Identify Uploaded FASTA", type="primary", key="btn_fasta")

        if predict_btn_fasta and fasta_file is not None:
            content = fasta_file.getvalue().decode("utf-8", errors="ignore")
            # Quick parse header and sequence
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            header_id = "UPLOADED_FASTA"
            fasta_seq_parts = []
            for ln in lines:
                if ln.startswith(">"):
                    header_id = ln[1:].split()[0]
                else:
                    fasta_seq_parts.append(ln)
            seq_to_predict = "".join(fasta_seq_parts)
            seq_id_to_predict = header_id
            trigger_prediction = True

    # 4. Prediction Execution & Rendering
    if trigger_prediction:
        if not seq_to_predict.strip():
            st.warning("⚠️ Please provide or paste a non-empty DNA sequence to classify.")
            return

        with st.spinner("Executing biological QC, hierarchical inference, similarity search, and evidence aggregation..."):
            result = predictor.predict(seq_to_predict, sequence_id=seq_id_to_predict, top_k_similar=5)

        # A. Biological QC Panel
        qc = result.get("qc", {})
        inp = result.get("input", {})
        st.markdown("---")
        st.markdown("### 🔬 Biological Input Quality Control (QC)")
        
        qc_col1, qc_col2, qc_col3, qc_col4 = st.columns(4)
        with qc_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Sequence Length</div>
                    <div class="metric-value">{inp.get('length', 0)} bp</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with qc_col2:
            is_valid = qc.get("is_valid", False)
            valid_txt = "✅ VALID" if is_valid else "❌ INVALID"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Bases (A,C,G,T)</div>
                    <div class="metric-value" style="font-size: 1.15rem; color: {'#10b981' if is_valid else '#ef4444'};">{valid_txt}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with qc_col3:
            # Calculate GC from input safely
            clean_s = seq_to_predict.upper()
            gc_pct = ((clean_s.count('G') + clean_s.count('C')) / max(len(clean_s), 1)) * 100 if clean_s else 0.0
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">GC Content</div>
                    <div class="metric-value">{gc_pct:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with qc_col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">QC Status</div>
                    <div class="metric-value" style="font-size: 1.15rem;">{qc.get('status', 'N/A').upper()}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Display QC warnings if any
        warnings = qc.get("warnings", [])
        if warnings:
            st.markdown(
                f"""
                <div class="warning-box">
                    <strong>⚠️ Quality Control Advisory:</strong><br>
                    {warnings[0]}
                </div>
                """,
                unsafe_allow_html=True
            )

        # B. If QC Failed -> Render failure card and stop
        if not qc.get("is_valid"):
            render_prediction_card(result)
            return

        # C. Render Prediction Results
        st.markdown("<br>", unsafe_allow_html=True)
        render_prediction_card(result)
        
        st.markdown("<br>", unsafe_allow_html=True)
        render_taxonomy_hierarchy(result)

        st.markdown("<br>", unsafe_allow_html=True)
        render_confidence_and_evidence(result)

        st.markdown("<br>", unsafe_allow_html=True)
        render_similarity_table(result)

        st.markdown("<br>", unsafe_allow_html=True)
        render_biodiversity_context(result, bio_records_df)

        # Disclaimer
        st.markdown("---")
        st.caption(f"**Scientific Transparency Notice**: {result.get('disclaimer', '')}")
