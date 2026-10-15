"""
app/pages/identify.py
Phase 9: DNA Identification Page (Core Interactive Demonstration)

Allows users to input a single DNA sequence (via paste or FASTA upload),
runs biological QC, executes the end-to-end Phase 8 prediction engine,
and presents clean, transparent evidence.
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
    # 1. Page Hero Banner
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="display: flex; align-items: baseline; flex-wrap: wrap; gap: 10px;">
                <span style="font-size: 1.75rem;">🧬</span>
                <span style="font-size: 1.6rem; font-weight: 700; color: #e5e7eb; letter-spacing: -0.02em;">BioScan</span>
                <span style="font-size: 1.1rem; color: #9ca3af; font-weight: 500;">DNA &rarr; Species Identification</span>
                <span class="hero-chip">12S rDNA V5</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 4px;">
                Identify an unknown DNA barcode using machine learning, sequence similarity, and taxonomic evidence.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Scope: 12S rDNA V5 (87–112 bp) · 101 reference barcodes, 90 species · marine fish, 2019 NJ Ocean Trawl Survey · "
        "close match ≤ 0.03, Distance Veto > 0.15"
    )

    # Load resources
    predictor = load_predictor()
    ref_df = load_reference_barcodes()
    bio_records_df = load_biodiversity_records()
    examples = get_example_sequences(ref_df)

    # 2. Dynamic Example Selector
    st.markdown("<div style='font-size: 0.88rem; font-weight: 600; color: #d1d5db; margin-top: 0.5rem; margin-bottom: 4px;'>Try an example:</div>", unsafe_allow_html=True)

    def on_example_change():
        sel = st.session_state.get("example_selector")
        if sel and sel != "(Custom Input)" and sel in examples:
            ex = examples[sel]
            st.session_state["input_seq_paste"] = ex["sequence"]
            st.session_state["input_id_paste"] = ex["sequence_id"]
        elif sel == "(Custom Input)":
            st.session_state["input_seq_paste"] = ""
            st.session_state["input_id_paste"] = "SAMPLE_QUERY_01"

    if "input_seq_paste" not in st.session_state:
        st.session_state["input_seq_paste"] = ""
    if "input_id_paste" not in st.session_state:
        st.session_state["input_id_paste"] = "SAMPLE_QUERY_01"

    selected_example_name = st.selectbox(
        "Try an example:",
        options=["(Custom Input)"] + list(examples.keys()),
        index=0,
        key="example_selector",
        label_visibility="collapsed",
        on_change=on_example_change
    )

    if "last_selected_example" not in st.session_state:
        st.session_state["last_selected_example"] = selected_example_name

    # Fallback synchronization if selection changes
    if selected_example_name != st.session_state["last_selected_example"]:
        st.session_state["last_selected_example"] = selected_example_name
        if selected_example_name != "(Custom Input)" and selected_example_name in examples:
            ex_info = examples[selected_example_name]
            st.session_state["input_seq_paste"] = ex_info["sequence"]
            st.session_state["input_id_paste"] = ex_info["sequence_id"]
        elif selected_example_name == "(Custom Input)":
            st.session_state["input_seq_paste"] = ""
            st.session_state["input_id_paste"] = "SAMPLE_QUERY_01"

    if selected_example_name != "(Custom Input)" and selected_example_name in examples:
        seq_len = len(st.session_state.get("input_seq_paste", ""))
        st.markdown(
            f"""
            <div style="font-size: 0.82rem; color: #14b8a6; font-weight: 600; margin: 4px 0 10px 2px;">
                ✓ Example loaded · {seq_len} bp
            </div>
            """,
            unsafe_allow_html=True
        )

    # 3. Input Tabs (Paste vs Upload FASTA)
    tab_paste, tab_fasta = st.tabs(["Paste Sequence", "Upload FASTA"])
    
    seq_to_predict = ""
    seq_id_to_predict = st.session_state.get("input_id_paste", "SAMPLE_QUERY_01")
    trigger_prediction = False

    with tab_paste:
        input_sequence = st.text_area(
            "DNA sequence",
            height=130,
            placeholder="Paste canonical DNA sequence (e.g. ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAC)...",
            key="input_seq_paste",
            label_visibility="collapsed"
        )
        col_id, col_btn = st.columns([2, 1])
        with col_id:
            input_seq_id = st.text_input("Sequence ID (optional)", key="input_id_paste", placeholder="Sequence ID (optional)", label_visibility="collapsed")
        with col_btn:
            predict_btn_paste = st.button("Identify Sequence", type="primary", use_container_width=True, key="btn_paste")

        if predict_btn_paste:
            seq_to_predict = input_sequence
            seq_id_to_predict = input_seq_id or "QUERY_01"
            trigger_prediction = True

    with tab_fasta:
        st.markdown("<div style='font-size: 0.85rem; color: #9ca3af; margin-bottom: 6px;'>Upload a single-sequence FASTA (.fasta, .fa, .txt):</div>", unsafe_allow_html=True)
        fasta_file = st.file_uploader("Upload single-sequence FASTA file", type=["fasta", "fa", "txt", "fna"], key="fasta_upload_single", label_visibility="collapsed")
        predict_btn_fasta = st.button("Identify Uploaded FASTA", type="primary", key="btn_fasta")

        if predict_btn_fasta and fasta_file is not None:
            content = fasta_file.getvalue().decode("utf-8", errors="ignore")
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

        with st.spinner("Analyzing sequence..."):
            result = predictor.predict(seq_to_predict, sequence_id=seq_id_to_predict, top_k_similar=5)

        qc = result.get("qc", {})
        inp = result.get("input", {})

        # If QC failed -> render failure card directly and stop
        if not qc.get("is_valid"):
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            render_prediction_card(result)
            return

        # Subtle QC chip if valid
        clean_s = seq_to_predict.upper()
        gc_pct = ((clean_s.count('G') + clean_s.count('C')) / max(len(clean_s), 1)) * 100 if clean_s else 0.0
        st.markdown(
            f"""
            <div style="font-size: 0.78rem; color: #9ca3af; margin: 12px 0 8px 2px;">
                ✓ Biological QC Passed · {inp.get('length', 0)} bp · {gc_pct:.1f}% GC
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(f"### Analysis Report · {seq_id_to_predict}")
        st.caption("1. Verdict  ·  2. Taxonomic lineage  ·  3. Reference similarity  ·  4. Model diagnostics  ·  5. Biodiversity context")

        # 1. Verdict
        render_prediction_card(result)
        
        # Distance gauge: where this query sits relative to the veto thresholds
        _render_distance_gauge(result.get("similarity", {}).get("nearest_distance"))

        # 2. Taxonomic Lineage Trail
        render_taxonomy_hierarchy(result)

        # Expandable Deep Technical Sections
        pred_status = result.get("prediction", {}).get("status")
        
        st.divider()
        st.markdown("#### 3 · Reference Similarity Evidence (Top 5 Barcodes)")
        with st.container():
            render_similarity_table(result)

        st.divider()
        st.markdown("#### 4 · Model Confidence & Diagnostics")
        with st.container():
            render_confidence_and_evidence(result)

        st.divider()
        st.markdown("#### 5 · Biodiversity Occurrence Context")
        with st.container():
            render_biodiversity_context(result, bio_records_df)


def _render_distance_gauge(distance: Optional[float]):
    """Horizontal gauge: query distance vs. the close (0.03) and veto (0.15) thresholds."""
    if distance is None:
        return
    import plotly.graph_objects as go

    max_x = 0.6
    fig = go.Figure()
    zones = [(0.0, 0.03, "#14532d", "Known"), (0.03, 0.15, "#6b5310", "Low confidence"),
             (0.15, max_x, "#5c1d1d", "Out of reference")]
    for x0, x1, color, label in zones:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=color, line_width=0, layer="below")
        fig.add_annotation(x=(x0 + x1) / 2 + (0.012 if x0 == 0.0 else 0), y=0.5, text=label, showarrow=False,
                           font=dict(size=11, color="#d1d5db"))
    x = min(distance, max_x)
    fig.add_shape(type="line", x0=x, x1=x, y0=-0.15, y1=1.15, line=dict(color="#e5e7eb", width=3))
    fig.add_annotation(x=x, y=1.35, text=f"Query d = {distance:.3f}", showarrow=False,
                       xanchor="left" if x < 0.1 else "center",
                       font=dict(size=12, color="#e5e7eb"))
    fig.update_xaxes(range=[0, max_x], title="Normalised edit distance to nearest reference",
                     showgrid=False, tickvals=[0, 0.03, 0.15, 0.3, 0.6])
    fig.update_yaxes(visible=False, range=[-0.2, 1.6])
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=10, b=40),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
