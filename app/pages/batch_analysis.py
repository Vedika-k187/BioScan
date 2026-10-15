"""
app/pages/batch_analysis.py
Phase 9: High-Throughput Batch FASTA Analysis Page

Allows users to upload multi-sequence FASTA files, runs independent biological
prediction per record with fault isolation, renders summary KPI analytics,
interactive filtering, and exports flat CSV tables.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from app.utils.ui_helpers import load_predictor
from app.components import (
    plot_status_distribution,
    plot_taxonomic_breakdown
)
from src.prediction.batch_predict import batch_results_to_dataframe


def render_batch_page():
    """Render the Batch DNA Analysis Page."""
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #e5e7eb; letter-spacing: -0.02em;">Batch DNA Analysis</div>
            <div style="font-size: 0.95rem; color: #9ca3af; margin-top: 4px;">Upload multiple sequences and classify them in high-throughput mode.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    predictor = load_predictor()

    col_up, col_demo = st.columns([2, 1])
    with col_up:
        fasta_file = st.file_uploader(
            "Upload multi-sequence FASTA (.fasta, .fa, .txt):",
            type=["fasta", "fa", "txt", "fna"],
            key="batch_fasta_upload"
        )
    with col_demo:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        demo_batch_btn = st.button("🧪 Load Sample 5-Sequence Batch", key="demo_batch_btn", use_container_width=True)

    batch_input_content = None
    if fasta_file is not None:
        batch_input_content = fasta_file.getvalue().decode("utf-8", errors="ignore")
    elif demo_batch_btn:
        batch_input_content = (
            ">SAMPLE_01_OPHIDION_EXACT\n"
            "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAC\n"
            ">SAMPLE_02_SEAROBIN_VARIANT\n"
            "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAA\n"
            ">SAMPLE_03_LOW_CONFIDENCE\n"
            "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGTTTTTTTGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAC\n"
            ">SAMPLE_04_OUT_OF_REFERENCE\n"
            "TAGCGTAATACCGTACCGTACTACGTATCGTAGCTAGCTAGCTAGCGTACGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG\n"
            ">SAMPLE_05_INVALID_QC_AMBIGUOUS\n"
            "ACTATGCTCAGTCATAAACTTTGGTGATAAATTACACATATTGCCCGCCAGGGNTACTACGAGCGCTAGCTTAAAACCCAAAGGACTTGGCGGTGCCCCAGACCCAC\n"
        )

    if not batch_input_content:
        st.markdown(
            """
            <div style="padding: 24px; background: #1a2332; border: 1px dashed #475569; border-radius: 8px; text-align: center; color: #9ca3af; font-size: 0.9rem; margin-top: 12px;">
                Upload a FASTA file or click <strong>Load Sample 5-Sequence Batch</strong> to run high-throughput analysis.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    with st.spinner("Processing batch sequences..."):
        results = predictor.predict_fasta(batch_input_content, top_k_similar=5)

    if not results:
        st.warning("⚠️ No valid sequence records found in FASTA input.")
        return

    # Convert to flat DataFrame
    df = batch_results_to_dataframe(results)

    # 1. KPI Summary Cards
    total_seqs = len(df)
    qc_passed = len(df[df["qc_status"] == "passed"])
    known_cnt = len(df[df["prediction_status"] == "known_reference_like"])
    low_cnt = len(df[df["prediction_status"] == "low_confidence"])
    out_cnt = len(df[df["prediction_status"] == "out_of_reference"])
    fail_cnt = len(df[df["prediction_status"] == "qc_failed"])

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Total</div>
                <div class="kpi-value">{total_seqs}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">QC Passed</div>
                <div class="kpi-value" style="color: #14b8a6;">{qc_passed}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Ref-Match</div>
                <div class="kpi-value" style="color: #14b8a6;">{known_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Low Conf</div>
                <div class="kpi-value" style="color: #fbbf24;">{low_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Out of Ref</div>
                <div class="kpi-value" style="color: #e11d48;">{out_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k6:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">QC Failed</div>
                <div class="kpi-value" style="color: #9ca3af;">{fail_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 2. Charts Row
    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_status = plot_status_distribution(df)
        st.plotly_chart(fig_status, use_container_width=True)
    with ch2:
        fig_tax = plot_taxonomic_breakdown(df, rank="class")
        st.plotly_chart(fig_tax, use_container_width=True)

    # 3. Interactive Filterable Table
    st.markdown("<div style='font-size: 1rem; font-weight: 700; color: #e5e7eb; margin: 16px 0 8px 0;'>Batch Results</div>", unsafe_allow_html=True)
    
    col_filter_status, col_filter_class, col_toggle = st.columns([1, 1, 1])
    with col_filter_status:
        status_options = ["All"] + list(df["prediction_status"].unique())
        selected_status = st.selectbox("Status:", options=status_options)
    with col_filter_class:
        class_options = ["All"] + [c for c in df["class"].dropna().unique() if c]
        selected_class = st.selectbox("Class:", options=class_options)
    with col_toggle:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        show_full_cols = st.checkbox("Show all columns", value=False)

    filtered_df = df.copy()
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["prediction_status"] == selected_status]
    if selected_class != "All":
        filtered_df = filtered_df[filtered_df["class"] == selected_class]

    # Present formatted columns: concise initially, full if toggled
    if show_full_cols:
        display_cols = [
            "sequence_id", "sequence_length", "qc_status", "prediction_status",
            "class", "order", "family", "genus", "species",
            "top1_probability", "nearest_reference", "nearest_distance", "nearest_similarity"
        ]
    else:
        display_cols = [
            "sequence_id", "prediction_status", "species",
            "top1_probability", "nearest_similarity", "nearest_distance"
        ]
    avail_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[avail_cols],
        use_container_width=True,
        hide_index=True
    )

    # 4. Secondary CSV Download Button
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results (CSV)",
        data=csv_bytes,
        file_name="bioscan_batch_predictions.csv",
        mime="text/csv"
    )
