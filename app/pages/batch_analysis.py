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
    st.markdown('<div class="main-header">📊 Batch DNA Sequence Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">High-throughput classification of multi-record FASTA files '
        'with independent sequence QC, fault-tolerant batch execution, and CSV export.</div>',
        unsafe_allow_html=True
    )

    predictor = load_predictor()

    st.markdown("##### 📁 Upload Multi-Sequence FASTA File")
    fasta_file = st.file_uploader(
        "Choose a FASTA file containing multiple sequences (.fasta, .fa, .txt):",
        type=["fasta", "fa", "txt", "fna"],
        key="batch_fasta_upload"
    )

    # Demo pre-loaded multi-sequence FASTA option
    st.markdown("Or load a sample 4-sequence demonstration batch:")
    demo_batch_btn = st.button("🧪 Load Sample Demonstration Batch (Known, Variant, Low-Conf, Out-of-Ref)", key="demo_batch_btn")

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
        st.info("Loaded 5-sequence sample batch demonstrating exact match, variant, low confidence, out-of-reference, and QC failure.")

    if not batch_input_content:
        st.caption("Upload a file above or click the sample demonstration button to begin batch prediction.")
        return

    with st.spinner("Processing batch sequences independently..."):
        results = predictor.predict_fasta(batch_input_content, top_k_similar=5)

    if not results:
        st.warning("⚠️ No valid sequence records found in FASTA input.")
        return

    # Convert to flat DataFrame
    df = batch_results_to_dataframe(results)

    # 1. KPI Summary Cards
    st.markdown("---")
    st.markdown("### 📈 Batch Summary Metrics")

    total_seqs = len(df)
    qc_passed = len(df[df["qc_status"] == "passed"])
    known_cnt = len(df[df["prediction_status"] == "known_reference_like"])
    low_cnt = len(df[df["prediction_status"] == "low_confidence"])
    out_cnt = len(df[df["prediction_status"] == "out_of_reference"])
    fail_cnt = len(df[df["prediction_status"] == "qc_failed"])

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Total Input</div>
                <div class="metric-value">{total_seqs}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">QC Passed</div>
                <div class="metric-value" style="color: #10b981;">{qc_passed}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Ref-Match</div>
                <div class="metric-value" style="color: #10b981;">{known_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Low Conf</div>
                <div class="metric-value" style="color: #f59e0b;">{low_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Out of Ref</div>
                <div class="metric-value" style="color: #ef4444;">{out_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k6:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">QC Failed</div>
                <div class="metric-value" style="color: #64748b;">{fail_cnt}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Row
    ch1, ch2 = st.columns(2)
    with ch1:
        fig_status = plot_status_distribution(df)
        st.plotly_chart(fig_status, use_container_width=True)
    with ch2:
        fig_tax = plot_taxonomic_breakdown(df, rank="class")
        st.plotly_chart(fig_tax, use_container_width=True)

    # 3. Interactive Filterable Table
    st.markdown("### 📋 Sequence Prediction Table")
    
    col_filter_status, col_filter_class = st.columns(2)
    with col_filter_status:
        status_options = ["All"] + list(df["prediction_status"].unique())
        selected_status = st.selectbox("Filter by Status:", options=status_options)
    with col_filter_class:
        class_options = ["All"] + [c for c in df["class"].dropna().unique() if c]
        selected_class = st.selectbox("Filter by Predicted Class:", options=class_options)

    filtered_df = df.copy()
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["prediction_status"] == selected_status]
    if selected_class != "All":
        filtered_df = filtered_df[filtered_df["class"] == selected_class]

    # Present formatted columns
    display_cols = [
        "sequence_id", "sequence_length", "qc_status", "prediction_status",
        "class", "order", "family", "genus", "species",
        "top1_probability", "nearest_reference", "nearest_distance", "nearest_similarity"
    ]
    avail_cols = [c for c in display_cols if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[avail_cols],
        use_container_width=True,
        hide_index=True
    )

    # 4. CSV Download Button
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Batch Results (CSV)",
        data=csv_bytes,
        file_name="bioscan_batch_predictions.csv",
        mime="text/csv",
        type="primary"
    )
