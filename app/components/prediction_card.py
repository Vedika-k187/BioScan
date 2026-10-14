"""
app/components/prediction_card.py
Phase 9: Main BioScan Prediction & Identification Card

Renders the primary biological identification summary, status indicator,
and scientific constraint warnings.
"""

import streamlit as st
from typing import Dict, Any
from app.utils.ui_helpers import render_status_badge, STATUS_CONFIG


def render_prediction_card(result: Dict[str, Any]):
    """Render the primary identification summary card for a prediction result."""
    qc = result.get("qc", {})
    prediction = result.get("prediction", {})
    confidence = result.get("confidence", {})
    similarity = result.get("similarity", {})
    evidence = result.get("evidence", {})
    
    status = prediction.get("status") if prediction else result.get("status", "qc_failed")
    
    # 1. QC Failed State
    if status == "qc_failed":
        st.markdown(
            f"""
            <div class="clean-alert-error" style="margin-bottom: 1.5rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 700; font-size: 1.05rem;">❌ Quality Control (QC) Validation Failed</span>
                    {render_status_badge("qc_failed")}
                </div>
                <div>The input sequence could not be classified because it violates canonical DNA requirements (A, C, G, T only) or is empty. BioScan strictly rejects ambiguous IUPAC codes without silent imputation to preserve scientific integrity.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        errors = qc.get("errors", [])
        if errors:
            st.markdown("**QC Failure Details:**")
            for err in errors:
                st.markdown(f"- ✕ `{err}`")
        return

    # Extract primary metrics
    d_min = similarity.get("nearest_distance")
    sim = similarity.get("nearest_similarity")
    p1 = confidence.get("top1_probability")
    nearest_ref = similarity.get("nearest_reference_id", "N/A")

    # 2. Out of Reference Library
    if status == "out_of_reference":
        st.markdown(
            f"""
            <div class="prediction-hero-card" style="border-left: 4px solid #e11d48;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;">Prediction</div>
                        <div class="prediction-species" style="color: #e11d48; font-style: normal;">Out of Reference Library</div>
                        <div class="prediction-lineage">Sequence divergence exceeds reference library boundary · Distance Veto enforced</div>
                    </div>
                    <div>
                        {render_status_badge("out_of_reference")}
                    </div>
                </div>
                <div class="kpi-row" style="margin-top: 18px;">
                    <div class="kpi-metric">
                        <div class="kpi-label">Nearest Distance</div>
                        <div class="kpi-value" style="color: #e11d48;">{f"{d_min:.3f}" if d_min is not None else "N/A"}</div>
                    </div>
                    <div class="kpi-metric">
                        <div class="kpi-label">Nearest Similarity</div>
                        <div class="kpi-value">{f"{sim:.1%}" if sim is not None else "N/A"}</div>
                    </div>
                    <div class="kpi-metric">
                        <div class="kpi-label">Nearest Barcode</div>
                        <div class="kpi-value" style="font-size: 1.1rem; padding-top: 4px;"><code>{nearest_ref}</code></div>
                    </div>
                    <div class="kpi-metric">
                        <div class="kpi-label">Distance Veto</div>
                        <div class="kpi-value" style="font-size: 1.05rem; padding-top: 4px; color: #e11d48;">Enforced (&gt;0.150)</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="clean-alert-warning">
                <strong>⚠️ Sequence appears outside the reference library:</strong><br>
                This sequence does not have sufficient similarity to the current 12S reference library for a reliable species identification (distance &gt; 0.150). Distance Veto has been enforced: classifier probabilities cannot override sequence divergence. Geographic distribution projection is unavailable.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # 3. Known or Low Confidence Match
    species_name = prediction.get("species") or "Species unassigned"
    genus_name = prediction.get("genus") or "Genus unassigned"
    family_name = prediction.get("family") or "Family unassigned"

    st.markdown(
        f"""
        <div class="prediction-hero-card" style="border-left: 4px solid {'#0d9488' if status == 'known_reference_like' else '#f59e0b'};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;">Prediction</div>
                    <div class="prediction-species">{species_name}</div>
                    <div class="prediction-lineage">{genus_name} · {family_name}</div>
                </div>
                <div>
                    {render_status_badge(status)}
                </div>
            </div>
            <div class="kpi-row" style="margin-top: 18px;">
                <div class="kpi-metric">
                    <div class="kpi-label">Confidence</div>
                    <div class="kpi-value" style="color: {'#0d9488' if status == 'known_reference_like' else '#b45309'};">{f"{p1:.1%}" if p1 is not None else "N/A"}</div>
                </div>
                <div class="kpi-metric">
                    <div class="kpi-label">Similarity</div>
                    <div class="kpi-value">{f"{sim:.1%}" if sim is not None else "N/A"}</div>
                </div>
                <div class="kpi-metric">
                    <div class="kpi-label">Distance</div>
                    <div class="kpi-value">{f"{d_min:.3f}" if d_min is not None else "N/A"}</div>
                </div>
                <div class="kpi-metric">
                    <div class="kpi-label">Nearest Barcode</div>
                    <div class="kpi-value" style="font-size: 1.05rem; padding-top: 4px;"><code>{nearest_ref}</code></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if status == "low_confidence":
        st.markdown(
            """
            <div class="clean-alert-warning">
                <strong>⚠️ Low confidence match:</strong><br>
                The query sequence is similar to cataloged records but exhibits intermediate divergence or a narrow classifier probability margin. Scientific evidence details are available below.
            </div>
            """,
            unsafe_allow_html=True
        )

