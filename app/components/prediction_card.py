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
    evidence = result.get("evidence", {})
    similarity = result.get("similarity", {})
    
    status = prediction.get("status") if prediction else result.get("status", "qc_failed")
    cfg = STATUS_CONFIG.get(status, STATUS_CONFIG["qc_failed"])
    
    # Header container
    st.markdown("### 🧬 Biological Identification Summary")
    
    # 1. QC Failed State
    if status == "qc_failed":
        st.markdown(render_status_badge("qc_failed"), unsafe_allow_html=True)
        st.markdown("")
        st.error(
            "**Sequence Biological QC Failed**: The input sequence could not be processed because it violates "
            "canonical DNA requirements (A, C, G, T only) or is empty. BioScan strictly rejects ambiguous "
            "IUPAC codes without silent imputation to preserve scientific integrity."
        )
        errors = qc.get("errors", [])
        if errors:
            st.markdown("**QC Failure Reasons:**")
            for err in errors:
                st.markdown(f"- ❌ `{err}`")
        return

    # 2. Status Badge & Description
    col_badge, col_info = st.columns([1, 2])
    with col_badge:
        st.markdown(render_status_badge(status), unsafe_allow_html=True)
    with col_info:
        st.caption(cfg["description"])

    st.markdown("---")

    # 3. Status-Specific Mandatory Scientific Caveats
    if status == "out_of_reference":
        st.markdown(
            """
            <div class="alert-box">
                <strong>⚠️ Sequence appears outside the current reference library.</strong><br>
                This does not prove that the sequence represents a new species. It indicates that the sequence 
                is sufficiently distant from the current BioScan reference library (normalized distance &gt; 0.150).
                Distance Veto has been enforced: posterior classifier probabilities cannot override sequence divergence.
            </div>
            """,
            unsafe_allow_html=True
        )
    elif status == "low_confidence":
        st.markdown(
            """
            <div class="warning-box">
                <strong>⚠️ BioScan could not establish a strong reference-library match.</strong><br>
                The query sequence is moderately diverged from reference barcodes (0.050 &lt; distance &le; 0.150)
                or exhibits a narrow classifier probability margin (&lt; 0.150) or taxonomic disagreement across ranks.
            </div>
            """,
            unsafe_allow_html=True
        )
    elif status == "known_reference_like":
        st.markdown(
            """
            <div class="success-box">
                <strong>✅ Reference-Library Match</strong><br>
                Sequence exhibits high sequence identity to cataloged 12S V5 reference barcodes (distance &le; 0.050)
                and concordance with reference taxonomy. Note: Represents reference-library matching, not biological proof 
                of complete species boundaries.
            </div>
            """,
            unsafe_allow_html=True
        )

    # 4. Main Taxonomic Predictions Grid
    species_name = prediction.get("species") or "Not available in reference data"
    genus_name = prediction.get("genus") or "Not available in reference data"
    family_name = prediction.get("family") or "Not available in reference data"
    order_name = prediction.get("order") or "Not available in reference data"
    class_name = prediction.get("class") or "Not available in reference data"

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Predicted Species Level")
        if species_name != "Not available in reference data":
            st.markdown(f"### *{species_name}*")
        else:
            st.markdown("### *Not available*")
        st.caption("Scope: `reference-library constrained` (89 of 90 cataloged species are singletons)")

        st.markdown("##### Genus")
        st.markdown(f"**{genus_name}**")

        st.markdown("##### Family")
        st.markdown(f"**{family_name}**")

    with col2:
        st.markdown("##### Order")
        st.markdown(f"**{order_name}**")

        st.markdown("##### Class")
        st.markdown(f"**{class_name}**")

        st.markdown("##### Kingdom")
        st.markdown("**Animalia**")
        
        nearest_ref = similarity.get("nearest_reference_id", "N/A")
        st.caption(f"Top Reference Barcode: `{nearest_ref}`")
