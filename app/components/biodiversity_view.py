"""
app/components/biodiversity_view.py
Phase 9: Biodiversity & Geographic Context View

Integrates Phase 7 biodiversity survey occurrence intelligence with
Plotly geographic mapping, strictly adhering to scientific boundaries.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any


def render_biodiversity_context(result: Dict[str, Any], bio_records_df: pd.DataFrame):
    """Render biodiversity occurrence intelligence and survey detection map."""
    prediction = result.get("prediction", {})
    biodiversity = result.get("biodiversity", {})
    similarity = result.get("similarity", {})
    status = prediction.get("status") if prediction else result.get("status")

    if not prediction or status == "qc_failed":
        return

    st.markdown("### 🌍 Biodiversity & Survey Occurrence Context")
    st.caption("Ground-truth occurrence context derived from the 2019 New Jersey Ocean Trawl eDNA Survey.")

    # 1. HARD RULE: Out of reference -> strictly suppress distribution projection
    if status == "out_of_reference":
        st.markdown(
            """
            <div class="warning-box">
                <strong>🚫 Geographic distribution projection unavailable:</strong><br>
                The query sequence is outside the current reference library. Projecting an occurrence map 
                or habitat distribution for an unverified taxon is scientifically unsupportable and is 
                strictly suppressed by BioScan.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # 2. Known / Low Confidence Context
    survey_ctx = biodiversity.get("reference_survey_context", {})
    bbox = biodiversity.get("geographic_bounding_box", {})
    nearest_ref = similarity.get("nearest_reference_id")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Survey Detections</div>
                <div class="metric-value">{survey_ctx.get('total_reference_detections', 'N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Distinct Stations</div>
                <div class="metric-value">{survey_ctx.get('unique_stations_count', 'N/A')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Reference Locality</div>
                <div class="metric-value" style="font-size: 1.05rem; padding-top: 6px;">New Jersey Coast</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. Interactive Plotly Map of Reference Detections
    if nearest_ref and not bio_records_df.empty and "sequence_id" in bio_records_df.columns:
        ref_detections = bio_records_df[bio_records_df["sequence_id"] == nearest_ref].copy()
        
        if not ref_detections.empty and "decimalLatitude" in ref_detections.columns:
            st.markdown(f"##### Reference Barcode (`{nearest_ref}`) Detections in Survey")
            
            scatter_fn = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
            style_kw = {"map_style": "open-street-map"} if hasattr(px, "scatter_map") else {"mapbox_style": "open-street-map"}

            fig = scatter_fn(
                ref_detections,
                lat="decimalLatitude",
                lon="decimalLongitude",
                hover_name="sampname",
                hover_data={"organismQuantity": ":,.0f", "depth": ":.1f m", "decimalLatitude": False, "decimalLongitude": False},
                size="organismQuantity",
                color_discrete_sequence=["#0d9488"],
                zoom=6.5,
                center=dict(lat=39.5, lon=-74.3),
                title=f"Survey Detection Locations for {nearest_ref} (Marker size proportional to eDNA read signal)",
                **style_kw
            )
            fig.update_layout(
                height=380,
                margin=dict(l=10, r=10, t=35, b=10),
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

    # 4. Mandatory Geographic Limitation Notice
    st.info(
        "ℹ️ **Geographic Interpretation Disclaimer**: Points represent sampling stations where this 12S barcode was "
        "observed during the 2019 New Jersey Ocean Trawl Survey. BioScan does **NOT** project global species ranges, "
        "and lack of detection does not indicate true biological absence outside the survey perimeter."
    )
