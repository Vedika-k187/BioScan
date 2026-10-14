"""
app/pages/biodiversity.py
Phase 9: Biodiversity Explorer Page

Explores Phase 7 survey analytics, station-level Shannon diversity metrics,
taxonomic richness, and geographic reference survey intelligence.
"""

import streamlit as st
import pandas as pd

from app.utils.ui_helpers import (
    load_reference_barcodes,
    load_diversity_metrics,
    load_species_distribution
)
from src.biodiversity.taxonomy_analysis import compute_taxonomic_richness
from app.components import (
    plot_station_diversity_map,
    plot_diversity_vs_depth
)


def render_biodiversity_page():
    """Render the Biodiversity Intelligence & Survey Explorer Page."""
    st.markdown(
        """
        <div style="margin-bottom: 1.25rem;">
            <div style="font-size: 1.6rem; font-weight: 700; color: #0f172a; letter-spacing: -0.02em;">Biodiversity Explorer</div>
            <div style="font-size: 0.95rem; color: #64748b; margin-top: 4px;">2019 New Jersey Ocean Trawl Survey</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    ref_df = load_reference_barcodes()
    diversity_df = load_diversity_metrics()
    dist_df = load_species_distribution()

    # 1. Compact KPI Row: 101 Barcodes, 90 Species, 87 Genera, 36 Stations
    num_seqs = 101
    num_species = 90
    num_genera = 87
    if not ref_df.empty:
        richness = compute_taxonomic_richness(ref_df)
        num_seqs = richness.get("unique_sequences", 101)
        num_species = richness.get("species_richness", 90)
        num_genera = richness.get("genus_richness", 87)

    num_stations = len(diversity_df) if not diversity_df.empty else 36

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Barcodes</div>
                <div class="kpi-value">{num_seqs}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Species</div>
                <div class="kpi-value">{num_species}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Genera</div>
                <div class="kpi-value">{num_genera}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Stations</div>
                <div class="kpi-value">{num_stations}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 2. Large Interactive Map
    st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;'>Sampling Station Survey Map</div>", unsafe_allow_html=True)
    if not diversity_df.empty:
        fig_map = plot_station_diversity_map(diversity_df)
        st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # 3. Diversity & Taxonomy Insights
    st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 8px;'>Diversity & Depth Insights</div>", unsafe_allow_html=True)
    
    if not diversity_df.empty:
        fig_depth = plot_diversity_vs_depth(diversity_df)
        st.plotly_chart(fig_depth, use_container_width=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("<div style='font-size: 0.92rem; font-weight: 600; color: #334155; margin-bottom: 6px;'>Reference Taxonomy Catalog</div>", unsafe_allow_html=True)
        if not ref_df.empty:
            show_cols = ["sequence_id", "species", "genus", "family", "order", "class"]
            avail_tax_cols = [c for c in show_cols if c in ref_df.columns]
            st.dataframe(ref_df[avail_tax_cols], use_container_width=True, hide_index=True, height=260)

    with col_t2:
        st.markdown("<div style='font-size: 0.92rem; font-weight: 600; color: #334155; margin-bottom: 6px;'>Top Species Survey Prevalence</div>", unsafe_allow_html=True)
        if not dist_df.empty:
            top_dist = dist_df.sort_values(by="station_prevalence_pct", ascending=False).head(15)
            display_dist_cols = ["species", "unique_stations", "station_prevalence_pct", "mean_depth"]
            avail_dist_cols = [c for c in display_dist_cols if c in top_dist.columns]
            st.dataframe(top_dist[avail_dist_cols], use_container_width=True, hide_index=True, height=260)

    # 4. Expandable Methodology & Disclaimers
    with st.expander("▸ Survey Methodology & Operational Caveats", expanded=False):
        st.markdown(
            """
            - **Survey Origin**: 2019 New Jersey Ocean Trawl eDNA Survey across 36 coastal stations.
            - **Shannon Diversity ($H_{\\text{reads}}$)**: Abundance-weighted index calculated using normalized operational eDNA read counts per station.
            - **Amplification Signal Caveat**: eDNA read counts represent an operational PCR/sequencing amplification signal and are **not** direct organism abundance measurements.
            - **Geographic Bounding**: Detections reflect survey sample locations, not global species range boundaries.
            """
        )
