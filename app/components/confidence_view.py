"""
app/components/confidence_view.py
Phase 9: Confidence and Scientific Evidence View

Renders multi-stream evidence justifying the prediction:
classifier probabilities, margins, entropy, normalized edit distance,
taxonomic concordance, and Distance Veto status.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any


def render_confidence_and_evidence(result: Dict[str, Any]):
    """Render multi-stream confidence metrics and evidence breakdown."""
    confidence = result.get("confidence", {})
    similarity = result.get("similarity", {})
    evidence = result.get("evidence", {})
    prediction = result.get("prediction", {})

    if not confidence or prediction.get("status") == "qc_failed":
        return

    # 1. Metric Cards Row
    p1 = confidence.get("top1_probability")
    margin = confidence.get("probability_margin")
    entropy = confidence.get("entropy")
    d_min = similarity.get("nearest_distance")
    sim = similarity.get("nearest_similarity")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Top-1 Prob</div>
                <div class="kpi-value">{f"{p1:.1%}" if p1 is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Margin (ΔP)</div>
                <div class="kpi-value">{f"{margin:.3f}" if margin is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Entropy (bits)</div>
                <div class="kpi-value">{f"{entropy:.2f}" if entropy is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Distance (d_min)</div>
                <div class="kpi-value">{f"{d_min:.3f}" if d_min is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"""
            <div class="kpi-metric">
                <div class="kpi-label">Similarity</div>
                <div class="kpi-value">{f"{sim:.1%}" if sim is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # 2. Evidence Gauge & Concordance Details
    col_chart, col_concordance = st.columns([1, 1])

    with col_chart:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 8px;'>Evidence Balance</div>", unsafe_allow_html=True)
        
        labels = ["Similarity", "Top-1 Prob", "Margin (ΔP)", "Distance"]
        values = [
            sim if sim is not None else 0.0,
            p1 if p1 is not None else 0.0,
            min(margin if margin is not None else 0.0, 1.0),
            d_min if d_min is not None else 0.0
        ]
        colors = ["#0d9488", "#0284c7", "#f59e0b", "#e11d48" if (d_min or 0) > 0.150 else "#64748b"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker=dict(color=colors, cornerradius=4),
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto"
                )
            ]
        )
        fig.update_layout(
            height=240,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[0, 1.05], title="Normalized Metric", gridcolor="#f1f5f9"),
            xaxis=dict(tickfont=dict(size=11, color="#475569")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.5)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_concordance:
        st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-bottom: 8px;'>Reference Concordance</div>", unsafe_allow_html=True)
        concordance = evidence.get("taxonomic_concordance", {})
        
        ranks = ["class", "order", "family", "genus", "species"]
        for r in ranks:
            is_conc = concordance.get(f"{r}_concordant")
            if is_conc is True:
                icon, text = "✓", "Concordant with top reference"
                color = "#0d9488"
            elif is_conc is False:
                icon, text = "⚠", "Discordant with top reference"
                color = "#e11d48"
            else:
                icon, text = "·", "Unassigned / not evaluated"
                color = "#94a3b8"

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 6px; border-bottom: 1px solid #f1f5f9; font-size: 0.86rem;">
                    <span style="font-weight: 600; text-transform: capitalize; color: #334155;">{r}</span>
                    <span style="color: {color}; font-weight: 500;">{icon} {text}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        if evidence.get("distance_veto_applied"):
            st.markdown(
                """
                <div style="margin-top: 8px; padding: 8px 12px; background: #fff1f2; border: 1px solid #fecdd3; border-radius: 6px; font-size: 0.82rem; color: #9f1239;">
                    <strong>Distance Veto Applied:</strong> Sequence divergence exceeds the 0.150 threshold. Status set to <code>out_of_reference</code>.
                </div>
                """,
                unsafe_allow_html=True
            )

    # 3. Decision Reasons
    reasons = evidence.get("reasons", [])
    if reasons:
        with st.expander("Decision Evidence Trace", expanded=False):
            for r in reasons:
                st.markdown(f"- {r}")
