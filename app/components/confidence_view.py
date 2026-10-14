"""
app/components/confidence_view.py
Phase 9: Confidence and Scientific Evidence View

Renders the multi-stream evidence justifying the prediction:
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

    st.markdown("### 📊 Confidence & Scientific Evidence Analysis")
    st.caption("BioScan integrates independent machine learning probabilities, sequence edit distances, and taxonomic concordance.")

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
            <div class="metric-card">
                <div class="metric-label">Top-1 Probability</div>
                <div class="metric-value">{f"{p1:.1%}" if p1 is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Probability Margin</div>
                <div class="metric-value">{f"{margin:.3f}" if margin is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Entropy (bits)</div>
                <div class="metric-value">{f"{entropy:.2f}" if entropy is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Nearest Distance</div>
                <div class="metric-value">{f"{d_min:.3f}" if d_min is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Nearest Similarity</div>
                <div class="metric-value">{f"{sim:.1%}" if sim is not None else "N/A"}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Evidence Gauge & Concordance Details
    col_chart, col_concordance = st.columns([1, 1])

    with col_chart:
        st.markdown("##### Evidence Balance Chart")
        
        # Plotly Bar chart comparing metrics
        labels = ["Similarity", "Top-1 Prob", "Margin (ΔP)", "Distance (d_min)"]
        values = [
            sim if sim is not None else 0.0,
            p1 if p1 is not None else 0.0,
            min(margin if margin is not None else 0.0, 1.0),
            d_min if d_min is not None else 0.0
        ]
        colors = ["#10b981", "#0284c7", "#f59e0b", "#ef4444" if (d_min or 0) > 0.150 else "#64748b"]

        fig = go.Figure(
            data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto"
                )
            ]
        )
        fig.update_layout(
            height=260,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(range=[0, 1.05], title="Normalized Metric Value"),
            xaxis=dict(tickfont=dict(size=11)),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(248,250,252,0.6)"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_concordance:
        st.markdown("##### Taxonomic Concordance vs Nearest Reference")
        concordance = evidence.get("taxonomic_concordance", {})
        
        ranks = ["class", "order", "family", "genus", "species"]
        for r in ranks:
            is_conc = concordance.get(f"{r}_concordant")
            if is_conc is True:
                icon, text = "✅", "Concordant with top reference"
                color = "#166534"
            elif is_conc is False:
                icon, text = "⚠️", "Discordant with top reference"
                color = "#991b1b"
            else:
                icon, text = "⚪", "Not evaluated / reference missing"
                color = "#64748b"

            st.markdown(
                f"""
                <div style="display: flex; justify-content: space-between; padding: 4px 8px; border-bottom: 1px solid #f1f5f9; font-size: 0.9rem;">
                    <span style="font-weight: 600; text-transform: capitalize;">{r}</span>
                    <span style="color: {color};">{icon} {text}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Distance Veto alert if triggered
        if evidence.get("distance_veto_applied"):
            st.markdown(
                """
                <div style="margin-top: 10px; padding: 8px 12px; background: #fef2f2; border: 1px solid #f87171; border-radius: 6px; font-size: 0.85rem; color: #991b1b;">
                    <strong>🛡️ Distance Veto Applied:</strong> Nearest distance exceeds 0.150 threshold.
                    Status forced to <code>out_of_reference</code> despite classifier posterior probability.
                </div>
                """,
                unsafe_allow_html=True
            )

    # 3. Decision Reasons
    reasons = evidence.get("reasons", [])
    if reasons:
        with st.expander("🔍 Detailed Decision Evidence Log", expanded=False):
            for r in reasons:
                st.markdown(f"- {r}")
