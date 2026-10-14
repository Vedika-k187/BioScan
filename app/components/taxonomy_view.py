"""
app/components/taxonomy_view.py
Phase 9: Visual Taxonomic Hierarchy View

Displays the hierarchical progression from Kingdom down to Species,
clearly marking ranks where reference data is available vs unassigned.
"""

import streamlit as st
from typing import Dict, Any


def render_taxonomy_hierarchy(result: Dict[str, Any]):
    """Render the multi-level taxonomic hierarchy cards."""
    prediction = result.get("prediction", {})
    if not prediction or prediction.get("status") == "qc_failed":
        return

    st.markdown("### 🌲 Taxonomic Hierarchy")
    
    ranks = [
        ("Kingdom", "Animalia"),
        ("Class", prediction.get("class")),
        ("Order", prediction.get("order")),
        ("Family", prediction.get("family")),
        ("Genus", prediction.get("genus")),
        ("Species", prediction.get("species")),
    ]

    cols = st.columns(len(ranks))
    for i, (rank, val) in enumerate(ranks):
        with cols[i]:
            is_avail = bool(val and val != "Not available in reference data")
            rank_bg = "#f0fdf4" if is_avail else "#f8fafc"
            border_color = "#86efac" if is_avail else "#cbd5e1"
            val_display = val if is_avail else "Not available in reference"
            val_style = "font-style: italic;" if rank == "Species" and is_avail else ""

            st.markdown(
                f"""
                <div style="background: {rank_bg}; border: 1px solid {border_color}; 
                            border-radius: 8px; padding: 10px; text-align: center; height: 100%;">
                    <div style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">
                        {rank}
                    </div>
                    <div style="font-size: 0.92rem; font-weight: 600; color: #0f172a; margin-top: 4px; {val_style}">
                        {val_display}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
