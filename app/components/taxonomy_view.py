"""
app/components/taxonomy_view.py
Phase 9: Visual Taxonomic Hierarchy View

Displays the hierarchical progression from Class down to Species,
clearly marking ranks where reference data is available vs unassigned.
"""

import streamlit as st
from typing import Dict, Any


def render_taxonomy_hierarchy(result: Dict[str, Any]):
    """Render the multi-level taxonomic hierarchy cards."""
    prediction = result.get("prediction", {})
    if not prediction or prediction.get("status") == "qc_failed":
        return

    status = prediction.get("status")

    st.markdown(
        """
        <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em; margin-bottom: 6px;">
            Taxonomic Lineage
        </div>
        """,
        unsafe_allow_html=True
    )

    if status == "out_of_reference":
        st.markdown(
            """
            <div style="padding: 10px 14px; background: #fff1f2; border: 1px solid #fecdd3; border-radius: 8px; font-size: 0.88rem; color: #9f1239;">
                <strong>Taxonomic lineage unverified:</strong> This sequence exceeds the reference divergence threshold. Speculative higher-rank classification is suppressed for out-of-reference queries.
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    ranks = [
        ("Class", prediction.get("class")),
        ("Order", prediction.get("order")),
        ("Family", prediction.get("family")),
        ("Genus", prediction.get("genus")),
        ("Species", prediction.get("species")),
    ]

    crumbs = []
    for rank, val in ranks:
        is_avail = bool(val and val != "Not available in reference data")
        display_val = val if is_avail else "Unassigned"
        val_style = "font-style: italic;" if rank == "Species" and is_avail else ""
        color = "#0f172a" if is_avail else "#94a3b8"

        crumbs.append(
            f'<div class="tax-crumb">'
            f'<span class="tax-crumb-rank">{rank}:</span> '
            f'<span class="tax-crumb-val" style="{val_style} color: {color};">{display_val}</span>'
            f'</div>'
        )

    separator = ' <span class="tax-arrow">&rarr;</span> '
    trail_html = f'<div class="taxonomy-trail">{separator.join(crumbs)}</div>'
    st.markdown(trail_html, unsafe_allow_html=True)
