"""
app/components/similarity_view.py
Phase 9: Top-K Reference Similarity Results View

Displays the closest reference sequences from the 101 barcode library
retrieved via normalized Levenshtein distance, highlighting Rank 1.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any


def render_similarity_table(result: Dict[str, Any]):
    """Render Top-K similarity table and distance ranking."""
    similarity = result.get("similarity", {})
    top_refs = similarity.get("top_references") or similarity.get("top_k_references", [])
    prediction = result.get("prediction", {})

    if not top_refs or (prediction and prediction.get("status") == "qc_failed"):
        return

    rows = []
    for item in top_refs:
        ref_tax = item.get("reference_taxonomy") or item
        is_exact = item.get("is_exact_match", False)
        
        rows.append({
            "Rank": item.get("rank"),
            "Reference ID": item.get("sequence_id"),
            "Species": (ref_tax.get("species") if ref_tax.get("species") not in (None, "Not available in reference data") else "Unlabeled"),
            "Similarity": f"{item.get('similarity', 0.0):.1%}",
            "Distance": f"{item.get('distance', 0.0):.3f}",
            "Edit Dist": item.get("edit_distance", 0),
            "Exact": "✓ Yes" if is_exact else "No",
            "Genus": ref_tax.get("genus") or "—",
            "Family": ref_tax.get("family") or "—",
            "Order": ref_tax.get("order") or "—",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    nearest_item = top_refs[0] if top_refs else {}
    if nearest_item.get("is_exact_match"):
        st.markdown(
            f"""
            <div class="clean-alert-success" style="font-size: 0.85rem; padding: 8px 12px;">
                ✓ <strong>Exact reference sequence match:</strong> Identical sequence (distance 0.000) to cataloged barcode <code>{nearest_item.get('sequence_id')}</code> (<em>{(nearest_item.get('reference_taxonomy') or nearest_item).get('species', 'N/A')}</em>).
            </div>
            """,
            unsafe_allow_html=True
        )
