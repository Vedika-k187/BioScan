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
    top_refs = similarity.get("top_k_references", [])
    prediction = result.get("prediction", {})

    if not top_refs or (prediction and prediction.get("status") == "qc_failed"):
        return

    st.markdown("### 🔎 Closest Reference Sequences (Top 5)")
    st.caption("Pairwise normalized Levenshtein sequence distance evaluated against all 101 cataloged reference barcodes.")

    rows = []
    for item in top_refs:
        ref_tax = item.get("reference_taxonomy", {})
        is_exact = item.get("is_exact_match", False)
        
        rows.append({
            "Rank": item.get("rank"),
            "Reference ID": item.get("sequence_id"),
            "Similarity": f"{item.get('similarity', 0.0):.1%}",
            "Normalized Distance": f"{item.get('distance', 0.0):.3f}",
            "Edit Distance": item.get("edit_distance", 0),
            "Exact Match": "⭐ Exact" if is_exact else "No",
            "Species": ref_tax.get("species") or "Unlabeled",
            "Genus": ref_tax.get("genus") or "Unlabeled",
            "Family": ref_tax.get("family") or "Unlabeled",
            "Order": ref_tax.get("order") or "Unlabeled",
            "Class": ref_tax.get("class") or "Unlabeled",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    nearest_item = top_refs[0] if top_refs else {}
    if nearest_item.get("is_exact_match"):
        st.info(
            f"⭐ **Exact Reference Sequence Match**: The query is identical (distance 0.000) to reference "
            f"`{nearest_item.get('sequence_id')}` (*{nearest_item.get('reference_taxonomy', {}).get('species', 'N/A')}*). "
            f"Represents reference library re-identification; does not constitute unconstrained real-world generalization."
        )
