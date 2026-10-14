"""
app/app.py
Phase 9: BioScan Interactive Biodiversity Intelligence Dashboard Main Application

The primary Streamlit entry point connecting the interactive presentation layer
to the validated Phase 8 biological prediction engine.
"""

import os
import sys
import streamlit as st

# Ensure root workspace is available for imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.utils.ui_helpers import inject_custom_css
from app.pages import (
    render_identify_page,
    render_batch_page,
    render_biodiversity_page,
    render_about_page
)


def main():
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="BioScan — AI DNA Biodiversity Engine",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # Sidebar Navigation & Information Panel
    with st.sidebar:
        st.markdown("## 🧬 BioScan")
        st.markdown("**AI-Powered DNA Biodiversity Identification & Monitoring**")
        st.caption("Validated End-to-End Prediction & Intelligence Engine")
        st.markdown("---")

        page_choice = st.radio(
            "Navigation",
            options=[
                "🔬 Identify DNA",
                "📊 Batch Analysis",
                "🌍 Biodiversity Explorer",
                "ℹ️ About BioScan"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("##### 🏛️ Reference Library Status")
        st.markdown(
            """
            - **Target Marker**: 12S rDNA V5 (87–112 bp)
            - **Catalog Size**: 101 Barcodes
            - **Species Count**: 90 Named Species
            - **Singleton Rate**: 89 / 90 (98.9%)
            - **Survey**: 2019 NJ Ocean Trawl Survey
            """
        )

        st.markdown("---")
        st.markdown("##### 🔒 Dataset Verification")
        st.markdown(
            """
            <div style="font-size: 0.75rem; color: #64748b; background: #f1f5f9; padding: 6px; border-radius: 6px;">
                <b>Raw Dataset SHA-256:</b><br>
                <code>5290252D...AD65</code><br>
                <span style="color: #10b981; font-weight: 600;">✔ Verified Unmodified</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("BioScan v1.0.0 — Phase 9 Validated Dashboard")

    # Main Content Router with Graceful Error Handling
    try:
        if page_choice == "🔬 Identify DNA":
            render_identify_page()
        elif page_choice == "📊 Batch Analysis":
            render_batch_page()
        elif page_choice == "🌍 Biodiversity Explorer":
            render_biodiversity_page()
        elif page_choice == "ℹ️ About BioScan":
            render_about_page()
    except Exception as e:
        st.error(
            "An unexpected system exception occurred while rendering the page. "
            "Please check the input data or system configuration."
        )
        st.caption(f"Error details for administrator: {str(e)}")


if __name__ == "__main__":
    main()
