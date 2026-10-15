"""
app/app.py
Phase 9: BioScan Interactive Biodiversity Intelligence Dashboard Main Application

The primary Streamlit entry point connecting the interactive presentation layer
to the validated Phase 8 biological prediction engine.
"""

import os
import sys
import streamlit as st

# Ensure root workspace is available for package imports and eliminate self-shadowing
APP_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

# When Streamlit runs 'streamlit run app/app.py', Python inserts APP_DIR ('.../app')
# into sys.path[0]. Because 'app.py' resides inside 'app/', any subsequent
# 'from app.utils...' statement resolves 'app' to the single-file module 'app.py'
# instead of the package 'app/', triggering:
#   ModuleNotFoundError: No module named 'app.utils'; 'app' is not a package
# Removing APP_DIR from sys.path and placing BASE_DIR at index 0 guarantees that
# 'app' cleanly resolves to the 'app/' package directory.
while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# If 'app' was registered in sys.modules as a standalone script without __path__,
# remove it so Python resolves 'app' as the 'app/' package directory.
if "app" in sys.modules and not hasattr(sys.modules["app"], "__path__"):
    del sys.modules["app"]

from app.utils.ui_helpers import inject_custom_css, bioscan_logo_svg, load_reference_barcodes
from src.prediction.evidence import THRESHOLD_CLOSE_SIMILARITY, THRESHOLD_OUT_OF_REFERENCE
from app.pages import (
    render_identify_page,
    render_batch_page,
    render_biodiversity_page,
    render_about_page,
    render_evaluation_page
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
        st.markdown(
            f"""
            <div style="padding: 2px 0 18px 0;">
                <div class="bs-brand">
                    {bioscan_logo_svg(66)}
                    <span class="bs-name">Bio<span>Scan</span></span>
                </div>
                <div class="bs-tag">DNA biodiversity intelligence</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        page_choice = st.radio(
            "Navigation",
            options=[
                "Identify DNA",
                "Batch Analysis",
                "Biodiversity Explorer",
                "Model Evaluation"
            ],
            index=0,
            label_visibility="collapsed"
        )

        try:
            _ref = load_reference_barcodes()
            _n_bar = len(_ref)
            _n_sp = int(_ref["species"].dropna().nunique())
        except Exception:
            _n_bar, _n_sp = "n/a", "n/a"
        st.markdown(
            f"""
            <div class="bs-foot">
                <div class="bs-card">
                    <div class="bs-h">Reference library</div>
                    <div class="bs-row"><span>Barcodes</span><b>{_n_bar}</b></div>
                    <div class="bs-row"><span>Species</span><b>{_n_sp}</b></div>
                    <div class="bs-row"><span>Marker</span><b>12S rDNA V5</b></div>
                </div>
                <div class="bs-card">
                    <div class="bs-h">Reading a result</div>
                    <div class="bs-leg"><i style="background:#4ade80"></i><span>Known reference<small>distance &le; {THRESHOLD_CLOSE_SIMILARITY:.2f}</small></span></div>
                    <div class="bs-leg"><i style="background:#fbbf24"></i><span>Low confidence<small>{THRESHOLD_CLOSE_SIMILARITY:.2f} to {THRESHOLD_OUT_OF_REFERENCE:.2f}</small></span></div>
                    <div class="bs-leg"><i style="background:#fb7185"></i><span>Out of reference<small>distance &gt; {THRESHOLD_OUT_OF_REFERENCE:.2f}</small></span></div>
                </div>
                <div class="bs-small">2019 NJ Ocean Trawl Survey</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main Content Router with Graceful Error Handling
    try:
        if page_choice == "Identify DNA":
            render_identify_page()
        elif page_choice == "Batch Analysis":
            render_batch_page()
        elif page_choice == "Biodiversity Explorer":
            render_biodiversity_page()
        elif page_choice == "Model Evaluation":
            render_evaluation_page()
    except Exception as e:
        st.error(
            "An unexpected system exception occurred while rendering the page. "
            "Please check the input data or system configuration."
        )
        st.caption(f"Error details for administrator: {str(e)}")


if __name__ == "__main__":
    main()
