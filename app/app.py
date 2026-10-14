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
        st.markdown(
            """
            <div style="padding: 6px 0 16px 0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.5rem;">🧬</span>
                    <span style="font-size: 1.3rem; font-weight: 700; letter-spacing: -0.02em; color: #0f172a;">BioScan</span>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; font-weight: 500; margin-top: 2px;">
                    DNA Biodiversity Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        page_choice = st.radio(
            "Navigation",
            options=[
                "Identify DNA",
                "Batch Analysis",
                "Biodiversity Explorer"
            ],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown(
            """
            <div style="margin-top: 4rem; padding-top: 1rem; border-top: 1px solid #f1f5f9; font-size: 0.75rem; color: #94a3b8; line-height: 1.5;">
                <span style="font-weight: 600; color: #64748b;">12S rDNA V5</span><br>
                2019 NJ Ocean Trawl Survey
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
    except Exception as e:
        st.error(
            "An unexpected system exception occurred while rendering the page. "
            "Please check the input data or system configuration."
        )
        st.caption(f"Error details for administrator: {str(e)}")


if __name__ == "__main__":
    main()
