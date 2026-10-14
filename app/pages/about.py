"""
app/pages/about.py
Phase 9: About BioScan, Architecture, and Scientific Safeguards

Documents the scientific problem, system architecture, methodology,
unknown detection safeguards, and operational limitations.
"""

import streamlit as st


def render_about_page():
    """Render the About BioScan documentation page."""
    st.markdown('<div class="main-header">ℹ️ About BioScan</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered DNA Biodiversity Identification & Monitoring System — '
        'A transparent, evidence-grounded approach to environmental DNA (eDNA) classification.</div>',
        unsafe_allow_html=True
    )

    # 1. Motivation & Problem Statement
    st.markdown("### 🎯 Problem & Scientific Challenge")
    st.write(
        """
        Environmental DNA (eDNA) analysis offers non-invasive biodiversity monitoring, but standard machine learning 
        classifiers suffer from severe failure modes:
        - **Black-box Predictions**: Returning an isolated species name without evidence or taxonomic justification.
        - **Overconfidence on Novel Sequences**: Standard closed-set multiclass models assign high posterior probability 
          even to completely unknown or out-of-reference organisms because softmax probabilities must sum to 1.0.
        - **Singleton Sparsity**: Real-world barcode libraries are dominated by singletons (in our reference dataset, 
          89 of 90 cataloged species have only one reference sequence).
        """
    )

    st.markdown("### 💡 The BioScan Solution")
    st.write(
        """
        BioScan turns machine learning into a transparent, multi-stream biological evidence layer. 
        It integrates parent-restricted hierarchical classification, normalized Levenshtein reference similarity search, 
        multi-stream confidence estimation, an independent **Distance Veto** for unknown detection, and ground-truth 
        biodiversity survey intelligence.
        """
    )

    st.markdown("---")

    # 2. System Architecture Flow
    st.markdown("### 🏛️ Prediction Pipeline Architecture")
    st.code(
        """
        INPUT: DNA Sequence String or FASTA File
           │
           ▼
        [1. Biological Quality Control] ──(Invalid)──> Structured QC Failure
           │ (Valid: strict {A,C,G,T}, 87-112 bp window)
           ▼
        [2. DNA Feature Extraction] (Phase 2 Canonical K-mer & Compositional Features)
           │
           ▼
        [3. Hierarchical ML Taxonomy] (Class ➔ Order ➔ Family ➔ Genus ➔ Species)
           │
           ▼
        [4. Reference Similarity Search] (Normalized Levenshtein, Top-5 Nearest Barcodes)
           │
           ▼
        [5. Confidence & Unknown Detection]
           │  ├─ Top-1 Probability (P1), Margin (ΔP), Shannon Entropy (H)
           │  └─ DISTANCE VETO: If d_min > 0.150 ➔ Strictly OUT_OF_REFERENCE
           ▼
        [6. Evidence Aggregation & Status Synthesis]
           │  ├─ KNOWN_REFERENCE_LIKE (d_min ≤ 0.050 or exact match, concordant taxonomy)
           │  ├─ LOW_CONFIDENCE (0.050 < d_min ≤ 0.150, or margin < 0.150, or discordance)
           │  └─ OUT_OF_REFERENCE (d_min > 0.150, probability veto enforced)
           ▼
        [7. Biodiversity Context Integration] (2019 NJ Ocean Trawl Survey Intelligence)
           │  (Geographic projection strictly suppressed for out_of_reference)
           ▼
        OUTPUT: Transparent Structured Result & Interactive Visualization
        """,
        language="text"
    )

    st.markdown("---")

    # 3. Operational Safeguards & The Distance Veto
    st.markdown("### 🛡️ The Distance Veto Principle")
    st.info(
        "**Why Closed-Set Models Fail Without a Distance Veto**:\n\n"
        "A closed-set classifier trained on 90 species will ALWAYS pick one of the 90 species for any input, "
        "often with >95% probability, even if the input is an insect, a plant, or synthetic random DNA. "
        "BioScan solves this by enforcing an **independent sequence edit distance veto**:\n\n"
        "$$\\text{If } d_{\\text{min}} > 0.150 \\implies \\text{Status } = \\mathbf{out\\_of\\_reference}$$\n\n"
        "High classifier probability CANNOT override sequence divergence."
    )

    st.markdown("---")

    # 4. Mandatory Scientific Boundaries & Limitations
    st.markdown("### ⚠️ Operational Boundaries & Limitations")
    st.markdown(
        """
        To maintain scientific rigor, BioScan explicitly states its operational constraints:
        1. **Target Locus**: The system is trained on 12S mitochondrial rDNA V5 barcodes (87–112 bp). It cannot classify COI, 16S, 18S, or ITS markers without retraining.
        2. **Reference Library Size**: The current reference library contains 101 unique sequences representing 90 named species.
        3. **Taxonomic Domain**: The system is fish-focused (teleosts and elasmobranchs). It cannot classify birds, mammals, plants, or invertebrates.
        4. **Singleton Constraint**: 89 of 90 species have singleton reference sequences. Species-level prediction functions as reference-library matching, not as unconstrained generalization.
        5. **Reference-Constrained Scope**: Predictions reflect identity to cataloged reference barcodes, not unconstrained global taxonomy.
        6. **Survey-Bounded Geography**: Occurrence data reflects sampling detections from the 2019 New Jersey Ocean Trawl Survey, not complete species geographic ranges.
        7. **Unknown Detection Caveat**: An `out_of_reference` status indicates divergence from the current reference library; it does **not** prove the discovery of a biologically new species.
        8. **eDNA Abundance Signal**: eDNA read counts represent an operational PCR/sequencing amplification signal and are **not** direct organism abundance measurements.
        """
    )
