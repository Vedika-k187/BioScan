# BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System

> **A scientifically transparent, evidence-grounded machine learning platform for environmental DNA (eDNA) taxonomic identification, out-of-reference unknown detection, and marine biodiversity intelligence.**

---

## 📌 Project Status
**Research Prototype / Hackathon-Ready System**  
BioScan is an applied academic and conservation prototype built to evaluate transparent AI methodology on regional marine environmental DNA surveys. It is **not** a clinical-grade diagnostic tool or regulatory seafood certification instrument.

- **Verified Test Suite**: **115 passed, 0 failed** across all Phases 1–9.
- **Raw Dataset Integrity**: Bit-for-bit SHA-256 verified (`5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65`).
- **Interactive UI**: Fully functional Streamlit & Plotly dashboard (`streamlit run app/app.py`).

---

## 🌊 The Problem

Environmental DNA (eDNA)—genetic material shed by organisms into water, soil, and air—enables rapid, non-invasive biomonitoring. However, traditional bioinformatics and contemporary machine learning tools suffer from critical vulnerabilities:
1. **Opaque Black-Box Predictions**: Standard classifiers return an isolated species label with zero confidence context, sequence edit metrics, or alternative candidates.
2. **Overconfidence on Novel Taxa**: Closed-set multiclass models force all posterior probabilities to sum to 1.0, frequently assigning $>95\%$ confidence to completely uncataloged, divergent, or synthetic organisms.
3. **Singleton Sparsity**: Real-world reference catalogs are heavily singleton-dominated (in our reference dataset, **89 of 90 species have only 1 reference sequence**). Standard claims of "100% generalized species accuracy" on singletons confuse library re-identification with real-world biological generalization.

---

## 💡 The BioScan Solution

BioScan transforms the biological classification task from a single flat classifier into an **end-to-end multi-tiered evidence engine**. When presented with a DNA sequence, BioScan:
1. Validates canonical nucleotides $\{A, C, G, T\}$ under strict biological quality control.
2. Generates zero-leakage compositional and k-mer frequency representations.
3. Infers taxonomy top-down across 5 hierarchical ranks (Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species).
4. Executes normalized Levenshtein sequence search across cataloged reference barcodes.
5. Evaluates multi-stream confidence (posterior probability, margin, entropy, concordance).
6. Enforces the **Distance Veto**: strictly overrides model probability and assigns `out_of_reference` when sequence divergence exceeds 15% ($d_{\text{min}} > 0.150$).
7. Integrates ground-truth field occurrence records from the 2019 New Jersey Ocean Trawl Survey to map station-level biodiversity.

---

## 🚀 Key Features

- **Biological Quality Control (QC)**: Validates canonical bases, rejects ambiguous IUPAC codes (`N`, `R`, `Y`, etc.) without silent imputation, and flags atypical barcode lengths (87–112 bp standard window).
- **Zero-Metadata Feature Extraction**: Strict barrier separating DNA features from survey metadata (no occurrence IDs or coordinates used as ML features).
- **Hierarchical Taxonomic ML**: Parent-restricted models preserving internal biological consistency down to species.
- **Normalized Levenshtein Similarity Search**: Pairwise edit distance retrieval returning Top-5 closest cataloged barcodes with exact match identification.
- **Multi-Stream Confidence Framework**: Integrates Top-1 probability ($P_1$), probability margin ($\Delta P = P_1 - P_2$), Shannon entropy ($H$), and rank-by-rank taxonomic concordance.
- **The Distance Veto (Unknown Detection)**: Independent sequence edit distance veto ($d_{\text{min}} > 0.150$) that prevents closed-set classifier overconfidence on uncataloged sequences.
- **Biodiversity Survey Intelligence**: Connects reference-matched barcodes to 1,604 field occurrence records across 36 sampling stations.
- **Station-Level Shannon Diversity**: Abundance-weighted Shannon diversity ($H_{\text{reads}}$) and Simpson evenness indices across oceanographic stations.
- **Strict Geographic Suppression**: Strictly forbids projecting geographic presence or survey detections for any sequence flagged as `out_of_reference`.
- **Fault-Tolerant Batch Analysis**: Multi-sequence FASTA parsing where a malformed record in sample 2 never disrupts sample 3.
- **Interactive Streamlit Dashboard**: Polished scientific user interface with dynamic example sequences, Plotly interactive maps, and flat CSV tabular export.

---

## 🏛️ System Architecture

```
DNA Input (Single Sequence or Multi-Record FASTA)
   │
   ▼
[1. Biological Quality Control] ──(Invalid)──> Structured QC Failure Result
   │  {A,C,G,T} check, 87-112 bp window
   ▼
[2. DNA Feature Extraction] (Compositional & Multi-Scale K-mer Distributions)
   │  Zero-metadata leakage guarantee
   ▼
[3. Hierarchical Taxonomic Inference] (Parent-Restricted Classifiers)
   │  Kingdom (Animalia) ➔ Class ➔ Order ➔ Family ➔ Genus ➔ Species
   ▼
[4. Reference Similarity Search] (Normalized Levenshtein Search)
   │  Pairwise retrieval against 101 barcodes, Top-5 nearest candidates
   ▼
[5. Confidence Analysis & Unknown Detection]
   │  P1 Probability, Margin (ΔP), Shannon Entropy (H), Concordance
   │  DISTANCE VETO: If d_min > 0.150 ➔ Strictly OUT_OF_REFERENCE
   ▼
[6. Biodiversity Intelligence Integration] (2019 NJ Ocean Trawl Survey Records)
   │  Station detections, Shannon diversity, geographic suppression for out-of-ref
   ▼
[7. Interactive Presentation Layer] (Streamlit Web Application)
   │  Identify DNA, Batch FASTA Analysis, Biodiversity Explorer, About Page
```

---

## 📊 Dataset Scope & Characterization

BioScan operates on a carefully audited marine vertebrate reference dataset:
- **Marker Locus**: Mitochondrial 12S ribosomal RNA (12S rDNA V5 hypervariable loop, 87–112 bp, median 104 bp).
- **Taxonomic Domain**: Exclusively teleost and elasmobranch marine fishes.
- **Total Unique Barcodes**: **101 unique reference sequences** (derived from 13,736 survey rows).
- **Taxonomic Coverage**:
  - **Kingdom**: 1 (*Animalia*)
  - **Classes**: 4 (*Teleostei* [97], *Elasmobranchii* [2], *Chondrostei* [1], *Petromyzonti* [1])
  - **Orders**: 30 (*Perciformes*, *Pleuronectiformes*, *Clupeiformes*, etc.)
  - **Families**: 66 (*Gobiidae*, *Sciaenidae*, *Paralichthyidae*, etc.)
  - **Genera**: 87 (73 singleton genera, 8 multi-species genera)
  - **Named Species**: **90 species (89 singletons, 1 doubleton)**
  - **Unlabeled Taxa**: 10 reference records retain valid higher taxonomy but lack species-level assignment.
- **Survey Provenance**: 2019 New Jersey Bureau of Marine Fisheries Ocean Trawl Survey (NJ OTS) deposited in NCBI BioProject **PRJNA638560** (68 water samples across 36 distinct sampling stations).
- **Scope Notice**: BioScan is **NOT** a universal biodiversity classifier. It cannot identify birds, mammals, terrestrial wildlife, plants, fungi, or invertebrates without reference database expansion.

---

## 🧠 Machine Learning & Inference Pipeline

1. **Feature Engineering**: Encodes sequences into 17 basic compositional features (GC content, nucleotide skews, purine/pyrimidine ratios) and multi-scale k-mer frequency representations ($k=2, 3, 4, 5$).
2. **Hierarchical Taxonomy**: Evaluates parent-restricted classifiers sequentially down the taxonomic tree. Species-level identification is explicitly constrained to reference-library matching because 89 of 90 species are singletons.
3. **Similarity Search**: Pairwise normalized edit distance:
   $$d_{\text{norm}}(s_1, s_2) = \frac{\text{Levenshtein}(s_1, s_2)}{\max(|s_1|, |s_2|)}, \quad \text{Similarity} = 1.0 - d_{\text{norm}}$$
4. **Unknown Detection (Distance Veto)**:
   - $d_{\text{min}} \le 0.050$ (or exact match) AND concordant taxonomy $\implies$ `known_reference_like`
   - $0.050 < d_{\text{min}} \le 0.150$ or margin $< 0.150 \implies$ `low_confidence`
   - $d_{\text{min}} > 0.150 \implies$ `out_of_reference` (Distance Veto enforced)

> **Important Unknown-Detection Notice**:  
> An `out_of_reference` status **does not prove the discovery of a biologically new species**. It indicates that the sequence diverges by $>15\%$ from all 101 barcodes in the current BioScan reference library.

---

## 🌍 Biodiversity Intelligence & Ecological Context

- **Survey Occurrences**: Cross-references verified reference barcodes with 1,604 deduplicated field detection events from the 2019 New Jersey survey.
- **Station-Level Diversity**: Aggregates eDNA detection signals across 36 oceanographic sampling stations to compute:
  - Abundance-weighted Shannon diversity ($H_{\text{reads}}$)
  - Incidence-based Shannon diversity ($H_{\text{incidence}}$)
  - Simpson diversity index ($1 - D$)
  - Pielou's evenness ($J$)
- **Ecological Signal Clarification**: eDNA read counts (`organismQuantity`) represent an operational PCR/sequencing amplification signal and **must not be interpreted as absolute organism abundance or biomass**.
- **Geographic Bounding**: Detection coordinates reflect regional survey sample points, **not** the global biological species distribution or complete native range.

---

## 💻 Quick Start & Dashboard Launch

### 1. Environment Setup
BioScan runs on Python 3.10:

```bash
# Clone the repository
git clone https://github.com/your-username/BioScan.git
cd BioScan

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
Launch the interactive web application with a single command:

```bash
streamlit run app/app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

### 3. Running the Test Suite
Execute the complete automated regression suite across all 9 phases:

```bash
pytest -v
```

Expected result: **115 passed, 0 failed** in ~22 seconds.

---

## 📁 Repository Structure

```
BioScan/
├── app/                                # Streamlit Interactive Dashboard
│   ├── app.py                          # Main Streamlit application entry point
│   ├── components/                     # Reusable UI presentation components
│   │   ├── prediction_card.py          # Primary identification summary card
│   │   ├── taxonomy_view.py            # Visual multi-rank hierarchy view
│   │   ├── confidence_view.py          # Multi-stream evidence & concordance
│   │   ├── similarity_view.py          # Top-5 reference similarity table
│   │   ├── biodiversity_view.py        # Occurrence metadata & survey map
│   │   └── charts.py                   # Plotly charts (status, taxonomy, map)
│   ├── pages/                          # Application view controllers
│   │   ├── identify.py                 # Single DNA sequence identification
│   │   ├── batch_analysis.py           # Multi-record FASTA batch prediction
│   │   ├── biodiversity.py             # Reference catalog & station diversity
│   │   └── about.py                    # Architecture, Distance Veto & limitations
│   └── utils/                          # Caching loaders & dynamic examples
├── data/
│   ├── raw/                            # Original reference survey dataset
│   │   └── hackathon_dna_taxonomy_merged.csv  (SHA-256 verified)
│   └── processed/                      # Curated barcodes, features, diversity tables
├── models/                             # Trained hierarchical taxonomy models (.joblib)
├── notebooks/                          # Research exploration & validation notebooks
├── reports/                            # Comprehensive milestone reports (Phases 0–10)
│   ├── ARCHITECTURE.md                 # Complete system architecture & dependency graph
│   ├── MODEL_CARD.md                   # Formal AI Model Card (limitations & intended use)
│   ├── DATASET_CARD.md                 # Formal Dataset Card (provenance & structure)
│   ├── DEMO_GUIDE.md                   # 3–5 minute hackathon live demonstration guide
│   ├── PITCH.md                        # 30s, 60s, and 2-min presentation pitches
│   ├── HACKATHON_FEATURES.md           # Judge-facing feature matrix & innovations
│   └── PHASE10_FINAL_RELEASE.md        # Final productization & release report
├── src/                                # Core scientific pipeline modules
│   ├── data/                           # Sequence QC & barcode cleaning
│   ├── features/                       # DNA compositional & k-mer feature extraction
│   ├── models/                         # Baseline, XGBoost & hierarchical training
│   ├── prediction/                     # End-to-end inference engine & FASTA batching
│   ├── similarity/                     # Normalized Levenshtein reference search
│   └── biodiversity/                   # Shannon diversity & geographic analysis
├── tests/                              # Automated unit & integration test suites
│   ├── test_phase1_data.py
│   ├── test_phase2_features.py
│   ├── test_phase3_models.py
│   ├── test_phase4_xgboost.py
│   ├── test_phase5_taxonomy.py
│   ├── test_phase6_prediction.py
│   ├── test_phase7_biodiversity.py
│   ├── test_phase8_prediction_engine.py
│   └── test_phase9_dashboard.py
├── requirements.txt                    # Python package dependencies
├── LICENSE                             # MIT Open-Source License & Dataset Notice
└── README.md                           # Project documentation (this file)
```

---

## ⚠️ Scientific Limitations & Ethical Guardrails

1. **Target Marker**: BioScan is calibrated strictly for the mitochondrial 12S rDNA V5 hypervariable loop (87–112 bp). It cannot classify COI, 16S, 18S, or ITS sequences without reference reconstruction.
2. **Taxonomic Domain**: Restricted to teleost and elasmobranch marine fishes present in the regional catalog (101 barcodes, 90 species). It cannot classify mammals, birds, plants, or invertebrates.
3. **Singleton Constraint**: 89 of 90 species classes are singletons. Species-level inference represents library re-identification rather than broad biological generalization.
4. **Distance Veto**: Any sequence diverging by $>15\%$ normalized Levenshtein distance is labeled `out_of_reference` to prevent closed-set overconfidence.
5. **No Geographic Range Projection**: Detections reflect regional sampling events in the 2019 New Jersey Ocean Trawl Survey, **not** complete global species distribution boundaries.
6. **Amplification vs. Biomass**: eDNA read counts represent PCR/sequencing amplification intensity and are not direct biomass or organism count measurements.

---

## 📜 License & Citation

BioScan is released under the **[MIT License](LICENSE)**.

### Reference Dataset Attribution
- **Survey**: New Jersey Bureau of Marine Fisheries Ocean Trawl Survey (NJ OTS 2019).
- **NCBI Repository**: BioProject PRJNA638560.
