# BioScan System Architecture & Dependency Graph

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Phase**: Phase 10 — Productization & Hackathon Release  
**Status**: Production & Evaluation Ready  

---

## 1. System Overview & Philosophy

BioScan is designed around a single guiding principle: **Scientific Transparency Over Black-Box Automation**. Rather than forcing a single flat species classification, the system decomposes the biological identification task into an end-to-end multi-tiered evidence engine:

$$\text{DNA Input} \longrightarrow \text{Strict QC} \longrightarrow \text{Feature Extraction} \longrightarrow \text{Hierarchical ML} \longrightarrow \text{Similarity Search} \longrightarrow \text{Confidence Analysis} \longrightarrow \text{Distance Veto} \longrightarrow \text{Evidence Aggregation} \longrightarrow \text{Biodiversity Intelligence} \longrightarrow \text{UI Presentation}$$

```
+---------------------------------------------------------------------------------------------------+
|                                      BIOSCAN SYSTEM ARCHITECTURE                                  |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ 1. INPUT & QUALITY CONTROL (src/prediction/input_validation.py)                        │
     │    - Normalization: case folding, whitespace stripping                                 │
     │    - Strict Alphabet Validation: {A, C, G, T} only                                     │
     │    - Ambiguity Rejection: IUPAC codes (N, R, Y, etc.) rejected without imputation      │
     │    - Barcode Window Warnings: 87–112 bp standard reference baseline                    │
     └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                 │
                                                 ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ 2. CANONICAL FEATURE EXTRACTION (src/features/dna_features.py)                         │
     │    - Basic compositional features (GC%, skew, purine/pyrimidine ratio)                 │
     │    - Multi-scale k-mer representations (k=2, 3, 4, 5 frequency distributions)          │
     │    - Strict Zero-Leakage: No occurrence IDs, coordinates, or metadata in feature space │
     └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                 │
                                                 ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ 3. HIERARCHICAL TAXONOMIC CLASSIFICATION (models/taxonomy_*_model.joblib)              │
     │    - Level-wise Parent-Restricted Inference:                                           │
     │      Kingdom (Animalia) ➔ Class ➔ Order ➔ Family ➔ Genus ➔ Species                     │
     │    - Partial Reference Handling: Preserves "Not available in reference data"           │
     │    - Singleton Annotation: 89/90 species singletons (reference-library constrained)    │
     └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                 │
                                                 ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ 4. REFERENCE SIMILARITY SEARCH (src/similarity/reference_search.py)                    │
     │    - Metric: Normalized Levenshtein edit distance: d_norm = edit_dist / max(len1, len2)│
     │    - Reference Library: 101 unique 12S V5 reference barcodes                           │
     │    - Top-K Retrieval: Top-1, Top-3, Top-5 nearest reference sequences                  │
     │    - Exact Match Flagging: Distance = 0.000 identifier                                 │
     └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                 │
                                                 ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ 5. CONFIDENCE ANALYSIS & UNKNOWN DETECTION (src/prediction/confidence.py)              │
     │    - Probabilistic Signals: Top-1 Probability (P1), Margin (ΔP), Shannon Entropy (H)   │
     │    - Multi-Rank Taxonomic Concordance: Class, Order, Family agreement vs Top Reference │
     │    - Operational Tiers:                                                                │
     │      * KNOWN_REFERENCE_LIKE: d_min ≤ 0.050 or exact match, concordant taxonomy         │
     │      * LOW_CONFIDENCE: 0.050 < d_min ≤ 0.150, or margin < 0.150, or discordance        │
     │      * OUT_OF_REFERENCE: d_min > 0.150 (regardless of classifier probability)          │
     │    - DISTANCE VETO: Classifier confidence cannot override sequence divergence          │
     └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                 │
                                                 ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ 6. BIODIVERSITY INTELLIGENCE LAYER (src/biodiversity/)                                 │
     │    - Ground-truth Occurrence Lookup: 2019 New Jersey Ocean Trawl Survey                │
     │    - Station-level Shannon Diversity (H_reads) & Simpson Evenness                      │
     │    - Geographic Safeguard: Distribution projection strictly suppressed for Out-of-Ref  │
     │    - Survey-Bounded Claims: Observations represent survey detections, not global range │
     └───────────────────────────────────────────┬────────────────────────────────────────────┘
                                                 │
                                                 ▼
     ┌────────────────────────────────────────────────────────────────────────────────────────┐
     │ 7. INTERACTIVE PRESENTATION LAYER (app/)                                               │
     │    - Streamlit Wide-Layout UI: Identify DNA, Batch Analysis, Biodiversity, About      │
     │    - Interactive Plotly Visualizations: Maps, Gauge Charts, Composition Bars           │
     │    - Flat Tabular Export: CSV download with full diagnostic transparency               │
     └────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase-by-Phase Artifact Traceability Graph

Every component in BioScan traces directly to a validated experimental phase:

| Phase | Milestone Name | Output Code Module | Persisted Artifact | Description & Role |
|---|---|---|---|---|
| **Phase 0** | Dataset Profiling & Audit | `notebooks/`, `scripts/` | `reports/DATA_PROFILE.md` | Initial dataset audit, duplicate analysis, and raw SHA-256 baseline |
| **Phase 1** | Reference Barcode Construction & QC | `src/data/sequence_qc.py` | `data/processed/reference_barcodes.csv` | 101 unique 12S V5 reference barcodes, length bounds (87–112 bp) |
| **Phase 2** | DNA Feature Extraction | `src/features/dna_features.py` | `data/processed/features_*.csv`, `feature_metadata.json` | 17 basic features, k-mer distributions (k=2..5), zero-leakage check |
| **Phase 3** | Baseline ML Classifiers | `src/models/train_baselines.py` | `models/baseline_knn.joblib`, `models/baseline_random_forest.joblib` | K-Nearest Neighbors & Random Forest multiclass benchmarks |
| **Phase 4** | XGBoost Classification | `src/models/train_xgboost.py` | `models/xgboost_species.joblib` | Gradient boosted tree evaluation on high-dimensional k-mer spaces |
| **Phase 5** | Hierarchical Taxonomy Inference | `src/models/train_hierarchical.py` | `models/taxonomy_*_model.joblib` (5 models) | Parent-restricted classifiers across Class, Order, Family, Genus, Species |
| **Phase 6** | Similarity Search & Unknown Detection | `src/similarity/reference_search.py`, `src/prediction/confidence.py` | Operational thresholds ($d \le 0.05, d > 0.15$) | Normalized Levenshtein search and the Distance Veto mechanism |
| **Phase 7** | Biodiversity Intelligence & Analytics | `src/biodiversity/` | `data/processed/diversity_metrics.csv`, `biodiversity_reference_records.csv` | Station Shannon diversity indices and New Jersey survey occurrence records |
| **Phase 8** | End-to-End Prediction Engine | `src/prediction/` | `src/prediction/predictor.py`, `batch_predict.py` | Unified public API (`predict_sequence`, `predict_fasta`, batch execution) |
| **Phase 9** | Interactive Intelligence Dashboard | `app/` | `app/app.py`, `app/components/`, `app/pages/` | Streamlit multi-page dashboard with Plotly interactive visualization |
| **Phase 10** | Release Polish & Documentation | Repository Scaffolding | `README.md`, `reports/*.md`, `tests/` | Complete release cards, pitch, demo guide, and 115 passing tests |

---

## 3. Detailed Data & Inference Flows

### 3.1 Single Sequence Prediction Flow (`predict_sequence`)
1. **Input Normalization**: Strips whitespace, uppercase conversion.
2. **Biological QC**: Validates character membership in $\{A, C, G, T\}$. If degenerate bases (e.g. `N`) or non-bases exist, aborts immediately returning structured `qc_failed` result.
3. **Feature Encoding**: Computes basic compositional metrics and canonical k-mer frequency representations.
4. **Hierarchical ML Inference**: Evaluates models top-down (Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species) restricting child class probabilities to valid taxonomic descendants.
5. **Similarity Retrieval**: Calculates normalized Levenshtein distances against all 101 cataloged references and sorts Top-5.
6. **Multi-Stream Confidence & Distance Veto**:
   - Calculates $P_1$, $\Delta P$, $H$, and taxonomic concordance.
   - **Enforces Distance Veto**: If $d_{\text{min}} > 0.150$, status is strictly forced to `out_of_reference`, overriding classifier probability.
7. **Biodiversity Context Retrieval**:
   - If `known_reference_like` or `low_confidence`: retrieves survey detections, stations, depth range.
   - If `out_of_reference`: strictly sets `projection_status = "NO_PROJECTION_PERMITTED"`.

### 3.2 Batch & FASTA Analysis Flow (`predict_fasta`)
1. Multi-line FASTA string or file parsed into discrete records with header preservation.
2. Each record is processed independently through the prediction engine.
3. **Fault Isolation**: An invalid sequence (e.g. containing `N`) in record 2 produces a structured QC failure without disrupting or terminating record 3.
4. Structured results compiled into a flat Pandas DataFrame.
5. Flat CSV export rendered for user download.

### 3.3 Dashboard Presentation Flow (`app/`)
- Streamlit application initializes once using `@st.cache_resource` for the `BioScanPredictor` singleton.
- Static datasets loaded once into memory via `@st.cache_data`.
- UI routes between:
  - 🔬 **Identify DNA**: Single sequence classification and evidence explorer.
  - 📊 **Batch Analysis**: Multi-record FASTA upload and KPI dashboard.
  - 🌍 **Biodiversity Explorer**: Reference catalog taxonomy browser and station Shannon diversity map.
  - ℹ️ **About BioScan**: Architecture, Distance Veto rationale, and explicit scientific limitations.
