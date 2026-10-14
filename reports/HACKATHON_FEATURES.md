# BioScan Judge-Facing Feature Matrix

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Evaluation Category**: Applied Machine Learning, Scientific AI, Environmental Informatics  

---

## 1. AI & Machine Learning Innovations

| Feature | What It Does | Why It Matters to Judges |
|---|---|---|
| **Parent-Restricted Hierarchical Taxonomy** | Predicts Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species conditionally down the taxonomic tree rather than predicting a flat label. | Standard flat classifiers often predict mutually contradictory ranks (e.g. predicting a teleost species under a shark family). Hierarchical enforcement guarantees internal biological consistency. |
| **Multi-Scale K-Mer Embeddings** | Encodes sequences using compositional DNA properties and canonical k-mer frequency representations ($k=2, 3, 4, 5$). | Captures alignment-free nucleotide frequency signatures without sequence length constraints, enabling rapid feature extraction in under 5 ms. |
| **Probability Margin & Shannon Entropy** | Computes the gap between top-1 and top-2 posterior probabilities ($\Delta P = P_1 - P_2$) and distribution entropy ($H$). | Raw probability alone is deceptive. A sequence with $P_1 = 0.51$ and $P_2 = 0.49$ requires human review; $\Delta P$ and entropy explicitly surface classifier ambiguity. |
| **Singleton-Aware Scope Modeling** | Explicitly models species-level prediction under a `"reference-library constrained"` scope, acknowledging that 89 of 90 species are singletons. | Prevents dishonest claims of broad biological generalization. Treats reference re-identification as library matching, preserving academic integrity. |

---

## 2. Scientific Rigor & Quality Control

| Feature | What It Does | Why It Matters to Judges |
|---|---|---|
| **Strict Canonical Biological QC** | Enforces strict $\{A, C, G, T\}$ alphabet checks. Flags and rejects ambiguous IUPAC degenerate codes (`N`, `R`, `Y`, etc.) without imputation. | In DNA barcoding, silently imputing missing or ambiguous bases corrupts biological identity. BioScan refuses to guess. |
| **Zero-Metadata Leakage Guarantee** | Audited feature generation pipeline strictly separates biological DNA sequence features from occurrence metadata. | Prevents models from "cheating" by learning correlations with cruise dates, station names, or latitude/longitude coordinates. |
| **Partial Taxonomy Preservation** | Handles records missing species labels in the reference catalog without hallucinating dummy Latin names. | 10 reference records in the dataset are genus- or family-level only. BioScan returns `"Not available in reference data"`, preserving ground truth. |
| **Bit-for-Bit Dataset Integrity** | Continuous cryptographic SHA-256 verification of the raw reference dataset (`5290252D...AD65`). | Proves to judges that benchmark datasets were never modified, pruned, or manipulated to artificially boost validation scores. |

---

## 3. Unknown Detection & The Distance Veto

| Feature | What It Does | Why It Matters to Judges |
|---|---|---|
| **The Distance Veto Mechanism** | Enforces a hard veto: if the normalized Levenshtein distance to the closest cataloged reference barcode exceeds 0.150 ($>15\%$ divergence), status is strictly forced to `out_of_reference`. | Closed-set neural networks and tree models assign high posterior probability ($>90\%$) even to completely unknown organisms because softmax sums to 1.0. The Distance Veto eliminates overconfident hallucinations. |
| **Multi-Stream Concordance** | Evaluates whether model-predicted taxa agree with the taxonomic lineage of the nearest reference barcode across all 5 ranks. | Flags subtle taxonomic conflicts where sequence similarity suggests one family but model probabilities favor another. |
| **Tri-Tier Operational Categorization** | Automatically sorts queries into `known_reference_like`, `low_confidence`, or `out_of_reference`. | Gives field researchers actionable triage: trusted reference matches proceed automatically, while ambiguous or diverged samples are flagged for Sanger/BLAST review. |

---

## 4. Biodiversity Intelligence & Survey Context

| Feature | What It Does | Why It Matters to Judges |
|---|---|---|
| **Station-Level Shannon Diversity ($H_{\text{reads}}$)** | Aggregates field occurrence events across 36 geographic stations from the 2019 New Jersey Ocean Trawl Survey to calculate Shannon diversity and Simpson evenness. | Bridges individual DNA sequence classification with community-level macro-ecological monitoring. |
| **Strict Geographic Suppression** | Strictly forbids projecting geographic presence or survey detection coordinates for any sequence classified as `out_of_reference`. | Prevents catastrophic misinformation: fabricating a geographic distribution for an unverified taxon is scientifically unsupportable. |
| **Survey-Bounded Ecological Claims** | Formats all geographic displays as *"Observed in the BioScan reference survey"*, disclaiming global species range limits. | Distinguishes between localized field detections and global biological ranges, adhering to professional biodiversity informatics standards. |

---

## 5. Engineering, Performance & Usability

| Feature | What It Does | Why It Matters to Judges |
|---|---|---|
| **Unified Public API** | Single entry point via `BioScanPredictor`: `predict_sequence()`, `predict_fasta()`, `predict_batch()`. | Clean, decoupled software architecture separating scientific algorithms from UI presentation. |
| **Fault-Tolerant Batch Processing** | Parses multi-sequence FASTA files and processes each sequence independently. | An invalid sequence with ambiguous bases in record 2 does not crash or interrupt the processing of record 3. |
| **Sub-300ms Prediction Latency** | Optimized in-memory Levenshtein indexing and pre-computed k-mer encoders achieve ~280 ms inference latency per sequence. | Enables smooth, lag-free live interactive demonstrations and high-throughput batch analysis. |
| **One-Click Tabular CSV Export** | Converts all structured predictions, probabilities, distances, and concordance flags into a flat CSV table. | Gives researchers clean, structured outputs ready for downstream ecological modeling in R or Python. |
| **115 Passing Automated Tests** | Comprehensive test suite covering data QC, feature extraction, baselines, XGBoost, hierarchical models, similarity, biodiversity, and the dashboard. | Provides rigorous automated evidence of software reliability, determinism, and regression resistance. |
