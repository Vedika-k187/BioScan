# Phase 6 — BioScan Unknown Detection, Similarity Search & Evidence Layer Report

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Phase**: Phase 6 — Unknown / Low-Confidence Detection, Reference Similarity Search, and Prediction Evidence Layer  
**Date**: October 2026  
**Status**: Complete  
**Raw Dataset SHA-256**: `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65` (Verified Unchanged)

---

## 1. Project Objective

The primary objective of Phase 6 is to transform the BioScan hierarchical taxonomy models into a **scientifically defensible, evidence-based identification and out-of-reference rejection layer**.

Given an input DNA sequence, the system executes an integrated end-to-end identification workflow:
$$\text{DNA Sequence} \longrightarrow \text{Biological QC} \longrightarrow \text{Feature Extraction} \longrightarrow \text{Hierarchical Classification} \longrightarrow \text{Similarity Search} \longrightarrow \text{Evidence Synthesis} \longrightarrow \text{Tri-Tier Status Assignment}$$

The system categorizes input queries into three distinct, evidence-supported statuses:
1. **`known_reference_like`**: Query has high sequence identity to a cataloged barcode and strong taxonomic concordance.
2. **`low_confidence`**: Query exhibits ambiguous similarity, narrow prediction margin, or taxonomic divergence across ranks.
3. **`out_of_reference`**: Query diverges substantially from all 101 cataloged reference barcodes.

---

## 2. Reuse of Prior Phase Implementations

Phase 6 directly integrates and preserves all prior phase accomplishments without duplication:
- **Phase 1**: Biological QC invariants (strict uppercase `{A, C, G, T}` alphabet, whitespace and newline stripping, amplicon length checks).
- **Phase 2**: Reuses `KMerEncoder` ($k=2, 3$) and `extract_basic_and_complexity_features` from `src/features/dna_features.py`.
- **Phase 3 & 4**: Preserves all model artifacts (`models/knn_species.joblib`, `models/random_forest_species.joblib`, `models/xgboost_species.joblib`).
- **Phase 5**: Reuses the optimal saved model artifacts (`models/taxonomy_*_model.joblib`) and the top-down parent-restricted hierarchical inference engine.

---

## 3. Reference Similarity Search & Distance Definitions

Because 12S rDNA V5 amplicons naturally vary in length across fish taxa (**87 to 112 bp**), similarity searching cannot rely on naive fixed-length comparisons.

Implemented in `src/similarity/reference_search.py`:
1. **Normalized Levenshtein Edit Distance**:
   $$\text{dist}_{\text{edit}}(s_1, s_2) = \frac{\text{Levenshtein}(s_1, s_2)}{\max(\text{len}(s_1), \text{len}(s_2))}$$
   Implemented via 2-row dynamic programming with $O(\min(L_1, L_2))$ space complexity.
2. **Normalized Sequence Identity**:
   $$\text{Similarity}(s_1, s_2) = 1.0 - \text{dist}_{\text{edit}}(s_1, s_2)$$
3. **Hamming Distance**:
   Evaluated strictly when $\text{len}(s_1) == \text{len}(s_2)$; returns `None` when lengths differ.
4. **Ranked Top-k Output**:
   Returns ranked matches with full metadata: `sequence_id`, `distance`, `similarity`, `hamming_distance`, `edit_distance`, `length`, `is_exact_match`, and complete taxonomy (Class, Order, Family, Genus, Species, Scientific Name).

---

## 4. Multi-Stream Confidence Assessment & Rejection Methodology

Rather than relying on an isolated model softmax output or an uncalibrated distance cutoff, Phase 6 implements a multi-stream evidence synthesis engine (`src/prediction/confidence.py`):

### Evidence Streams Evaluated
1. **Model Probability Signals**:
   - Class-level top-1 probability ($P_{\text{top1}}$)
   - Class-level probability margin ($\Delta P = P_1 - P_2$)
   - Shannon entropy of the predicted probability distribution ($H = -\sum p_i \log_2 p_i$)
2. **Reference Similarity Signals**:
   - Normalized distance to nearest reference ($d_{\text{min}}$)
   - Sequence identity to nearest reference ($1 - d_{\text{min}}$)
   - Distance margin to second-nearest reference ($\Delta d = d_2 - d_{\text{min}}$)
   - Exact match indicator ($\text{is\_exact\_match}$)
3. **Taxonomic Concordance Signals**:
   - Agreement between model-predicted taxonomy and nearest reference taxonomy at Class, Order, and Family levels.

---

## 5. Threshold Calibration & Operating Rules

Thresholds were calibrated empirically using Leave-One-Reference-Out (LORO) nearest-neighbor distance distributions across the 101 reference sequences:

| Metric / Parameter | Calibrated Threshold | Empirical Dataset Basis |
|---|---|---|
| **Close Similarity Boundary ($\tau_{\text{sim}}$)** | **$\le 0.050$** ($\ge 95\%$ identity) | Accommodates observed conspecific variation in this dataset (the single conspecific pair *O. marginatum* differs by 4 bp / 98 bp = $0.0408$, ~95.9% identity) and aligns near the 25th percentile ($0.0377$) of nearest-neighbor reference distances. |
| **Out-of-Reference Boundary ($\tau_{\text{out}}$)** | **$> 0.150$** ($< 85\%$ identity) | Exceeds the 75th percentile ($0.1215$) of nearest-neighbor reference distances; captures queries that fall outside the typical reference-library neighborhood. |
| **Model Probability High ($\tau_{\text{prob}}$)** | **$\ge 0.700$** | Provides decisive separation above uniform class distribution baseline. |
| **Probability Margin Low ($\tau_{\text{margin}}$)** | **$< 0.150$** | Flags ambiguous predictions where competitor taxa share comparable posterior support. |

> [!IMPORTANT]
> **Dataset-Specific Operational Thresholds**: These numerical cutoffs ($\tau_{\text{sim}} = 0.050$, $\tau_{\text{out}} = 0.150$) are calibrated specifically for the 87–112 bp 12S rDNA V5 amplicon reference library and its observed pairwise distance distribution. They must NOT be interpreted as universal biological species, congeneric, or confamilial boundaries. Universal COI "barcoding gap" values (~2–3% K2P) do not apply uniformly across different genetic markers or short ribosomal hypervariable loops.

### Status Assignment Logic
1. **`out_of_reference`**:
   - $d_{\text{min}} > 0.150$ and not an exact match.
   - Explanatory message: *"Sequence exhibits substantial divergence from all 101 cataloged 12S rDNA V5 references. This indicates lack of support in this reference library, not proof of a novel biological species."*
2. **`known_reference_like`**:
   - ($d_{\text{min}} \le 0.050$ or exact match) AND predicted Class agrees with nearest reference AND predicted Order agrees with nearest reference.
3. **`low_confidence`**:
   - Assigned when $0.050 < d_{\text{min}} \le 0.150$, OR when model probability margin $\Delta P < 0.150$, OR when model prediction conflicts with nearest reference taxonomy.

---

## 6. Controlled Pseudo-Unknown Simulation Experiments

To evaluate the evidence layer without modifying raw data or leaking labels, a controlled divergence simulation was executed across 7 divergence levels ($N=20$ sampled reference sequences per level, 140 evaluations total):

| Simulated Divergence | Operational Simulation Scenario | Known Reference-Like (%) | Low Confidence (%) | Out-of-Reference (%) |
|---|---|---|---|---|
| **0%** | Exact reference barcode (unmutated) | **100.0%** | 0.0% | 0.0% |
| **2%** | Low simulated divergence (~2 bp substitutions) | **100.0%** | 0.0% | 0.0% |
| **5%** | Moderate simulated divergence (~5 bp substitutions; comparable to library conspecific/congeneric variation) | **80.0%** | **20.0%** | 0.0% |
| **10%** | Substantial simulated divergence (~10 bp substitutions) | **0.0%** | **100.0%** | 0.0% |
| **15%** | Near-boundary divergence (~15 bp substitutions) | **0.0%** | **100.0%** | 0.0% |
| **20%** | Beyond-reference divergence (>15% distance cutoff) | **0.0%** | 0.0% | **100.0%** |
| **30%** | Highly divergent synthetic sequence | **0.0%** | 0.0% | **100.0%** |

### Key Findings & Simulation Caveats
1. **In-Silico Rejection Verification**:
   - In this controlled in-silico simulation, 100% of exact sequences and 100% of 2% mutated sequences retained `known_reference_like` status, and 100% of sequences mutated at $\ge 20\%$ were assigned `out_of_reference`.
   - **Crucial Scope Clarification**: These results represent deterministic verification of the distance and decision logic under uniform in-silico nucleotide substitutions. They must **NOT** be claimed as real-world biological sensitivity/specificity on wild or uncataloged field specimens, which involve sequencing artifacts, biological indels, heteroplasmy, and variable lineage-specific rates.
2. **Graceful Uncertainty Degradation**:
   - Between 5% and 15% divergence, the system automatically transitions into `low_confidence`, avoiding forced false classifications.
3. **Classifier Training Inclusion & Probability Optimism**:
   - In Experiment 2, the ancestor sequences from which pseudo-unknowns were mutated were part of the 101 sequences used to train the saved Phase 5 classifiers.
   - Consequently, **model-predicted probabilities (`model_top1_prob`, `model_prob_margin`) for low-divergence variants (0%, 2%, 5%) are optimistic and not strictly holdout-independent**.
   - **Independence of Rejection**: This does **not** invalidate the out-of-reference rejection claim, because `out_of_reference` status is triggered independently by sequence distance ($d_{\text{min}} > 0.150$). At $\ge 20\%$ divergence, distance to all 101 cataloged references exceeds 0.150 regardless of model probabilities. However, classifier probabilities must not be claimed as independently calibrated on out-of-training taxa.

---

## 7. Leave-One-Reference-Out (LORO) Similarity Retrieval Results

Evaluated across all 101 reference sequences by searching the library strictly excluding the query sequence (`reports/phase6_similarity_results.csv`):

- **Nearest Neighbor Distance Distribution**:
  - Min: **0.0093** (~1 bp diff)
  - 25th Percentile: **0.0377** (~4 bp diff)
  - Median: **0.0660** (~7 bp diff)
  - 75th Percentile: **0.1215** (~12 bp diff)
  - Max: **0.2500** (~25 bp diff)
  - Mean: **0.0785** (Std: 0.0533)
- **Taxonomic Concordance with Nearest Reference**:
  - **Class**: **99 / 101 (98.0%)**
  - **Order**: **71 / 101 (70.3%)**
  - **Family**: **47 / 101 (46.5%)**
  - **Genus**: **16 / 92 (17.4%)**
  - **Species**: **2 / 76 (2.6%)**

---

## 8. Prediction API & Example Outputs

Unified API in `src/prediction/predictor.py`:
```python
from src.prediction.predictor import predict_sequence
result = predict_sequence("ACTATGCTAAGCCTTAAACACAGACAAATGCTCTGTCCGCCAGGGTATTACGAGCATCAGCTTAAAACCCAAAGGACTTGGCGGTACTTTACACCCAC")
```

### Example 1: Known Reference Barcode (`SEQ_054`, *Ophidion marginatum*)
```json
{
  "qc_status": "PASSED",
  "status": "known_reference_like",
  "confidence_tier": "high",
  "predicted_taxonomy": {
    "class": "Teleostei",
    "order": "Ophidiiformes",
    "family": "Ophidiidae",
    "genus": "Ophidion",
    "species": "Ophidion marginatum"
  },
  "evidence": {
    "nearest_distance": 0.0,
    "nearest_similarity": 1.0,
    "is_exact_match": true,
    "model_top1_prob": 1.0,
    "explanation": "Sequence is an exact match (100% identity) to a cataloged reference barcode."
  }
}
```

### Example 2: Out-of-Reference Sequence (Simulated Divergence > 20%)
```json
{
  "qc_status": "PASSED",
  "status": "out_of_reference",
  "confidence_tier": "low",
  "predicted_taxonomy": {
    "class": "Teleostei",
    "order": "Perciformes",
    "family": "Sciaenidae",
    "genus": "Not available in reference data",
    "species": "Not available in reference data"
  },
  "evidence": {
    "nearest_distance": 0.2200,
    "nearest_similarity": 0.7800,
    "is_exact_match": false,
    "explanation": "Sequence exhibits substantial divergence (distance 0.2200, similarity 78.0%) from all 101 cataloged 12S rDNA V5 references. This indicates that the sequence falls outside the coverage of the current reference library."
  }
}
```

### Example 3: Invalid Sequence with IUPAC Ambiguous Base (N)
```json
{
  "qc_status": "FAILED",
  "status": "qc_failed",
  "qc_errors": [
    "Invalid characters detected in sequence: ['N']. Ambiguous IUPAC nucleotide codes (e.g., N, R, Y) are not permitted in reference-grade barcodes."
  ],
  "prediction": null
}
```

---

## 9. Critical Scientific Constraints & Honest Biological Scope

1. **Why "Out-of-Reference" Does NOT Mean "New Species"**:
   - The reference library contains **101 sequences covering 90 species**.
   - A query that diverges from all 101 reference sequences may simply represent one of the tens of thousands of described fish species that are currently absent from this specific hackathon reference library.
   - The system explicitly reports: *"This indicates lack of support in this reference library, not proof of a novel biological species."*
2. **Species-Level Singleton Constraint**:
   - With 89 of 90 species represented by a single sequence, species-level identification operates strictly under instance retrieval.
   - For queries that do not match a reference barcode with high identity, the Species rank is safely designated as `"Not available in reference data"` to prevent false taxonomic hallucination.

---

## 10. Audit & Verification Checklist

- [x] Reference library similarity search implemented (`src/similarity/reference_search.py`) handling unequal sequence lengths.
- [x] Confidence evaluation and out-of-reference rejection engine implemented (`src/prediction/confidence.py`).
- [x] Unified end-to-end predictor implemented (`src/prediction/predictor.py`).
- [x] LORO similarity experiment logged to `reports/phase6_similarity_results.csv`.
- [x] Pseudo-unknown simulation logged to `reports/phase6_unknown_detection_results.csv`.
- [x] 3 publication figures generated in `reports/figures/`.
- [x] Test suite `tests/test_phase6_prediction.py` created (13 unit, integration, and adversarial tests).
- [x] Full regression test suite passed: **62 passed, 0 failed** in 6.2s.
- [x] Raw dataset integrity verified: SHA-256 remains `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65`.
- [x] Prior Phase 3, Phase 4, and Phase 5 model artifacts preserved intact.
- [x] Phase 7 scope boundary strictly respected.
