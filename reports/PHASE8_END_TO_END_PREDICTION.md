# Phase 8: End-to-End BioScan Prediction Engine Report

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Phase**: Phase 8 — End-to-End Prediction Engine  
**Status**: COMPLETE  
**Dataset Reference**: 101 unique 12S rDNA barcodes (87–112 bp), 90 cataloged species (89 singletons), 2019 New Jersey Ocean Trawl eDNA Survey metadata.  
**Raw Dataset SHA-256**: `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65` (Verified Unchanged)

---

## 1. Executive Summary & Objective

The objective of Phase 8 is to transform the validated Phase 1–7 research and model pipelines into a unified, production-oriented, end-to-end biological prediction engine. The resulting system accepts either a single DNA sequence string or a multi-record FASTA file/string and executes a deterministic, multi-tiered inference and evidence pipeline:

$$\text{DNA Sequence} \longrightarrow \text{Strict QC} \longrightarrow \text{Feature Extraction} \longrightarrow \text{Hierarchical ML} \longrightarrow \text{Similarity Search} \longrightarrow \text{Confidence Analysis} \longrightarrow \text{Unknown Detection} \longrightarrow \text{Evidence Aggregation} \longrightarrow \text{Biodiversity Context} \longrightarrow \text{Structured Result}$$

> **Scientific Transparency Statement**:  
> Phase 8 integrates previously validated components; it does not constitute a new biological validation study. The engine does not simply return an isolated taxon name. Instead, every output is paired with taxonomic rank resolution, probability margins, normalized edit distances to cataloged references, top-5 nearest matches, and survey detection metadata.

---

## 2. End-to-End System Architecture

The prediction engine is housed in `src/prediction/` and exposed via a single, coherent public interface (`BioScanPredictor`).

```
+-----------------------------------------------------------------------------------------+
|                                BIOSCAN PREDICTION ENGINE                                |
+-----------------------------------------------------------------------------------------+
                                             |
                   [Input: Single Sequence, FASTA, or Batch List]
                                             |
                                             v
                      +---------------------------------------------+
                      |       Input Validation & Biological QC      |
                      |  - Uppercase & whitespace normalization     |
                      |  - Strict {A,C,G,T} validation (reject N/R) |
                      |  - 12S Reference length checks (87-112 bp)  |
                      +---------------------------------------------+
                                             |
                      +----------------------+----------------------+
                      | (Pass)                                      | (Fail)
                      v                                             v
       +-------------------------------+             +-------------------------------+
       |       Feature Extraction      |             |     Structured QC Failure     |
       |  - Phase 2 Canonical Features |             |  - status: "qc_failed"        |
       |  - Basic / K-mer Frequencies  |             |  - Explicit violation errors  |
       |  - Strict zero-leakage check  |             +-------------------------------+
       +-------------------------------+
                      |
                      v
       +-------------------------------+
       |   Hierarchical ML Taxonomy    |
       |  - Parent-restricted models   |
       |  - Class -> Order -> Family   |
       |    -> Genus -> Species        |
       +-------------------------------+
                      |
                      v
       +-------------------------------+
       |    Reference Similarity       |
       |  - Normalized Levenshtein     |
       |  - Top-K Reference Retrieval  |
       |  - Exact match determination  |
       +-------------------------------+
                      |
                      v
       +-------------------------------------------------------------+
       |             Confidence Analysis & Distance Veto             |
       |  - Top-1 probability, Top-1/Top-2 margin, Shannon entropy   |
       |  - Distance Veto: If d_min > 0.150 -> OUT_OF_REFERENCE      |
       |  - Status: KNOWN_REFERENCE_LIKE | LOW_CONFIDENCE |          |
       |            OUT_OF_REFERENCE                                 |
       +-------------------------------------------------------------+
                      |
                      v
       +-------------------------------+
       |      Evidence Aggregation     |
       |  - Concordance across ranks   |
       |  - Multi-stream justification |
       +-------------------------------+
                      |
                      v
       +-------------------------------+
       |      Biodiversity Context     |
       |  - Occurrence lookups         |
       |  - Station detection records  |
       |  - Strictly bounded claims    |
       +-------------------------------+
                      |
                      v
       +-------------------------------------------------------------+
       |                   Structured Prediction Output              |
       |  - Dictionary / JSON format                                 |
       |  - Flat tabular export (Batch CSV)                          |
       +-------------------------------------------------------------+
```

---

## 3. Reused Components from Phases 1–7

Phase 8 adheres strictly to the modular engineering principle: no validated component is reinvented.

| Phase | Validated Component | Production Implementation Reused |
|---|---|---|
| **Phase 1** | Barcode reference database & strict QC rules | `data/processed/reference_barcodes.csv`, `src/data/sequence_qc.py` |
| **Phase 2** | DNA feature representation | `src/features/dna_features.py` (K-mer frequency & composition) |
| **Phase 3 & 4** | Baseline & Tree-based models | Verified feature matrix invariants and model serialization formats |
| **Phase 5** | Hierarchical taxonomy models | `models/taxonomy_*_model.joblib` (Class, Order, Family, Genus, Species) |
| **Phase 6** | Similarity search & Unknown detection | `src/similarity/reference_search.py`, `src/prediction/confidence.py` |
| **Phase 7** | Occurrence & Station Biodiversity records | `data/processed/station_biodiversity.csv`, `data/processed/reference_occurrences.csv` |

---

## 4. Input Contract & Biological Quality Control

### 4.1 Supported Input Modes
1. **Raw Sequence String**: Single DNA string via `predict_sequence(sequence, sequence_id)`.
2. **FASTA String or File**: Single or multi-record FASTA formatted inputs via `predict_fasta(path_or_string)`.
3. **Batch List**: List of sequence dictionaries or strings via `predict_batch(sequences)`.

### 4.2 Biological Sequence Validation Rules
- **Canonical Bases**: Only $\{A, C, G, T\}$ are accepted.
- **Normalization**: Leading/trailing whitespace is stripped, and lower-case characters are safely converted to upper-case without altering biological content.
- **Strict Ambiguity Rejection**: Any input containing degenerate IUPAC nucleotide codes (e.g., `N`, `R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`) or non-alphabetical characters is **rejected** immediately. BioScan never silently imputes ambiguous bases.
- **Length Warnings vs. Hard Filters**:
  - The 12S V5 reference barcode window spans **87–112 bp** (median: 104 bp).
  - Sequences between 50 bp and 300 bp outside the 87–112 bp range receive informative biological QC warnings (e.g., `Sequence length (130 bp) is outside typical 12S V5 barcode window (87-112 bp)`), but are not arbitrarily aborted.
  - Sequences $< 50$ bp or $> 300$ bp are flagged with severe length anomaly warnings.
- **Fail-Safe Contract**: Inputs failing biological QC return a structured `qc_failed` response without raising uncaught exceptions or aborting batch operations.

---

## 5. Hierarchical Taxonomic Inference

Taxonomic prediction is executed hierarchically across 5 ranks: **Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species**, with Kingdom fixed to `Animalia`.

1. **Parent-Restricted Hierarchy**: Lower taxonomic ranks are evaluated conditionally within the taxonomic subtree determined by higher ranks.
2. **Preservation of Partial Reference Taxonomy**: If an organism's reference record lacks family, genus, or species labeling, the engine returns `"Not available in reference data"`. Taxa are never fabricated.
3. **Species Scope Limitation**:
   - Out of 90 cataloged species, 89 are singletons in the reference dataset.
   - The engine explicitly annotates species predictions with `"species_prediction_scope": "reference-library constrained"`.
   - Species predictions are bounded strictly to reference-matching re-identification and are never presented as unconstrained biological generalization.

---

## 6. Similarity Search, Confidence Framework & Unknown Detection

### 6.1 Reference Similarity Search
For every query sequence, pairwise sequence retrieval is performed against the catalog of 101 reference barcodes using normalized Levenshtein distance:

$$d_{\text{norm}}(s_1, s_2) = \frac{\text{Levenshtein}(s_1, s_2)}{\max(|s_1|, |s_2|)}, \quad \text{Similarity} = 1.0 - d_{\text{norm}}$$

Top-1, Top-3, and Top-5 nearest reference sequences are retrieved alongside sequence IDs, taxonomic lineage, edit distance, and exact match flags.

### 6.2 Multi-Stream Confidence Metrics
The engine computes:
- **Top-1 Probability ($P_1$)**: Maximum output probability from the taxonomic classifier.
- **Probability Margin ($\Delta P = P_1 - P_2$)**: Difference between highest and second-highest class probabilities.
- **Shannon Entropy ($H$)**: Distribution entropy measuring classification spread:
  $$H = -\sum_{i} p_i \log_2(p_i + 10^{-12})$$
- **Nearest Reference Distance ($d_{\text{min}}$)**: Normalized edit distance to closest reference barcode.
- **Taxonomic Concordance**: Agreement between model-predicted taxa and the nearest reference barcode at Class, Order, Family, Genus, and Species levels.

### 6.3 Operational Thresholds & The Distance Veto

The system applies three operational status tiers:

| Status Tier | Operational Conditions | Biological Meaning |
|---|---|---|
| `known_reference_like` | $(d_{\text{min}} \le 0.050 \text{ OR exact match}) \text{ AND } \text{Class Concordant} \text{ AND } \text{Order Concordant}$ | Sequence exhibits strong sequence and taxonomic identity to cataloged reference barcodes. |
| `low_confidence` | $d_{\text{min}} \le 0.150 \text{ AND } (\Delta P < 0.150 \text{ OR discordant taxonomy})$ | Sequence is within the broad reference domain, but ambiguity exists between candidate taxa. |
| `out_of_reference` | $d_{\text{min}} > 0.150 \text{ AND NOT exact match}$ | Sequence is significantly divergent from all cataloged reference barcodes in the library. |

> **CRITICAL SCIENTIFIC PRINCIPLE: The Distance Veto**:  
> Closed-set classifiers can assign high probabilities to completely unknown or divergent sequences because their softmax output must sum to 1.0. To prevent false biological certainty, the prediction engine enforces a **hard distance veto**:  
> **If $d_{\text{min}} > 0.150$, the status is strictly forced to `out_of_reference`, regardless of classifier probability.**

---

## 7. Biodiversity Context Integration

For queries classified as `known_reference_like` or `low_confidence`, the engine cross-references the nearest reference record with Phase 7 survey intelligence (`reference_occurrences.csv` and `station_biodiversity.csv`).

### Scientific Safeguards on Geographic Claims
- **Strict Distribution Boundary**: Reference occurrences document where an organism was **observed during the 2019 New Jersey Ocean Trawl Survey**. The engine explicitly disclaims that detection locations represent complete species ranges.
- **Hard Rule for Out-of-Reference Sequences**: For any sequence classified as `out_of_reference`, the engine **strictly suppresses geographic distribution projection** (`projected_geographic_distribution: None`). Fabricating a geographic presence for an uncharacterized sequence is forbidden.

---

## 8. Structured Prediction Schema

Every prediction adheres to a standardized JSON/dict schema:

```json
{
  "input": {
    "sequence_id": "QUERY_001",
    "sequence": "ACTG...ACTG",
    "length": 104
  },
  "qc": {
    "status": "passed",
    "is_valid": true,
    "errors": [],
    "warnings": []
  },
  "prediction": {
    "status": "known_reference_like",
    "confidence_tier": "tier_1_reference_matched",
    "kingdom": "Animalia",
    "class": "Actinopterygii",
    "order": "Gadiformes",
    "family": "Merlucciidae",
    "genus": "Merluccius",
    "species": "Merluccius bilinearis",
    "species_prediction_scope": "reference-library constrained"
  },
  "confidence": {
    "top1_probability": 0.985,
    "probability_margin": 0.970,
    "entropy": 0.082,
    "confidence_assessment": "High probability and substantial margin"
  },
  "similarity": {
    "nearest_reference_id": "REF_001",
    "nearest_distance": 0.000,
    "nearest_similarity": 1.000,
    "is_exact_match": true,
    "top_k_references": [...]
  },
  "evidence": {
    "overall_status": "known_reference_like",
    "distance_veto_applied": false,
    "taxonomic_concordance": {
      "class_concordant": true,
      "order_concordant": true,
      "family_concordant": true,
      "genus_concordant": true,
      "species_concordant": true
    },
    "reasons": [...]
  },
  "biodiversity": {
    "reference_survey_context": {
      "country": "United States",
      "locality": "New Jersey Coast (Ocean Trawl Survey)",
      "total_reference_detections": 12,
      "stations_detected": ["Station_1", "Station_4"]
    },
    "geographic_bounding_box": {
      "min_latitude": 38.89,
      "max_latitude": 40.12,
      "min_longitude": -74.55,
      "max_longitude": -73.80
    },
    "projected_geographic_distribution": "Observed in 2019 New Jersey reference survey records only.",
    "scientific_caveat": "Detection in reference survey does not represent complete global species distribution."
  },
  "disclaimer": "BioScan predictions are derived from a curated reference library..."
}
```

---

## 9. FASTA Parsing & Batch Processing

The batch processing pipeline (`src/prediction/batch_predict.py` and `src/prediction/fasta_parser.py`) provides:
1. **Multi-Record Parsing**: Standard multi-line FASTA files and strings are cleanly parsed, extracting record headers as `sequence_id`.
2. **Individual Sequence Isolation**: Each sequence in a batch is evaluated independently. A malformed sequence or QC failure in record 2 never disrupts the prediction of record 3.
3. **Tabular CSV Export**: Batch prediction results can be exported directly to a flat CSV table containing sequence IDs, lengths, QC status, prediction status, predicted taxonomy, probabilities, nearest references, and distance metrics.

---

## 10. Performance Benchmarks

Inference latency was benchmarked on the reference dataset:
- **Predictor Startup & Artifact Verification**: ~1.2 s (loads 5 taxonomy models, reference barcodes, and biodiversity indices once into memory).
- **Single Sequence In-Memory Prediction**: **~278 ms** (includes QC, k-mer feature extraction, hierarchical inference, normalized Levenshtein across all 101 references, evidence aggregation, and biodiversity lookup).
- **FASTA Batch Processing**: **~280 ms / sequence**.

---

## 11. Verification & Testing

A dedicated test suite (`tests/test_phase8_prediction_engine.py`) comprising 23 comprehensive tests was created and executed alongside the full historical regression suite (Phases 1–7).

### Test Coverage Breakdown

| Test ID | Test Name | Target Behavior Verified | Result |
|---|---|---|---|
| `test_01` | `test_01_valid_dna_prediction` | Valid 104 bp DNA produces complete structured output | **PASSED** |
| `test_02` | `test_02_lowercase_dna_normalization` | Lowercase sequences normalized to uppercase without error | **PASSED** |
| `test_03` | `test_03_invalid_dna_rejection` | Sequences containing `N` or non-bases return `qc_failed` | **PASSED** |
| `test_04` | `test_04_empty_sequence_handling` | Empty or whitespace-only sequences return structured QC failure | **PASSED** |
| `test_05` | `test_05_sequence_length_warning_handling` | Atypical lengths (e.g. 150 bp) emit warnings without crashing | **PASSED** |
| `test_06` | `test_06_exact_reference_sequence` | Known reference barcode yields `known_reference_like`, dist 0.0 | **PASSED** |
| `test_07` | `test_07_known_reference_like_sequence` | 1-bp mutation sequence achieves $d \le 0.05$ and `known_reference_like` | **PASSED** |
| `test_08` | `test_08_low_confidence_case` | Moderately diverged sequence ($d \approx 0.08$) assigns `low_confidence` | **PASSED** |
| `test_09` | `test_09_out_of_reference_case` | Synthetic divergent sequence ($d > 0.150$) assigns `out_of_reference` | **PASSED** |
| `test_10` | `test_10_high_probability_cannot_override_distance_veto` | Hard veto enforced: distance $> 0.150$ overrides high classifier probability | **PASSED** |
| `test_11` | `test_11_top_k_similarity_structure` | Top-5 nearest references sorted by ascending distance | **PASSED** |
| `test_12` | `test_12_taxonomy_fields_present` | All taxonomic levels present (Kingdom, Class, Order, Family, Genus, Species) | **PASSED** |
| `test_13` | `test_13_missing_species_label_handled_correctly` | Unlabeled references return `"Not available in reference data"` | **PASSED** |
| `test_14` | `test_14_biodiversity_context_does_not_fabricate` | Out-of-reference sequence suppresses geographic projection | **PASSED** |
| `test_15` | `test_15_fasta_single_sequence` | FASTA string parsed and predicted accurately | **PASSED** |
| `test_16` | `test_16_fasta_multiple_sequences` | Multi-record FASTA evaluated in correct order | **PASSED** |
| `test_17` | `test_17_malformed_fasta` | Headerless or corrupted FASTA returns structured error | **PASSED** |
| `test_18` | `test_18_one_invalid_sequence_does_not_crash_batch` | Invalid sequence in batch does not halt valid sequence processing | **PASSED** |
| `test_19` | `test_19_batch_csv_export` | Tabular export produces valid CSV with expected columns | **PASSED** |
| `test_20` | `test_20_model_artifact_feature_compatibility` | Artifacts verified: Class, Order, Family, Genus, Species | **PASSED** |
| `test_21` | `test_21_deterministic_repeated_prediction` | Repeated inferences on same sequence yield identical results | **PASSED** |
| `test_22` | `test_22_result_schema_validation` | JSON serializability and schema completeness confirmed | **PASSED** |
| `test_23` | `test_23_raw_dataset_sha256_unmodified` | Raw dataset integrity verified: SHA-256 matches ground truth | **PASSED** |

### Regression Test Suite Summary
- **Phase 8 Tests**: 23 passed, 0 failed
- **All Project Tests**: **102 passed**, 0 failed (13.25 s execution time)

---

## 12. Explicit System Limitations & Boundaries

To preserve strict scientific integrity, BioScan explicitly states its operational constraints:
1. **Target Locus**: The system is trained on 12S mitochondrial rDNA V5 barcodes. It is not suitable for COI, 16S, 18S, or ITS loci without retraining.
2. **Taxonomic Domain**: The reference database consists of 101 barcodes representing teleost and elasmobranch fishes. It cannot classify birds, mammals, plants, or microorganisms.
3. **Singleton Constraint**: 89 of 90 species are singletons in the training set. Species-level identification functions as reference-library matching, not as unconstrained generalization.
4. **Geographic Caveats**: Biodiversity occurrence data is derived from the 2019 New Jersey Ocean Trawl Survey. Detections reflect survey sample points, not species range limits.
5. **Distance Veto**: Sequences with $>15\%$ normalized Levenshtein distance are labeled `out_of_reference` to prevent overconfident classification of uncataloged taxa.
