# BioScan — Phase 2 DNA Feature Engineering Report

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Pipeline Module**: `src/features/dna_features.py`  
**Test Suite**: `tests/test_phase2_features.py` (12 tests, 100% passing)  
**Status**: Verified & Exported  

---

## 1. Objective

The objective of Phase 2 is to construct a reproducible, leakage-free DNA feature-engineering pipeline that translates variable-length biological nucleotide sequences (87–112 bp) into structured numerical matrices. 

The pipeline generates multiple distinct representation families so that downstream machine learning models (KNN, Random Forest, XGBoost) can be systematically benchmarked across different feature representations without data leakage.

---

## 2. Input Dataset & Sample Granularity

- **Source File**: `data/processed/reference_barcodes.csv` (approved Phase 1 artifact)
- **Sample Unit**: Exactly **101 genuinely independent reference DNA sequences**.
- **Alphabet**: 100% canonical `{A, C, G, T}`.
- **Marker Region**: Mitochondrial 12S rDNA V5 hypervariable subfragment.
- **Sequence Length**: 87 to 112 bp (median: 106 bp).

> [!IMPORTANT]
> Because there are only 101 independent biological reference sequences in this catalog, feature space dimensionality ($p$) relative to sample size ($n=101$) is a critical design constraint. The pipeline creates independent, modular feature sets to prevent the curse of dimensionality and overfitting.

---

## 3. Basic & Compositional Feature Definitions

For every sequence, 11 primary sequence composition metrics were extracted directly from the raw string:

1. **`sequence_length`**: Total base pairs ($L \in [87, 112]$).
2. **`A_count`**, **`T_count`**, **`G_count`**, **`C_count`**: Integer occurrence counts of each nucleotide base.
3. **`A_frequency`**, **`T_frequency`**, **`G_frequency`**, **`C_frequency`**: Base proportions relative to total length:
   $$\text{Base Frequency} = \frac{\text{Base Count}}{L}$$
4. **`GC_content`**: Proportion of Guanine and Cytosine bases:
   $$\text{GC\_content} = \frac{\text{G\_count} + \text{C\_count}}{L}$$
5. **`AT_content`**: Proportion of Adenine and Thymine bases:
   $$\text{AT\_content} = \frac{\text{A\_count} + \text{T\_count}}{L}$$

All metrics obey strict mathematical conservation laws:
$$\sum_{b \in \{A, T, G, C\}} \text{Frequency}(b) = 1.0 \quad \text{and} \quad \text{GC\_content} + \text{AT\_content} = 1.0$$

---

## 4. Sequence Complexity & Diversity Features

In addition to basic counts, 6 interpretable complexity metrics were calculated:

1. **`unique_base_count`**: Number of distinct nucleotide types present (strictly 4 for all 101 sequences).
2. **`unique_k2_count`**: Number of distinct dinucleotides observed (ranges from 14 to 16 out of 16 possible).
3. **`unique_k3_count`**: Number of distinct trinucleotides observed (ranges from 46 to 60 out of 64 possible).
4. **`shannon_entropy`**: Information entropy of base composition (in bits):
   $$H = -\sum_{b \in \{A, C, G, T\}, p_b > 0} p_b \log_2(p_b)$$
   Mean: 1.954 bits (out of theoretical maximum $\log_2(4) = 2.000$ bits for uniform distribution).
5. **`max_base_frequency`**: Dominant base proportion per sequence (ranges from 0.283 to 0.387).
6. **`min_base_frequency`**: Least frequent base proportion (ranges from 0.123 to 0.217, consistently Guanine).

These 17 features are unified into [`data/processed/features_basic.csv`](file:///c:/Users/Vedika%20Kurhade/OneDrive/Desktop/BioScan/data/processed/features_basic.csv).

---

## 5. k-mer Frequency Feature Engineering

A reusable `KMerEncoder` was built to count sliding overlapping subwords of length $k$ ($k \in \{2, 3, 4, 5\}$) with a step size of 1 bp:

$$\text{Number of sliding windows} = L - k + 1$$
$$\text{Frequency}(m) = \frac{\text{Count}(m)}{L - k + 1}$$

- **Alphabet**: Lexicographically ordered cartesian products of `['A', 'C', 'G', 'T']`.
- **Normalization**: Relative frequency (sum of all $4^k$ frequencies per sequence equals $1.0$).
- **Handling Short Sequences**: Safely handled via window count bounds ($L \ge 87 \gg k=5$).

---

## 6. Feature Matrix Summary

The following modular feature tables were constructed and exported in `data/processed/`:

| Feature Set | Exported File | Number of Samples ($n$) | Number of Features ($p$) | Dimensionality Ratio ($p / n$) |
|---|---|---|---|---|
| **Basic (Composition + Complexity)** | `features_basic.csv` | 101 | **17** | 0.17 |
| **k = 2 (Dinucleotides)** | `features_k2.csv` | 101 | **16** | 0.16 |
| **k = 3 (Trinucleotides)** | `features_k3.csv` | 101 | **64** | 0.63 |
| **k = 4 (Tetranucleotides)** | `features_k4.csv` | 101 | **256** | 2.53 |
| **k = 5 (Pentanucleotides)** | `features_k5.csv` | 101 | **1024** | 10.14 |
| **Basic + k=2** | `features_basic_k2.csv` | 101 | **33** | 0.33 |
| **Basic + k=3** | `features_basic_k3.csv` | 101 | **81** | 0.80 |
| **Basic + k=4** | `features_basic_k4.csv` | 101 | **273** | 2.70 |
| **Basic + k=5** | `features_basic_k5.csv` | 101 | **1041** | 10.31 |

Every table includes `sequence_id` as a stable key and **strictly zero** taxonomic labels or metadata fields.

---

## 7. Feature Redundancy & Multicollinearity Findings

Pearson correlation analysis across the basic numerical features revealed several structural relationships:

1. **Perfect Collinearity ($r = -1.0$)**:
   - `GC_content` vs. `AT_content`: Exactly collinear ($r = -1.000$). Retaining both in linear models would cause singular covariance matrices, though tree-based models and distance metrics can handle both.
2. **Sum-to-One Multicollinearity**:
   - `A_frequency + T_frequency + G_frequency + C_frequency = 1.0`. The 4-simplex composition has only 3 degrees of freedom.
3. **Count-Frequency Redundancy**:
   - `T_count` vs. `T_frequency`: $r = 0.976$.
   - `G_count` vs. `G_frequency`: $r = 0.974$.
   Because sequence lengths vary moderately (87–112 bp), counts and frequencies are strongly correlated. Frequencies are preferred for length-invariant distance comparison.
4. **Compositional Extremum Correlation**:
   - `G_frequency` vs. `min_base_frequency`: $r = 0.952$. Guanine is almost universally the minimum-frequency nucleotide in vertebrate mitochondrial 12S rDNA V5.

> [!NOTE]
> All features are retained in the exported tables for interpretability. Redundant features will be filtered or regularized during model training in Phase 3.

---

## 8. Scaling Decision & Leakage Prevention

- **Export State**: All numerical feature matrices are exported in their **raw, unscaled state** (counts as integers, frequencies as proportions in $[0, 1]$).
- **Rationale**:
  - Global pre-scaling across the entire dataset before train/test splitting causes **data leakage** (the mean and variance of the test set leak into training).
  - Tree-based architectures (Random Forest, XGBoost) are invariant to monotonic feature scaling.
  - Distance-based models (KNN) require scaling, but scaling parameters must be fitted **strictly on the training folds** of cross-validation.

---

## 9. Recommendations for Phase 3 Model Benchmarking

Rather than guessing which feature set is "best", Phase 3 should systematically evaluate:

1. **Primary Candidate — $k=3$ (64 features) and Basic + $k=3$ (81 features)**:
   - High biological relevance (reading frames, codon-adjacent motifs).
   - Favorable sample-to-feature ratio ($p=64 < n=101$), well-suited for tree models and KNN without extreme sparsity.
2. **Compact Candidate — $k=2$ (16 features) and Basic (17 features)**:
   - Extremely compact, low risk of overfitting, high interpretability.
3. **High-Dimensional Regularization Candidate — $k=4$ (256 features)**:
   - Evaluates whether higher-order nucleotide context adds discriminative value when regularized (e.g. via L1/L2 penalties in XGBoost).
4. **Sparsity Caution — $k=5$ (1024 features)**:
   - $p \gg n$ ($1024 \gg 101$). Over $90\%$ of pentanucleotide bins are zero for any single 106 bp sequence. High risk of overfitting unless aggressive dimensionality reduction or sparse regularization is applied.

The final model and feature set selection will be made based solely on empirical cross-validation metrics.
