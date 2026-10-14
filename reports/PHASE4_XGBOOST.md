# Phase 4 — BioScan XGBoost Species Classification Report

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Phase**: Phase 4 — Gradient Boosted Trees (XGBoost) Species-Level Classification  
**Date**: October 2026  
**Status**: Complete  
**XGBoost Version**: `3.2.0`  
**Raw Dataset SHA-256**: `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65` (Verified Unchanged)

---

## 1. Project Objective

The primary objective of Phase 4 is to build, systematically evaluate, and interpret an **XGBoost (eXtreme Gradient Boosting)** multiclass classifier for DNA barcode species identification. 

Following the completion of Phase 0 (audit), Phase 1 (deduplication & reference library curation), Phase 2 (k-mer feature engineering), and Phase 3 (KNN and Random Forest baselines), Phase 4 explores whether gradient-boosted decision trees using regularized tree ensembles can effectively map DNA barcode k-mer representations to taxonomic species labels on this reference library.

---

## 2. Dataset Structure & Extreme Class Sparsity

As established in Phase 1 and maintained throughout all subsequent phases:
- **Total Reference Barcodes**: 101 unique non-redundant DNA sequences (12S rDNA V5 marker, 87–112 bp).
- **Labeled Reference Sequences**: **91 sequences** with validated species-level annotations.
- **Unlabeled Sequences**: **10 sequences** lacking species-level annotations (preserved for later unknown-detection/hierarchical taxonomy phases; zero data discarded).
- **Taxonomic Species Count**: **90 distinct species** across the 91 sequences.
- **Class Distribution Breakdown**:
  - **89 species** are represented by **exactly 1 sequence** (singleton classes).
  - **1 species** (*Ophidion marginatum*, Striped cusk-eel; barcode IDs `SEQ_054` and `SEQ_055`) is represented by **2 distinct sequences**.
- **Class Sparsity**: 97.8% of species classes in this dataset are singletons.

This extreme class sparsity constitutes an essential structural property of this reference dataset and requires careful separation between reference-library re-identification and true holdout generalization.

---

## 3. Feature Representations Tested

All 9 scientifically validated DNA feature representations engineered in Phase 2 were evaluated without reduction:

| # | Feature Representation | Dimensions ($D$) | Composition | Feature File |
|---|------------------------|------------------|-------------|--------------|
| 1 | **Basic** | 6 | Length, GC content, Dinucleotide GC, Purine/Pyrimidine ratio, AT skew, GC skew | `features_basic.csv` |
| 2 | **k=2** | 16 | All 16 normalized dinucleotide frequencies | `features_k2.csv` |
| 3 | **k=3** | 64 | All 64 normalized trinucleotide frequencies | `features_k3.csv` |
| 4 | **k=4** | 256 | All 256 normalized 4-mer frequencies | `features_k4.csv` |
| 5 | **k=5** | 1,024 | All 1,024 normalized 5-mer frequencies | `features_k5.csv` |
| 6 | **Basic + k=2** | 22 | Basic (6) + Dinucleotide (16) | `features_basic_k2.csv` |
| 7 | **Basic + k=3** | 70 | Basic (6) + Trinucleotide (64) | `features_basic_k3.csv` |
| 8 | **Basic + k=4** | 262 | Basic (6) + 4-mer (256) | `features_basic_k4.csv` |
| 9 | **Basic + k=5** | 1,030 | Basic (6) + 5-mer (1,024) | `features_basic_k5.csv` |

All feature matrices contain normalized frequency and composition metrics computed strictly per sequence independently without cross-sequence aggregation or target leakage.

---

## 4. XGBoost Configurations Tested

For each of the 9 feature representations, an exhaustive grid search was performed over **8 hyperparameter configurations** (72 total experimental runs):

- **Number of Estimators (`n_estimators`)**: `100`, `200`
- **Maximum Tree Depth (`max_depth`)**: `3`, `5`
- **Learning Rate (`learning_rate`)**: `0.05`, `0.1`
- **Subsample Ratio (`subsample`)**: `0.8` (stochastic row bagging)
- **Column Subsample by Tree (`colsample_bytree`)**: `0.8` (feature bagging)
- **Multiclass Objective**: `multi:softprob`
- **Class Count (`num_class`)**: `90` (for reference fitting) / `K_train` (fold-specific for LOOCV)
- **Evaluation Metric**: `mlogloss`
- **Random Seed**: `42`

---

## 5. Evaluation Methodology & Metric Definitions

To prevent misleading claims, evaluation is split into two conceptually and mathematically distinct categories:

### A. Reference-Library Re-Identification (Indexing)
- **Protocol**: The model is trained on all 91 reference sequences across all 90 classes, and evaluated on its ability to re-identify and rank each reference sequence against the library.
- **Metrics**: Top-1, Top-3, and Top-5 accuracy.
- **Scientific Meaning**: This metric evaluates the capacity of the model to serve as a **reference library retrieval index**. It measures memorization and separation within the catalog. **It does NOT evaluate out-of-sample generalization to new specimens.**

### B. Held-Out LOOCV Generalization
- **Protocol**: For every sequence $i \in \{1, \dots, 91\}$, the sequence is held out ($X_{\text{test}} = X[[i]]$), and the model is trained on all remaining sequences ($X_{\text{train}} = X_{\setminus i}$).
- **The Singleton-Class Limitation**:
  - In 89 of the 91 folds, the held-out sample belongs to a singleton species.
  - When a singleton sample is held out, its class is completely absent from the training partition ($K_{\text{train}} = 89$).
  - A closed-set multiclass classifier trained with `multi:softprob` outputs probabilities strictly over the classes present in its training fold.
  - Because the true species is not in the support of the predicted distribution, $\mathbb{P}(\hat{y} = y_{\text{true}}) = 0$.
  - Therefore, for all 89 singleton species, true closed-set holdout accuracy is mathematically constrained to **0.0%**.
- **Evaluable Holdout Subset (*Ophidion marginatum*)**:
  - *Ophidion marginatum* has 2 distinct reference sequences (`SEQ_054` and `SEQ_055`).
  - When `SEQ_054` is held out, `SEQ_055` remains in the training partition (and vice versa). All 90 classes are present in training, making closed-set generalization validly evaluable on these 2 samples.
- **Dataset-Specific Insight**:
  > *For this highly singleton-dominated reference dataset, the experimental setup behaves more like a reference-library retrieval problem than a conventional closed-set multiclass classification problem.*

---

## 6. Complete Experimental Results (Summary across Feature Sets)

All 72 configurations completed deterministically and are logged in `reports/phase4_xgboost_results.csv`. The table below presents the peak performance achieved for each of the 9 feature representations:

| Feature Representation | D | Best Hyperparameters (`n_est`, `depth`, `lr`) | Ref Top-1 | Ref Top-3 | Ref Top-5 | Eval Holdout Acc (*O. marginatum*) | Mean Margin | Mean Entropy |
|---|---|---|---|---|---|---|---|---|
| **Basic** | 6 | `n_est=200, depth=3, lr=0.1` | 0.0330 | 0.0440 | 0.0659 | 0.0000 | 0.0163 | 4.394 |
| **k=2** | 16 | `n_est=200, depth=3, lr=0.1` | **0.0549** | **0.0659** | 0.0769 | 0.0000 | 0.0186 | 4.392 |
| **k=3** | 64 | `n_est=200, depth=3, lr=0.1` | 0.0220 | 0.0330 | 0.0549 | 0.0000 | 0.0185 | 4.392 |
| **k=4** | 256 | `n_est=200, depth=3, lr=0.1` | 0.0440 | 0.0549 | 0.0659 | 0.0000 | 0.0176 | 4.393 |
| **k=5** | 1024 | `n_est=100, depth=3, lr=0.05` | 0.0330 | 0.0330 | 0.0440 | 0.0000 | 0.0142 | 4.398 |
| **Basic + k=2** | 22 | `n_est=200, depth=3, lr=0.1` | 0.0330 | 0.0549 | 0.0659 | 0.0000 | 0.0185 | 4.392 |
| **Basic + k=3** | 70 | `n_est=200, depth=3, lr=0.1` | 0.0440 | 0.0549 | 0.0659 | 0.0000 | 0.0187 | 4.392 |
| **Basic + k=4** | 262 | `n_est=200, depth=3, lr=0.1` | 0.0330 | 0.0440 | 0.0549 | 0.0000 | 0.0176 | 4.393 |
| **Basic + k=5** | 1030 | `n_est=200, depth=3, lr=0.1` | 0.0440 | 0.0440 | 0.0440 | 0.0000 | 0.0173 | 4.394 |

*Baseline Context: Uniform random guessing over 90 classes yields Top-1 $\approx 1/90 = 0.0111$ (1.11%). XGBoost achieves 5.49%, ~5x above random guess, but substantially trails instance-based indexing.*

---

## 7. Comparison with Phase 3 Baselines (KNN & Random Forest)

Comparing the best configuration from Phase 4 with the actual saved results from Phase 3:

| Model Family | Feature Representation | Best Configuration | Reference Top-1 | Reference Top-3 | Reference Top-5 | Evaluable Holdout Acc (*O. marginatum*) |
|---|---|---|---|---|---|---|
| **K-Nearest Neighbors (KNN)** | `Basic` / `k=2` / `k=3` | $k=1$, metric=`manhattan` / `euclidean` | **1.0000** | **1.0000** | **1.0000** | 0.0000 |
| **Random Forest (RF)** | `Basic` / `k=2` / `k=3` | `n_est=200`, `max_depth=None` | **1.0000** | **1.0000** | **1.0000** | 0.0000 |
| **XGBoost (XGB)** | `k=2` | `n_est=200`, `max_depth=3`, `lr=0.1` | **0.0549** | **0.0659** | **0.0769** | 0.0000 |

### Critical Scientific Distinctions
1. **Reference-Library Indexing vs. Generalization**:
   - The Top-1 = 100% result for KNN ($k=1$) and Random Forest reflects their ability to **re-identify sequences already in the reference catalog**.
   - **It must NOT be presented as evidence of real-world generalization to unknown specimens.** In a 1-NN or unpruned decision tree, querying an existing library barcode naturally retrieves itself.
   - On true held-out LOOCV generalization (*Ophidion marginatum*), all models (KNN, RF, and XGBoost) yielded 0.0% holdout accuracy on this amplicon representation.
2. **Analysis of Observed XGBoost Performance**:
   - The observed underperformance of XGBoost on reference re-identification is consistent with the characteristics of multiclass gradient boosting under extreme one-vs-rest imbalance:
     - With `multi:softprob` over 90 classes, each boosting step fits 90 separate trees. For each tree, exactly 1 positive sample competes against 90 negative samples (a 1:89 class imbalance).
     - Learning rate shrinkage ($\eta = 0.05, 0.1$) and shallow tree depth constraints (`max_depth=3, 5`) apply heavy regularization, which prevents the booster from isolating individual singleton sample vectors in feature space.
     - Consequently, the predicted probabilities remain close to uniform ($~0.011$ per class), resulting in compressed logit margins.

---

## 8. Predicted Probability & Confidence Analysis

Because Phase 5 will introduce open-set / unknown detection, Phase 4 analyzed the output probability distribution of the reference-fitted XGBoost model:

- **Uniform Baseline**: For 90 classes, a flat uniform distribution assigns probability $p_i = \frac{1}{90} \approx 0.0111$ per class (Entropy $H_0 = \ln(90) \approx 4.4998$ nats).
- **Observed Mean Max Probability ($\max_c p_c$)**: **0.0298** ($\sim 2.98\%$).
- **Observed Mean Top-1 vs. Top-2 Probability Margin**: **0.0186** ($\sim 1.86\%$).
- **Observed Mean Shannon Entropy**: **4.392** nats.

### Empirical Findings
1. **Compressed Softmax Probabilities**: Because the tree ensembles cannot sufficiently overcome the 1:89 negative gradient pressure under shallow depth and shrinkage, output probabilities are compressed into a narrow range ($0.010$ to $0.035$).
2. **Narrow Margin**: The average separation between the top-ranked class and the runner-up is only 1.86 percentage points.
3. **Implications for Phase 5**: The observed distribution suggests that raw XGBoost softmax probabilities from shallow boosted trees may not provide a sufficiently sharp threshold for out-of-distribution unknown detection on this dataset, highlighting the utility of distance-based confidence metrics.

---

## 9. Feature Importance Analysis

Feature importance was extracted directly from the booster using **Gain** (the average improvement in loss brought by a feature across all splits where it is used).

### Ranked Dinucleotide Features by Gain (`k=2`)
| Rank | Dinucleotide Feature | Importance Gain |
|---|---|---|
| 1 | `GG` | **1.1833** |
| 2 | `AG` | **0.6494** |
| 3 | `GT` | **0.6458** |
| 4 | `AC` | **0.5757** |
| 5 | `CC` | **0.5721** |
| 6 | `TC` | **0.5090** |
| 7 | `CA` | **0.4859** |
| 8 | `CG` | **0.3815** |
| 9 | `GA` | **0.2832** |
| 10 | `AA` | **0.2611** |
| 11 | `TT` | **0.2561** |
| 12 | `CT` | **0.2472** |
| 13 | `TG` | **0.2327** |
| 14 | `GC` | **0.2061** |
| 15 | `TA` | **0.1790** |
| 16 | `AT` | **0.1649** |

*Artifact visual*: `reports/figures/xgboost_feature_importance.png`  
*Artifact table*: `reports/xgboost_feature_importance.csv`

### Scientifically Cautious Interpretation
> **Notice on Causal Attribution**:  
> These rankings reflect **model-derived importance** in the evaluated dataset. We do **not** claim that dinucleotides `GG` or `AG` biologically cause or uniquely determine species identity. Rather, variations in guanine-rich dinucleotide frequencies provided the highest empirical reduction in training log-loss within this specific tree ensemble on these 91 sequences.

---

## 10. Model Selection Criterion & Saved Artifact

### Selection Criterion
XGBoost is the specific model family investigated in Phase 4. Across the 72 evaluated XGBoost configurations, the final model was selected using the following explicit criteria:
1. **Highest Reference-Library Re-Identification Performance**: Highest Ref Top-1 accuracy (0.0549) and Ref Top-3 accuracy (0.0659).
2. **Discriminative Margin**: Highest separation between Top-1 and Top-2 probabilities (0.0186).
3. **Feature Parsimony**: Selecting the most compact feature representation achieving top performance (`k=2` with 16 features, rather than larger combinations).

*Note: This designates the best configuration within the XGBoost family under these criteria; it does not claim that XGBoost is the overall best model compared to Phase 3 baselines.*

### Final Saved Model Artifact
- **Artifact Path**: `models/xgboost_species.joblib`
- **Model**: `XGBClassifier`
- **Feature Set**: `k=2` (16 dinucleotide features)
- **Hyperparameters**:
  - `n_estimators`: 200
  - `max_depth`: 3
  - `learning_rate`: 0.1
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
  - `objective`: `multi:softprob`
  - `num_class`: 90
  - `random_state`: 42
- **Saved Metadata**:
  - `feature_columns`: List of 16 dinucleotide names
  - `label_encoder`: Fitted `LabelEncoder` instance mapping 90 species
  - `target_classes`: All 90 species strings
  - `metrics`: Dictionary of evaluation metrics
  - `random_state`: 42
  - `xgboost_version`: `3.2.0`

---

## 11. Scientific Limitations & Boundaries

1. **Extreme Class Sparsity**: With 89 of 90 species represented by a single sequence, supervised holdout generalization cannot be evaluated for the vast majority of species in a closed-set formulation.
2. **Retrieval vs. Generalization**: High reference-library re-identification in non-parametric models (e.g., KNN Top-1 = 100%) demonstrates indexing and lookup capability, but must not be confused with generalizability to unseen biological specimens.
3. **Dataset-Specific Behavior**: For this highly singleton-dominated reference dataset, the experimental setup behaves more like a reference-library retrieval problem than a conventional closed-set multiclass classification problem.
4. **Scope Adherence**: In strict adherence to instructions, Phase 4 concludes here. No hierarchical taxonomy, unknown detection, or application deployment has been implemented.

---

## 12. Audit & Verification Checklist

- [x] All 72 XGBoost experiments executed, validated, and logged to `reports/phase4_xgboost_results.csv`.
- [x] Best XGBoost model saved to `models/xgboost_species.joblib` with complete metadata.
- [x] Feature importance CSV and plot saved to `reports/xgboost_feature_importance.csv` and `reports/figures/xgboost_feature_importance.png`.
- [x] All test suites passing (`pytest -q` -> 39 passed).
- [x] Raw dataset integrity verified: SHA-256 remains `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65`.
- [x] Strict Phase 4 scope boundary preserved.
