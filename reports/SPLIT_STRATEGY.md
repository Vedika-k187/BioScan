# BioScan — Leakage-Safe Data Splitting & Evaluation Strategy

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Document**: `reports/SPLIT_STRATEGY.md`  
**Phase**: Phase 1 Design Specification (Pre-Modeling)  

---

## 1. Executive Summary & Why Naive Splitting Fails

In biological sequence machine learning, naive data splitting is the primary source of inflated, non-reproducible performance metrics. 

In our audited raw dataset (`13,736 rows` across `101 unique sequences`), **every single sequence was replicated 136 times**. 
- A standard random row split (`train_test_split(df, test_size=0.2)`) guarantees that **100% of test sequences are present in the training set**.
- The model merely performs trivial string lookup / exact memorization rather than learning generalizable biological motif representations.

To ensure scientific validity and real-world applicability, all subsequent machine learning in BioScan must strictly adhere to the leakage-safe evaluation protocols detailed below.

---

## 2. Evaluation Scenario A: Reference-Sequence Level Classification

### Purpose:
Evaluate how well the model predicts taxonomic labels (Class, Order, Family, Genus, Species) on unseen biological DNA barcodes within the known reference taxonomic domain.

### Splitting Protocol:
1. **Sample Unit**: The atomic unit of data is the **unique normalized DNA sequence** ($N = 101$ in `data/processed/reference_barcodes.csv`).
2. **Strict Identity Isolation**:
   $$\text{Set}(\text{Train Sequences}) \cap \text{Set}(\text{Test Sequences}) = \emptyset$$
   No sequence in the test set may match any sequence in the training set.
3. **Cross-Validation Scheme**:
   - For higher taxonomic levels (e.g., **Class** with 4 classes, **Order** with 30 orders):
     - **Stratified K-Fold Cross-Validation** ($K = 5$) at the sequence level, stratified on `class` or high-level taxonomic grouping.
     - Ensures each fold maintains balanced representation of dominant classes (*Teleostei*, *Elasmobranchii*) and minority classes (*Petromyzonti*, *Chondrostei*).
4. **Handling Haplotypes**:
   - The two barcodes for *Ophidion marginatum* (`SEQ_060` and `SEQ_061`) must either:
     - Reside together in the same fold (via `GroupKFold` on species name) to prevent intraspecific memorization, OR
     - Intentionally separated to benchmark intraspecific haplotype generalization.

---

## 3. Evaluation Scenario B: Taxonomic Generalization (Grouped Holdout)

### Purpose:
In practical biodiversity monitoring and environmental DNA (eDNA) surveillance, field sequencers frequently encounter novel species not yet cataloged in the local reference barcode library. 
Scenario B tests the critical question:
> *"Can the AI model correctly predict the Family, Order, or Class of an organism whose exact Species or Genus has never been seen in the training data?"*

### Splitting Protocol:
1. **Genus-Level Grouped Holdout (`GroupKFold` / `StratifiedGroupKFold` on `genus`)**:
   - All sequences belonging to a selected Genus are assigned exclusively to either the training set OR the test set.
   - The test set evaluates whether genus-level unobserved species can be correctly assigned to their parent **Family**, **Order**, and **Class**.
2. **Family-Level Grouped Holdout**:
   - Entire families are held out to test Order- and Class-level hierarchical generalizability.
3. **Unlabeled Sequence Evaluation Set**:
   - The **10 reference sequences lacking species labels** (6 resolved to genus, 4 resolved to family) serve as a built-in validation testbed for hierarchical and out-of-distribution prediction:
     - The model should predict their known Class, Order, and Family with high confidence.
     - The model should flag their Species prediction as **low-confidence / unknown**, triggering the system's "Out-of-Distribution / Candidate Review" flag.

---

## 4. Evaluation Metrics for Future Phases

Because biodiversity taxonomy is strictly hierarchical, flat accuracy is insufficient. The future evaluation suite will track:

1. **Hierarchical Accuracy**:
   - Top-1 and Top-3 accuracy at Class, Order, Family, Genus, and Species levels.
2. **Class-Balanced Metrics**:
   - Macro-averaged and Weighted Precision, Recall, and F1-score to account for class imbalance (e.g. *Teleostei* vs. *Chondrostei*).
3. **Confidence Calibration & Unknown Detection**:
   - Expected Calibration Error (ECE).
   - Area Under the Receiver Operating Characteristic (AUROC) for distinguishing in-library species from novel/out-of-distribution barcodes.

---

## 5. Execution Roadmap

| Step | Action | Status |
|---|---|---|
| **Phase 0** | Audit raw dataset, identify 136x replication structure and leakage threat | COMPLETED |
| **Phase 1** | Construct deduplicated 101-barcode reference table & design split protocols | COMPLETED |
| **Phase 2** | Extract DNA-only features (k-mers, GC, transitions) & implement `SequenceGroupKFold` split generators | NEXT STEP |
| **Phase 3** | Train baseline models (KNN, Random Forest, XGBoost) strictly following Scenarios A & B | FUTURE |
