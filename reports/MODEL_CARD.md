# BioScan Model Card

**Model Family**: BioScan Hierarchical Taxonomy & Reference Evidence Engine  
**Release Version**: v1.0.0 (Phase 10 Release)  
**Task**: Hierarchical Taxonomic Inference, Reference Barcode Similarity Search, and Out-of-Reference Unknown Detection  
**Primary Domain**: Teleost and Elasmobranch Fishes  
**Biological Marker**: Mitochondrial 12S ribosomal RNA (12S rDNA V5 hypervariable loop, 87–112 bp)  

---

## 1. Model Details & Purpose

BioScan is a multi-stage machine learning and sequence retrieval system developed to classify vertebrate environmental DNA (eDNA) barcodes. Rather than treating species identification as an unconstrained closed-set classification problem, BioScan evaluates biological taxonomy across a 5-level hierarchy (Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species), calculates pairwise edit distances to cataloged reference sequences, and enforces a hard **Distance Veto** to detect out-of-reference organisms.

- **Developer**: BioScan Applied AI Team
- **Model Type**: Ensemble of parent-restricted hierarchical classifiers paired with normalized Levenshtein sequence search.
- **Underlying Algorithms**:
  - Class-level model: Multi-class Classifier on canonical compositional and k-mer frequency features.
  - Order, Family, Genus, Species models: Parent-restricted classifiers trained conditionally within higher taxonomic clades.
  - Sequence Retrieval: Normalized Levenshtein distance:
    $$d_{\text{norm}}(s_1, s_2) = \frac{\text{Levenshtein}(s_1, s_2)}{\max(|s_1|, |s_2|)}$$

---

## 2. Input Specifications & Feature Representation

- **Input Modality**: Raw DNA sequence string or standard multi-record FASTA file.
- **Allowed Alphabet**: Strict canonical nucleotides $\{A, C, G, T\}$. Ambiguous IUPAC codes (`N`, `R`, `Y`, `S`, etc.) are rejected without silent imputation.
- **Sequence Length**: 87–112 bp typical window (warnings emitted for sequences outside this baseline).
- **Feature Space**:
  - Basic compositional metrics: GC content, AT skew, GC skew, purine/pyrimidine ratio, sequence length.
  - Multi-scale k-mer representations: Canonical k-mer relative frequency distributions ($k = 2, 3, 4, 5$).
- **Strict Leakage Prevention**: Metadata (occurrence IDs, sampling dates, latitude/longitude coordinates, station names, collection water depth) are **strictly forbidden** from entering the feature space and are used exclusively downstream for biodiversity intelligence.

---

## 3. Training & Reference Data Summary

The model family is trained on the curated BioScan reference library constructed in Phase 1:
- **Total Unique Reference Barcodes**: 101 unique 12S rDNA V5 sequences.
- **Named Species**: 90 named species across 87 genera, 66 families, 30 orders, and 4 classes (*Teleostei*, *Elasmobranchii*, *Chondrostei*, *Petromyzonti*).
- **Singleton Distribution**: **89 of 90 named species classes have singleton reference sequences** (1 sequence per species in the catalog).
- **Unlabeled Taxa**: 10 reference sequences retain valid Genus, Family, Order, or Class labels but lack species assignment in the source catalog.

---

## 4. Unknown Detection & Confidence Framework

Standard closed-set multiclass classifiers suffer from severe overconfidence on uncataloged sequences because softmax posterior probabilities must sum to 1.0. BioScan counters this failure mode via multi-stream evidence and the **Distance Veto**:

1. **Top-1 Posterior Probability ($P_1$)**: Maximum class probability from the taxonomic model.
2. **Probability Margin ($\Delta P = P_1 - P_2$)**: Difference between highest and second-highest class probabilities.
3. **Distribution Entropy ($H$)**: Shannon entropy quantifying classification uncertainty.
4. **Nearest Sequence Edit Distance ($d_{\text{min}}$)**: Normalized Levenshtein edit distance to the closest cataloged reference barcode.
5. **Taxonomic Concordance**: Agreement between model-predicted taxa and the top reference barcode across Class, Order, Family, Genus, and Species ranks.

### Operational Status Rules:
- **`KNOWN_REFERENCE_LIKE`**: $d_{\text{min}} \le 0.050$ (or exact match) AND Class and Order concordant with top reference barcode.
- **`LOW_CONFIDENCE`**: $d_{\text{min}} \le 0.150$ with probability margin $\Delta P < 0.150$ or taxonomic discordance.
- **`OUT_OF_REFERENCE`**: $d_{\text{min}} > 0.150$ (and not an exact match).

> **THE DISTANCE VETO**:  
> If $d_{\text{min}} > 0.150$, the sequence is strictly assigned `out_of_reference` regardless of classifier probability. High posterior probability is forbidden from overriding sequence divergence.

---

## 5. Performance & Scientific Transparency

- **Reference Re-Identification**: High accuracy on exact reference re-identification across cataloged sequences.
- **Critical Methodological Distinction**: Because 89 of 90 species are singletons in the reference catalog, **reference re-identification performance MUST NOT be equated with real-world biological generalization**.
- **Scope Annotation**: All species-level predictions are explicitly labeled:  
  `"species_prediction_scope": "reference-library constrained"`.

---

## 6. Intended Use & Applications

### Intended For:
- Preliminary taxonomic triage of 12S rDNA V5 fish eDNA barcodes.
- Rapid detection of reference-matched marine organisms against curated regional catalogs.
- Highlighting candidate taxonomic lineage (Class, Order, Family) when species-level certainty is unsupported.
- Automated screening for anomalous, diverged, or uncataloged sequences requiring manual Sanger sequencing or BLAST verification.
- Educational, hackathon, and research benchmarking demonstration.

### NOT Intended For:
- **Clinical or Regulatory Diagnostics**: Not certified for food safety, commercial seafood authentication, or legal enforcement.
- **Unconstrained Global Biodiversity Classification**: Not trained on terrestrial mammals, birds, plants, fungi, or invertebrates.
- **Universal Species Discovery**: An `out_of_reference` classification indicates sequence divergence from the cataloged reference library; it does **not** prove the discovery of a new biological species.
- **Alternative Marker Loci**: Cannot process COI, 16S, 18S, or ITS loci without retraining and reference catalog reconstruction.
