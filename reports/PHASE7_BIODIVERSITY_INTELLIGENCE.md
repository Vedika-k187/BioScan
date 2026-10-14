# Phase 7 — BioScan Biodiversity Intelligence, Geographic Analysis & Analytics Layer Report

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Phase**: Phase 7 — Biodiversity Intelligence, Spatial Reference Analysis, and Diversity Layer  
**Date**: October 2026  
**Status**: Complete & Verified  
**Raw Dataset SHA-256**: `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65` (Verified Unchanged)

---

## 1. Executive Summary & Objective

Phase 7 transforms the approved Phase 6 prediction and evidence pipeline into an evidence-based **Biodiversity Intelligence Layer**.

Given an input DNA barcode, the system executes an integrated end-to-end scientific workflow:
$$\text{DNA Sequence} \longrightarrow \text{QC} \longrightarrow \text{Hierarchical Classification} \longrightarrow \text{Similarity Search} \longrightarrow \text{Confidence Status} \longrightarrow \text{Reference Occurrence Metadata} \longrightarrow \text{Geographic & Diversity Intelligence}$$

### Fundamental Scientific Boundaries
1. **Reference Records vs Global Fauna**: The geographic analysis describes the **distribution of available reference records in this specific dataset** (offshore New Jersey, USA). It does **NOT** represent complete global biological species distributions or population sizes.
2. **Unit of Analysis**: The dataset contains **101 unique DNA reference sequences** assayed across **13,736 raw screening rows** (representing 68 water samples and 36 trawl stations with 2 forward primer replicates).
3. **Out-of-Reference Rejection**: Sequences flagged as `out_of_reference` indicate lack of support in the 101-barcode library and are **never** projected as novel species or given speculative geographic ranges.

---

## 2. Processed Biodiversity Data Products

All data products were deterministically generated and saved to `data/processed/`:

| Data Product | File Path | Record Count | Unit of Analysis |
|---|---|---|---|
| **Occurrence Records** | [`biodiversity_reference_records.csv`](../data/processed/biodiversity_reference_records.csv) | **1,604 rows** | Deduplicated positive field detection events |
| **Country Summary** | [`country_biodiversity_summary.csv`](../data/processed/country_biodiversity_summary.csv) | **1 row** | National political & oceanographic jurisdiction |
| **Taxonomic Hierarchy** | [`taxonomic_summary.csv`](../data/processed/taxonomic_summary.csv) | **8 rows** | Taxonomic ranks from Kingdom to Species |
| **Species Distribution** | [`species_distribution_summary.csv`](../data/processed/species_distribution_summary.csv) | **101 rows** | Unique reference DNA barcodes |
| **Station Diversity** | [`diversity_metrics.csv`](../data/processed/diversity_metrics.csv) | **36 rows** | Distinct geographic sampling stations |
| **Geographic QC Audit** | [`geographic_quality_summary.csv`](geographic_quality_summary.csv) | **14 rows** | Coordinate validation and quality metrics |

---

## 3. Reference-Record Biodiversity Metrics

### A. Taxonomic Richness
- **Unique DNA Barcodes**: **101** (87–112 bp, 12S rDNA V5)
- **Named Species**: **90 species** across 91 labeled sequences (89 singletons, 1 doubleton: *Ophidion marginatum*)
- **Species-Unlabeled Sequences**: **10 sequences** (retain valid Genus/Family/Order/Class)
- **Genera**: **87 genera** (85 singleton genera in library, 2 double-species genera)
- **Families**: **66 families**
- **Orders**: **30 orders** (dominated by Perciformes with 28 barcodes)
- **Classes**: **4 classes** (Teleostei: 97, Elasmobranchii: 2, Chondrostei: 1, Petromyzonti: 1)

### B. Geographic Richness (United States / NJ OTS)
- **Survey Region**: Offshore New Jersey Continental Shelf (Northwest Atlantic Ocean)
- **Latitude / Longitude Bounds**: 38.658°N to 40.468°N, -74.969°W to -73.862°W
- **Sampling Stations**: **36 distinct stations**
- **Depth Range**: **1.0 m to 26.84 m** (Mean: **17.84 m**)
- **Temporal Range**: January 11, 2019 to November 10, 2019

### C. Survey Detection vs Library Representation
- **Barcodes Detected in Survey**: **98 / 101 (97.0%)**
- **Barcodes Assayed but Undetected (Negative Screening)**: **3 / 101 (3.0%)**
  - `SEQ_022`: *Membras martinica* (Rough silverside)
  - `SEQ_026`: *Tylosurus acus* (Agujon needlefish)
  - `SEQ_085`: Unlabeled teleost barcode

---

## 4. Shannon Diversity Methodology & Observational Units

### The Exact Computational Pipeline

To prevent artifactual inflation from survey screening designs or technical PCR replicates, BioScan implements a strictly defined 7-step aggregation pipeline:

$$\begin{aligned}
\text{Raw PCR Assay } (N=13,736) 
&\longrightarrow \text{Primer-Replicate Deduplication } (N=3,208 \to 1,604) \\
&\longrightarrow \text{Field Occurrence Detection Event } (1,604 \text{ positive records}) \\
&\longrightarrow \text{Station-Level Aggregation } (36 \text{ geographic coordinate stations}) \\
&\longrightarrow \text{Taxon Abundance Summation } (A_i = \sum \overline{\text{reads}}_{i, s}) \\
&\longrightarrow \text{Relative Abundance Normalization } (p_i = A_i / \sum A_k) \\
&\longrightarrow \text{Shannon Diversity } (H = -\sum p_i \ln p_i)
\end{aligned}$$

#### Step-by-Step Methodology:
1. **Raw Observational Unit**:
   - A single PCR metabarcoding assay screening: one water sample (`sampname`) tested for one target 12S V5 barcode (`dnasequence`) using one forward primer (`pcrprimernameforward`: `AF` or `AFS`).
   - Total raw rows: $68 \text{ water samples} \times 101 \text{ target barcodes} \times 2 \text{ primer replicates} = 13,736 \text{ assays}$.
2. **Primer-Replicate Deduplication**:
   - Filters to positive field detections (`occurrenceStatus == 'PRESENT'`, 3,208 raw rows).
   - Groups by `(occurrenceID, sequence_id)` to collapse the two technical forward primer assays (`AF` and `AFS`).
   - Read counts (`organismQuantity`) of the two primer assays are **averaged** (`mean`).
   - Result: Exactly **1,604 deduplicated field occurrence records** in [`biodiversity_reference_records.csv`](../data/processed/biodiversity_reference_records.csv).
3. **Station Definition & Multi-Sample Aggregation**:
   - A **station** is strictly defined by a unique geographic coordinate pair `(decimalLatitude, decimalLongitude)`. There are exactly **36 distinct sampling stations** along the New Jersey continental shelf.
   - At 32 of the 36 stations, oceanographers collected two distinct depth strata: a Surface sample (`S`, depth = 1.0 m) and a Bottom sample (`B`, seafloor depth, up to 26.84 m). At 4 stations, a single sample was collected.
   - All occurrence records from multiple depth samples at the same station are pooled into that station's community pool.
4. **Taxon Abundance ($A_i$)**:
   - For each taxon $i$ detected at that station, its abundance is calculated by **summing** the mean read counts across all occurrences at that station:
     $$A_i = \sum_{j \in \text{station occurrences of taxon } i} \overline{\text{organismQuantity}}_{i, j}$$
5. **Relative Abundance ($p_i$)**:
   - Normalized relative read abundance:
     $$p_i = \frac{A_i}{\sum_{k=1}^S A_k}, \quad \text{where } p_i > 0 \text{ and } \sum_{i=1}^S p_i = 1.0$$
6. **Shannon Diversity Formulations**:
   - **Abundance-Weighted Shannon ($H_{\text{reads}}$)**:
     $$H_{\text{reads}} = -\sum_{i=1}^S p_i \ln(p_i)$$
   - **Incidence-Based Shannon ($H_{\text{incidence}}$)**:
     $$H_{\text{incidence}} = -\sum_{i=1}^S q_i \ln(q_i), \quad \text{where } q_i = \frac{c_i}{\sum_k c_k}$$
     ($c_i$ is the count of positive water samples detecting taxon $i$ at that station, either 1 or 2).
   - **Simpson Diversity ($1 - D$)**:
     $$1 - D = 1 - \sum_{i=1}^S p_i^2$$
   - **Pielou's Evenness ($J$)**:
     $$J = \frac{H}{\ln(S)}$$
7. **Handling Zero Detections**:
   - Taxa with zero positive detections at a given station have $A_i = 0$. By standard mathematical convention ($\lim_{p \to 0^+} p \ln p = 0$), non-detected taxa contribute zero to the summation and are excluded from richness $S$.
8. **Handling Species-Unlabeled Records**:
   - 211 occurrence records belong to 9 reference barcodes that lack a cataloged species binomen.
   - In default species-level evaluation (`tax_level="species"`), these are mapped to distinct lineage buckets (`{Genus} sp.` or `{Family} sp.`, e.g. *Urophycis sp.*, *Leucoraja sp.*, *Clupeidae sp.*). Because each unlabeled barcode represents a distinct fish lineage, this preserves true detected biological diversity without artificial collapse or loss of data.
   - In strict named-species evaluation (`tax_level="named_species_only"`), unlabeled records are excluded (1,393 occurrences evaluated).
   - Diversity calculations can also be performed at `genus` (85 detected genera) and `family` (65 detected families).
9. **Reference-Library Baseline Caveat**:
   - Shannon diversity across the 101 reference sequences in `reference_barcodes.csv` is uniform ($H = \ln 90 \approx 4.50$) because 89 of 90 species are singletons by catalog design. This reflects reference library curation, NOT natural ecological community diversity.
10. **Observed Station Metrics**:
    - Station-level $H_{\text{reads}}$ ranges from **1.54** to **3.22** (Mean: **2.38**).
    - Station-level Pielou evenness ($J$) ranges from **0.56** to **0.88**.

---

## 5. Prediction-to-Biodiversity Integration API

Unified API in `src/biodiversity/prediction_biodiversity.py`:
```python
from src.biodiversity.prediction_biodiversity import analyze_prediction

result = analyze_prediction("ACTATGCTAAGCCTTAAACACAGACAAATGCTCTGTCCGCCAGGGTATTACGAGCATCAGCTTAAAACCCAAAGGACTTGGCGGTACTTTACACCCAC")
```

### Tri-Tier Evidence Behavior
1. **`known_reference_like`**:
   - Synthesizes classifier prediction with occurrence records from `biodiversity_reference_records.csv`.
   - Returns supporting reference record counts, sampling station coordinates, depth ranges, and collection dates.
   - Disclaims: *"Reflects observed reference records from 2019 NJ OTS survey; does NOT represent complete global species range."*
2. **`low_confidence`**:
   - Supplies candidate taxonomy and nearest reference metadata with prominent uncertainty warnings.
3. **`out_of_reference`**:
   - **Strictly forbids species distribution projection**.
   - Displays nearest reference metadata labeled exclusively as distance context.
   - Disclaims: *"Indicates lack of representation in the 101-barcode reference library; NOT proof of a novel species."*

---

## 6. Visualization Suite

Generated and saved under `reports/figures/phase7/`:
1. `taxonomic_composition.png`: Richness counts across hierarchical ranks.
2. `country_richness.png`: Reference-record representation for US waters.
3. `sampling_distribution.png`: Seasonal month frequencies and depth histogram.
4. `taxonomic_hierarchy.png`: Top 10 orders by barcode representation.
5. `reference_record_abundance.png`: Top 10 species by reference occurrence count (explicitly labeled: *"NOT Population Abundance"*).
6. `geographic_station_map.png`: Spatial map of 36 trawl survey stations showing depth and detected richness.
7. `shannon_diversity_stations.png`: Shannon diversity gradient across depths.
8. `sequence_coverage_summary.png`: Amplicon length distribution and library detection rate (97.0% detected, 3.0% negative screening).

---

## 7. Verification Checklist

- [x] Raw dataset checksum unchanged: `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65`.
- [x] No metadata used as classifier features.
- [x] Reusable modules implemented in `src/biodiversity/`.
- [x] Processed CSV data products generated with valid schemas.
- [x] 8 publication-quality figures generated in `reports/figures/phase7/`.
- [x] Shannon diversity unit of analysis rigorously defined and tested.
- [x] Tri-tier prediction-to-biodiversity integration fully functional.
- [x] All 62 regression tests across Phases 1–6 preserved and passing.
