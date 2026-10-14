# BioScan — Phase 0 Dataset Audit & Data Profile Report

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Dataset**: `data/raw/hackathon_dna_taxonomy_merged.csv`  
**Audit Date**: October 2026  
**Environment**: Python 3.10 (.venv)  

---

## 1. Executive Summary & Biological Origin

An in-depth empirical audit of `data/raw/hackathon_dna_taxonomy_merged.csv` was conducted.

### What the Dataset Actually Is:
- **Origin & BioProject**: The dataset originates from an **environmental DNA (eDNA) metabarcoding biomonitoring survey** of coastal marine fishes in **New Jersey, USA** (Atlantic Ocean), associated with **NCBI BioProject PRJNA638560** and **GBIF Dataset `46488f96-b51e-4c06-9a02-8c4c90`** (published by the New Jersey Bureau of Marine Fisheries, collected between August and November 2019).
- **Genetic Marker**: 100% of the records target the **mitochondrial 12S ribosomal RNA (12S rDNA) V5 hypervariable subfragment**, amplified using the standard **MiFish-U** forward and reverse universal fish primers (`FWD: ACTGGGATTAGATACCCC...`, `REV: AR...`) and sequenced on an **Illumina MiSeq (2 x 150 bp)**.
- **Taxonomic Scope**: The dataset is **strictly marine and estuarine Chordates (Fishes)** within the Kingdom *Animalia* and Phylum *Chordata*. It contains **4 classes**: Ray-finned fishes (*Teleostei*), Sharks/Rays (*Elasmobranchii*), Lampreys (*Petromyzonti*), and Sturgeons (*Chondrostei*). It does **not** contain plants, fungi, bacteria, terrestrial mammals, or universal biodiversity.
- **Data Matrix Structure**: The 13,736 rows represent a structured survey matrix consisting of **101 reference DNA barcode sequences** evaluated across **68 distinct sampling stations/events** with **2 forward primer variants** (`AF` and `AFS`):
  $$\text{Total Rows} = 101 \text{ unique sequences} \times 68 \text{ sampling events} \times 2 \text{ primer runs} = 13,736 \text{ rows}$$
- In 3,208 records, the organism's DNA was confirmed `PRESENT` (read counts > 0, up to 285,963 reads), while in 10,528 records the status was `ABSENT` (0 reads).

---

## 2. Dataset Dimensions & Schema Overview

- **Total Rows**: `13,736`
- **Total Columns**: `59`
- **Data Types**:
  - `object` (string): 40 columns
  - `int64`: 8 columns
  - `float64`: 8 columns
  - `bool`: 3 columns

### Complete Column Inventory

| # | Column Name | Data Type | Null Count | Null % | Unique Values | Column Purpose & Category |
|---|---|---|---|---|---|---|
| 1 | `dnasequence` | `object` | 0 | 0.0% | 101 | **Core Predictor**: 12S V5 DNA barcode sequence |
| 2 | `class` | `object` | 0 | 0.0% | 4 | **Target Label**: Taxonomic Class |
| 3 | `order` | `object` | 0 | 0.0% | 30 | **Target Label**: Taxonomic Order |
| 4 | `family` | `object` | 0 | 0.0% | 66 | **Target Label**: Taxonomic Family |
| 5 | `genus` | `object` | 544 | 3.96% | 87 | **Target Label**: Taxonomic Genus |
| 6 | `species` | `object` | 1360 | 9.90% | 90 | **Target Label**: Taxonomic Species |
| 7 | `scientificName` | `object` | 0 | 0.0% | 100 | Taxonomic authority & full taxon name |
| 8 | `taxonRank` | `object` | 0 | 0.0% | 3 | Rank of resolution: SPECIES, GENUS, FAMILY |
| 9 | `taxonomicStatus` | `object` | 0 | 0.0% | 1 | Status (100% 'ACCEPTED') |
| 10 | `gbifid` | `int64` | 0 | 0.0% | 6,868 | GBIF Record ID (Identifier) |
| 11 | `occurrenceID` | `object` | 0 | 0.0% | 6,868 | Event/occurrence sample identifier |
| 12 | `eventDate` | `object` | 0 | 0.0% | 36 | Collection ISO timestamp |
| 13 | `countryCode` | `object` | 0 | 0.0% | 1 | Country (100% 'US') |
| 14 | `decimalLatitude` | `float64` | 0 | 0.0% | 36 | Sampling station latitude |
| 15 | `decimalLongitude` | `float64` | 0 | 0.0% | 34 | Sampling station longitude |
| 16 | `targetgene` | `object` | 0 | 0.0% | 1 | Genetic marker (100% '12S rDNA') |
| 17 | `targetsubfragment` | `object` | 0 | 0.0% | 1 | Subfragment (100% 'V5') |
| 18 | `ampliconsize` | `int64` | 0 | 0.0% | 1 | Amplicon target size (100% 106 bp) |
| 19 | `sampname` | `object` | 0 | 0.0% | 68 | Biological sampling station code |
| 20 | `pcrprimernameforward` | `object` | 0 | 0.0% | 2 | Forward primer variant ('AF', 'AFS') |
| 21 | `pcrprimers` | `object` | 0 | 0.0% | 1 | Primer sequence string |
| 22 | `pcrprimernamereverse` | `object` | 0 | 0.0% | 1 | Reverse primer name ('AR') |
| 23 | `seqmeth` | `object` | 0 | 0.0% | 1 | Sequencing platform ('Illumina MiSeq 2 x 150') |
| 24 | `concentration` | `float64` | 0 | 0.0% | 63 | DNA extract concentration (ng/ul) |
| 25 | `concentrationunit` | `object` | 0 | 0.0% | 1 | Concentration unit ('ng/ul') |
| 26 | `contaminationassessment`| `bool` | 0 | 0.0% | 1 | Contamination QA flag (100% True) |
| 27 | `nuclacidext` | `object` | 0 | 0.0% | 1 | Extraction protocol DOI |
| 28 | `nuclacidamp` | `object` | 0 | 0.0% | 1 | Amplification protocol DOI |
| 29 | `otudb` | `object` | 0 | 0.0% | 1 | Reference OTU database reference DOI |
| 30 | `pcrprimerreference` | `object` | 0 | 0.0% | 1 | Primer publication citation DOI |
| 31 | `gbifID` | `int64` | 0 | 0.0% | 6,868 | Redundant GBIF ID column |
| 32 | `basisOfRecord` | `object` | 0 | 0.0% | 1 | Record type (100% 'MATERIAL_SAMPLE') |
| 33 | `occurrenceStatus` | `object` | 0 | 0.0% | 2 | eDNA detection: 'ABSENT' (10,528), 'PRESENT' (3,208) |
| 34 | `eventID` | `object` | 0 | 0.0% | 68 | Cruise/station sampling event ID |
| 35 | `year` | `int64` | 0 | 0.0% | 1 | Sampling year (100% 2019) |
| 36 | `month` | `int64` | 0 | 0.0% | 4 | Sampling months (8, 9, 10, 11) |
| 37 | `day` | `int64` | 0 | 0.0% | 15 | Sampling calendar day |
| 38 | `continent` | `object` | 8,686 | 63.24% | 1 | Continent ('NORTH_AMERICA') |
| 39 | `stateProvince` | `object` | 0 | 0.0% | 1 | State (100% 'New Jersey') |
| 40 | `locality` | `object` | 0 | 0.0% | 1 | Survey survey location text |
| 41 | `waterBody` | `object` | 0 | 0.0% | 1 | Body of water (100% 'Atlantic Ocean') |
| 42 | `depth` | `float64` | 0 | 0.0% | 23 | Marine water depth (meters) |
| 43 | `kingdom` | `object` | 0 | 0.0% | 1 | Taxonomic Kingdom (100% 'Animalia') |
| 44 | `phylum` | `object` | 0 | 0.0% | 1 | Taxonomic Phylum (100% 'Chordata') |
| 45 | `associatedSequences` | `object` | 0 | 0.0% | 1 | NCBI BioProject PRJNA638560 accession |
| 46 | `samplingProtocol` | `object` | 0 | 0.0% | 1 | Protocol citation DOI |
| 47 | `sampleSizeValue` | `float64` | 0 | 0.0% | 68 | Station sequencing depth read count |
| 48 | `sampleSizeUnit` | `object` | 0 | 0.0% | 1 | Read count unit ('DNA sequence reads') |
| 49 | `organismQuantity` | `float64` | 0 | 0.0% | 1,360 | Number of eDNA reads detected (0 to 285,963) |
| 50 | `organismQuantityType` | `object` | 0 | 0.0% | 1 | Quantity unit ('DNA sequence reads') |
| 51 | `iucnRedListCategory` | `object` | 1,904 | 13.86% | 5 | IUCN Category (LC, VU, NT, EN, CR) |
| 52 | `license` | `object` | 0 | 0.0% | 1 | License ('CC0_1_0') |
| 53 | `datasetKey` | `object` | 0 | 0.0% | 1 | GBIF dataset UUID |
| 54 | `datasetName` | `float64` | 13,736 | 100.0% | 0 | Empty column (100% Null) |
| 55 | `sequence_length_bp` | `int64` | 0 | 0.0% | 13 | Precomputed length (87 to 112 bp) |
| 56 | `sequence_is_valid_acgt`| `bool` | 0 | 0.0% | 1 | ACGT validity flag (100% True) |
| 57 | `has_species_label` | `bool` | 0 | 0.0% | 2 | Species label present flag |
| 58 | `has_class_label` | `bool` | 0 | 0.0% | 1 | Class label present flag (100% True) |
| 59 | `sequence_key` | `object` | 0 | 0.0% | 101 | Hash / unique key of DNA sequence |

---

## 3. DNA Sequence Quality & Compositional Analysis

- **Target DNA Column**: `dnasequence`
- **Total Sequence Rows**: `13,736`
- **Unique DNA Sequences**: **`101`**
- **Strict ACGT Validity**: **100.0%** (13,736 / 13,736 sequences contain strictly `{A, C, G, T}`).
- **Ambiguous Bases / IUPAC codes** (`N`, `R`, `Y`, `S`, `W`, etc.): **`0`** across all records.
- **Invalid Characters**: None.

### Sequence Length Statistics

| Statistic | Value (Base Pairs) |
|---|---|
| Minimum Length | 87 bp |
| Maximum Length | 112 bp |
| Mean Length | 105.74 bp |
| Median Length | 106.00 bp |
| Mode Length | 106 bp (86.1% of barcodes) |
| 25th Percentile | 106.00 bp |
| 75th Percentile | 106.00 bp |
| Standard Deviation | 3.16 bp |

### Nucleotide Composition

Across all 13,736 rows (and equivalently across the 101 reference sequences):
- **Adenine (A)**: 443,632 bases (30.55%)
- **Cytosine (C)**: 413,576 bases (28.48%)
- **Thymine (T)**: 333,336 bases (22.95%)
- **Guanine (G)**: 261,936 bases (18.03%)
- **GC Content**: **46.51%**

The length profile is tightly centered around 106 bp, matching the expected amplicon length for the MiFish 12S V5 region.

---

## 4. Taxonomic Hierarchy & Distribution

The dataset provides structured hierarchical annotations across 7 ranks:

| Taxonomic Level | Unique Values | Missing Rows | Missing % | Unique Barcodes Covered | Dominant Taxa |
|---|---|---|---|---|---|
| **Kingdom** | 1 | 0 | 0.0% | 101 (100%) | *Animalia* (100%) |
| **Phylum** | 1 | 0 | 0.0% | 101 (100%) | *Chordata* (100%) |
| **Class** | 4 | 0 | 0.0% | 101 (100%) | *Teleostei* (82 barcodes), *Elasmobranchii* (17 barcodes), *Petromyzonti* (1 barcode), *Chondrostei* (1 barcode) |
| **Order** | 30 | 0 | 0.0% | 101 (100%) | *Perciformes* (20 barcodes), *Scombriformes* (10), *Clupeiformes* (9), *Pleuronectiformes* (8) |
| **Family** | 66 | 0 | 0.0% | 101 (100%) | *Scombridae* (6 barcodes), *Sciaenidae* (6), *Carangidae* (6), *Engraulidae* (4) |
| **Genus** | 87 | 544 | 3.96% | 97 (96.0%) | *Anchoa* (3 barcodes), *Prionotus* (2), *Scomber* (2), *Ophidion* (2) |
| **Species** | 90 | 1,360 | 9.90% | 90 (89.1%) | 89 species with 1 barcode; 1 species (*Ophidion marginatum*) with 2 barcodes |

### Breakdown of the 10 Unresolved Species Sequences:
In 10 of the 101 barcodes (representing 1,360 rows), the species field is `NaN`:
- **6 barcodes** are resolved to **Genus** rank (e.g. *Thunnus*, *Urophycis*, *Ictalurus*, *Hemiramphus*, *Psenopsis*, *Leucoraja*).
- **4 barcodes** are resolved only to **Family** rank (*Sciaenidae*, *Clupeidae*, *Pleuronectidae*, *Scombridae*).
All 10 barcodes have complete annotations down to Family, Order, and Class.

---

## 5. Duplicate, Multiplicity & Identity Analysis

A rigorous distinction between different types of duplicate structures reveals:

1. **Exact Duplicate Rows (All 59 columns)**: **`0` rows**. Every row in the CSV is uniquely distinguishable by its combination of sampling event, primer name, and sequence read count.
2. **Identical DNA Sequences**:
   - There are only **101 unique DNA sequences** across the entire dataset.
   - Every single unique DNA sequence appears **exactly 136 times** (mean = 136.0, std = 0.0).
   - Multiplicity explanation: 68 field sampling events × 2 PCR forward primer runs (`AF` vs `AFS`) = 136 rows per reference barcode.
3. **Cross-Species Sequence Collisions (Barcode Sharing)**:
   - **`0` sequence collisions**.
   - No single DNA sequence is associated with more than one species or genus. The mapping from sequence to taxonomic lineage is 100% consistent and deterministic.
4. **Intraspecific Genetic Variation (Haplotypes)**:
   - Exactly **1 species** (*Ophidion marginatum* — Striped cusk-eel) is represented by **2 distinct sequences** (haplotypes) in the library.
   - All other 89 species are represented by exactly **1 reference barcode sequence**.
5. **Specimen & Occurrence Identifiers**:
   - `occurrenceID`: 6,868 unique values (each appears exactly twice: once with primer `AF`, once with `AFS`).
   - `gbifid`: 6,868 unique values (1-to-1 match with `occurrenceID`).
   - `sampname` / `eventID`: 68 unique values (corresponding to the 68 field sampling stations).

---

## 6. Missing Data Analysis

Out of 59 columns, **54 columns have 0 missing values (100% complete)**.

Only 5 columns contain missing values:

| Column | Missing Count | Missing Percentage | Reason / Impact |
|---|---|---|---|
| `datasetName` | 13,736 | **100.0%** | Column header exists without values; safely drop. |
| `continent` | 8,686 | **63.24%** | Omitted in some GBIF records; all coordinates are coastal NJ, USA. |
| `iucnRedListCategory` | 1,904 | **13.86%** | Corresponds to 14 reference taxa unassessed by IUCN Red List. |
| `species` | 1,360 | **9.90%** | Corresponds to 10 barcodes resolved only to Genus or Family rank. |
| `genus` | 544 | **3.96%** | Corresponds to 4 barcodes resolved only to Family rank. |

---

## 7. Class Imbalance & Representation Profile

### Taxonomic Class Imbalance (Unique Barcodes):
- **Teleostei** (Ray-finned fishes): 82 barcodes (**81.19%**)
- **Elasmobranchii** (Sharks, skates, rays): 17 barcodes (**16.83%**)
- **Petromyzonti** (Sea lamprey): 1 barcode (**0.99%**)
- **Chondrostei** (Atlantic sturgeon): 1 barcode (**0.99%**)

### Species-Level Representation:
- Number of named species: **90**
- Species with 2 distinct barcodes: **1** (*Ophidion marginatum*)
- Species with 1 barcode: **89**
- Sequences without species name: **10** (6 resolved to genus, 4 to family)

> [!NOTE]
> In terms of the raw table (13,736 rows), each species appears either 136 times (89 species) or 272 times (1 species). There are zero traditional "singletons" in row count, but at the biological DNA sequence level, almost every species has only a single reference barcode sequence in this library.

---

## 8. Critical Data Leakage Audit & Train/Test Splitting Strategy

### The Data Leakage Threat:
If a machine learning practitioner applies a standard random row split:
```python
# CATASTROPHIC LEAKAGE — DO NOT DO THIS:
train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=42)
```
- Because each sequence is duplicated across 136 rows, **every single test sequence will already exist in the training set multiple times!**
- A model evaluated this way will report artificial ~100% accuracy, but it will have learned nothing about generalizable DNA patterns; it will merely memorize identical strings.

### Identifiers Leakage:
- `gbifid`, `occurrenceID`, `eventID`, `sampname`, `latitude`, `longitude`, `date`, and `depth` reflect the geographical distribution of sampling stations.
- Using any identifier or environmental feature will cause severe data leakage and shortcut learning.

### Recommended Grouping / Splitting Strategies:
1. **Deduplicated Reference Barcode Split**:
   - For DNA sequence classification, the core training set should be formed by deduplicating down to the **101 unique DNA sequences**.
2. **Stratified Splitting by Taxonomic Hierarchy**:
   - Class-level and Order-level evaluation should be stratified across the unique barcodes so that test folds assess true sequence generalization.
3. **Out-of-Distribution / Zero-Shot Taxon Generalization**:
   - For evaluating the model's ability to classify unseen species to their correct Genus, Family, or Order, a **taxonomic group holdout split** (e.g. hold out 20% of genera or families) will be the most rigorous benchmark.

---

## 9. Columns to Exclude from ML Features

The following 58 columns MUST be strictly excluded from model feature inputs:

```
[EXCLUDE FROM FEATURES]:
- Taxonomic identifiers & ground truths:
  'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species',
  'scientificName', 'taxonRank', 'taxonomicStatus'
  
- Sample, Station & GBIF identifiers:
  'gbifid', 'gbifID', 'occurrenceID', 'eventID', 'sampname', 'datasetKey',
  'datasetName', 'license', 'sequence_key', 'associatedSequences'
  
- Spatial & Temporal metadata:
  'countryCode', 'continent', 'stateProvince', 'locality', 'waterBody',
  'decimalLatitude', 'decimalLongitude', 'year', 'month', 'day', 'eventDate', 'depth'
  
- Experimental / PCR / Sequencing conditions:
  'targetgene', 'targetsubfragment', 'ampliconsize', 'pcrprimernameforward',
  'pcrprimernamereverse', 'pcrprimers', 'pcrprimerreference', 'seqmeth',
  'concentration', 'concentrationunit', 'contaminationassessment',
  'nuclacidext', 'nuclacidamp', 'otudb', 'basisOfRecord', 'samplingProtocol',
  'sampleSizeValue', 'sampleSizeUnit'
  
- Biological occurrence / Abundance measurements:
  'occurrenceStatus', 'organismQuantity', 'organismQuantityType', 'iucnRedListCategory'
  
- Precomputed flags:
  'sequence_length_bp', 'sequence_is_valid_acgt', 'has_species_label', 'has_class_label'
```

**Permitted Feature Source**:
- ONLY information mathematically derived directly from `dnasequence` (k-mer frequencies, nucleotide composition, GC content, transition/transversion profiles, sequence length).

---

## 10. Recommendations for Phase 1 ML Strategy

1. **Target Ranks**:
   - Multi-tier hierarchical prediction:
     - Primary tier 1: **Class** (4 classes: Teleostei, Elasmobranchii, Petromyzonti, Chondrostei)
     - Primary tier 2: **Order** (30 orders)
     - Primary tier 3: **Family** (66 families)
     - Primary tier 4: **Species & Genus** (Retrieval / nearest neighbor similarity)
2. **Model Selection**:
   - Because the unique reference set is 101 barcodes of 87-112 bp, high-cardinality classification (90 species) from only 101 training samples is best served by a **hybrid architecture**:
     - **Tree-based Classifier (XGBoost / Random Forest)** for Class, Order, and Family classification using k-mer and compositional features.
     - **Exact / Approximate Nearest Neighbor (KNN / K-mer Distance / Sequence Similarity)** for Species-level retrieval, confidence scoring, and candidate ranking.
3. **Reference Database Framing**:
   - The system must explicitly present itself as an **eDNA Marine Fish Identification & Monitoring System (12S V5 reference library)** and not claim universal biodiversity coverage.

---

## 11. Visual Artifacts Generated

The following visualization artifacts have been generated and saved to `reports/figures/`:
1. `reports/figures/sequence_length_distribution.png`: Reference DNA barcode sequence length distribution (87-112 bp, peak at 106 bp).
2. `reports/figures/class_distribution.png`: Breakdown of taxonomic classes across reference barcodes.
3. `reports/figures/top_orders_distribution.png`: Top 10 taxonomic orders represented in the library.
4. `reports/figures/iucn_status_distribution.png`: IUCN conservation status of the reference species.
5. `reports/figures/missing_values_summary.png`: Summary of missing percentages across columns.
