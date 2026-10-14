# BioScan Dataset Card

**Dataset Name**: BioScan Curated 12S rDNA V5 eDNA Reference Library & Occurrence Records  
**Raw File**: `data/raw/hackathon_dna_taxonomy_merged.csv`  
**Raw Dataset SHA-256**: `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65`  
**Processed Reference Barcodes**: `data/processed/reference_barcodes.csv` (101 barcodes)  
**Processed Survey Records**: `data/processed/biodiversity_reference_records.csv` (1,604 deduplicated positive occurrences)  
**Survey Station Metrics**: `data/processed/diversity_metrics.csv` (36 sampling stations)  

---

## 1. Dataset Overview & Provenance

The BioScan dataset is derived from open scientific environmental DNA (eDNA) survey data collected during the **2019 New Jersey Bureau of Marine Fisheries Ocean Trawl Survey (NJ OTS)** and deposited in NCBI BioProject **PRJNA638560**.

### Critical Methodological Distinction: 13,736 Rows vs. 101 Barcodes
> **IMPORTANT DATASET STRUCTURE FACT**:  
> The raw CSV file contains **13,736 rows**. However, **these rows DO NOT represent 13,736 independent biological DNA sequences**.  
> In environmental DNA monitoring, water samples are collected across multiple geographic stations, two depths (surface and bottom), two primer replicates (AF and AFS), and seasonal survey cruises. Across all 13,736 raw rows, there are exactly **101 unique DNA barcode sequences**.

Treating 13,736 rows as independent samples would cause massive data leakage, artificially inflating validation accuracy. BioScan rigorously separated the reference sequence catalog from field occurrence events during Phase 1.

---

## 2. Barcode Marker & Sequence Properties

- **Locus**: Mitochondrial 12S ribosomal RNA (12S rDNA V5 hypervariable loop).
- **Target Primers**: Teleost- and elasmobranch-specific MiFish-style primers (AF / AFS replicates).
- **Sequence Length**: Spans **87 to 112 bp** (median: 104 bp; mean: 104.2 bp).
- **Nucleotide Quality**: Canonical bases only $\{A, C, G, T\}$. Zero non-nucleotide characters or ambiguity codes exist in the validated reference library.
- **GC Content**: Range: 35.8% to 56.7% (mean: 46.9%).

---

## 3. Taxonomic Structure & The Singleton Distribution

The 101 reference barcodes represent a highly singleton-dominated marine teleost and elasmobranch assemblage:

| Taxonomic Rank | Validated Count | Dominant Lineage / Characteristic | Coverage |
|---|---|---|---|
| **Kingdom** | 1 | *Animalia* | 100% |
| **Phylum** | 1 | *Chordata* | 100% |
| **Class** | 4 | *Teleostei* (97 seqs), *Elasmobranchii* (2 seqs), *Chondrostei* (1 seq), *Petromyzonti* (1 seq) | 100% |
| **Order** | 30 | *Perciformes* (28 seqs), *Pleuronectiformes* (10 seqs), *Clupeiformes* (9 seqs) | 100% |
| **Family** | 66 | *Gobiidae* (5 seqs), *Sciaenidae* (5 seqs), *Paralichthyidae* (4 seqs) | 100% |
| **Genus** | 87 | 73 singleton genera, 8 multi-species genera (*Anchoa*, etc.) | 100% |
| **Species (Named)** | 90 | **89 singletons** (1 seq/species), 1 doubleton (*Ophidion marginatum*, 2 seqs) | 89.1% |
| **Species (Unlabeled)** | 10 | Retain valid Genus/Family/Order/Class identities; species unspecified | 9.9% |

### Scientific Consequence of Singletons:
Because 89 of 90 cataloged species have only one reference sequence, traditional k-fold cross-validation at the species level is mathematically unviable for training generalization models. The machine learning pipeline treats higher ranks (Class, Order, Family) as hierarchical classification targets while treating species-level prediction as reference-library constrained matching.

---

## 4. Geographic & Survey Metadata

- **Survey Locality**: Continental Shelf offshore New Jersey, Northwest Atlantic Ocean.
- **Latitude Range**: 38.658°N to 40.468°N.
- **Longitude Range**: -74.969°W to -73.862°W.
- **Water Depth**: 1.0 m to 26.8 m (mean: 8.5 m).
- **Observational Grid**: 68 individual water sample events across **36 distinct geographic sampling stations**.
- **Field Occurrence Records**: **1,604 deduplicated positive detection events** across 98 detected reference barcodes.

---

## 5. Potential Leakage Concerns & Safeguards

During Phase 0 and Phase 2, a comprehensive data audit established strict isolation rules:
1. **Metadata Isolation**: Specimen IDs, occurrence IDs, coordinates, dates, and read counts are strictly excluded from DNA feature extraction.
2. **Deterministic Sequence Identity**: The reference database deduplicates identical sequences to prevent train/test cross-contamination.
3. **No Geographic Projection for Novel Sequences**: Out-of-reference queries are strictly barred from inheriting survey coordinates.

---

## 6. Dataset Limitations & Caveats

1. **Marine Fish Bias**: The reference dataset is exclusively marine teleost and elasmobranch fish from the Mid-Atlantic Bight. It contains no freshwater, terrestrial, avian, plant, or invertebrate barcodes.
2. **Survey-Bounded Range**: Coordinate records represent detections within the 2019 New Jersey Ocean Trawl Survey. They do **not** represent global biological range limits or native habitat boundaries.
3. **Amplification vs. Abundance**: `organismQuantity` represents eDNA read counts from high-throughput sequencing. This is an operational amplification signal and **must not be interpreted as absolute organism abundance or biomass**.
