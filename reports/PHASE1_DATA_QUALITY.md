# BioScan — Phase 1 Data Quality & Cleaning Report

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Pipeline**: `src/preprocessing/clean_and_build_dataset.py`  
**Phase**: Phase 1 — Data Cleaning & ML Dataset Construction  
**Status**: Verified & Tested (`tests/test_phase1_data.py` passing)

---

## 1. Input Dataset Dimensions & Raw Integrity

- **Source File**: `data/raw/hackathon_dna_taxonomy_merged.csv`
- **Input Dimensions**: **13,736 rows** × **59 columns**
- **File Checksum (SHA256)**:
  `5290252D9496F35A6AC750F9B535842AEDFE9D843C30D0C497136571A115AD65`
- **Raw File Status**: **100% Untouched & Unmodified**. Verified via pre- and post-processing cryptographic checksums.

---

## 2. Output Reference Dataset Dimensions

Two clean, deduplicated artifacts have been generated in `data/processed/`:

1. **`data/processed/reference_barcodes.csv`**:
   - **Dimensions**: **101 rows** × **20 columns**
   - **Granularity**: Exactly 1 representative record per unique DNA barcode sequence.
   - **Primary Key**: `sequence_id` (`SEQ_001` through `SEQ_101`).
   - **Contents**: Raw sequence, normalized sequence, length, base counts (A, T, G, C), GC content, taxonomic annotations, IUCN status, and raw occurrence count.

2. **`data/processed/taxonomy_reference.csv`**:
   - **Dimensions**: **101 rows** × **7 columns**
   - **Granularity**: Exactly 1 representative record per unique DNA barcode sequence.
   - **Columns**: `sequence_id`, `sequence_normalized`, `class`, `order`, `family`, `genus`, `species`.
   - **Purpose**: Ground truth taxonomy lookup supporting hierarchical classification in Phase 2+.

---

## 3. Unique Sequences & Multiplicity Structure

- **Total Unique Sequences**: **101**
- **Multiplicity in Raw Dataset**: Every unique sequence appeared **exactly 136 times** in the raw CSV:
  $$101 \text{ unique barcodes} \times 68 \text{ sampling events} \times 2 \text{ PCR forward primer variants} = 13,736 \text{ rows}$$
- **Deduplication Method**: Grouping by strictly validated, normalized DNA sequence (`sequence_normalized`), verifying taxonomic consensus across all 136 occurrences, and preserving a unified reference record.

---

## 4. Sequence Validation & Quality Audit

The raw sequence column (`dnasequence`) was audited for formatting, encoding, and biochemical integrity:

| Validation Check | Finding | Status | Action Taken |
|---|---|---|---|
| Null / Missing DNA sequences | 0 | PASSED | None needed |
| Lowercase characters | 0 | PASSED | Enforced uppercase `.str.upper()` |
| Leading / Trailing whitespace | 0 | PASSED | Enforced `.str.strip()` |
| Internal whitespace characters | 0 | PASSED | Validated clean |
| Alignment gaps / InDels (`-`, `.`) | 0 | PASSED | Validated clean |
| Ambiguous IUPAC bases (`N`, `R`, `Y`, `S`, etc.) | 0 | PASSED | 100% canonical bases |
| Strict canonical alphabet validation (`^[ACGT]+$`) | 13,736 / 13,736 | PASSED | Zero non-ACGT characters |

---

## 5. Sequence Length Statistics

All 101 reference sequences fall within the expected target amplicon size of the MiFish 12S V5 hypervariable region:

| Metric | Reference Barcodes (n=101) |
|---|---|
| **Minimum Length** | 87 bp |
| **Maximum Length** | 112 bp |
| **Mean Length** | 105.74 bp |
| **Median Length** | 106.00 bp |
| **Mode Length** | 106.00 bp (87 sequences, 86.1%) |
| **Standard Deviation** | 3.16 bp |
| **Interquartile Range (IQR)** | 106.00 – 106.00 bp |

---

## 6. Nucleotide Composition & GC Content Statistics

Calculated directly on the 101 reference sequences:

| Base / Metric | Total Base Count | Mean Count per Barcode | Composition (%) |
|---|---|---|---|
| **Adenine (A)** | 3,262 | 32.30 ± 1.83 | 30.54% |
| **Cytosine (C)** | 3,041 | 30.11 ± 2.05 | 28.47% |
| **Thymine (T)** | 2,451 | 24.27 ± 1.70 | 22.95% |
| **Guanine (G)** | 1,926 | 19.07 ± 1.76 | 18.03% |
| **Total Bases** | 10,680 | 105.74 ± 3.16 | 100.00% |
| **GC Content** | — | — | **46.52% ± 2.61%** (Range: 38.3% – 52.4%) |

---

## 7. Taxonomy Completeness & Hierarchy Breakdown

Taxonomic ranks across the 101 reference sequences:

| Taxonomic Rank | Populated Sequences | Missing Sequences | Completeness (%) | Distinct Classes / Taxa |
|---|---|---|---|---|
| **Kingdom** | 101 | 0 | 100.0% | 1 (*Animalia*) |
| **Phylum** | 101 | 0 | 100.0% | 1 (*Chordata*) |
| **Class** | 101 | 0 | 100.0% | 4 (*Teleostei*: 82, *Elasmobranchii*: 17, *Petromyzonti*: 1, *Chondrostei*: 1) |
| **Order** | 101 | 0 | 100.0% | 30 orders |
| **Family** | 101 | 0 | 100.0% | 66 families |
| **Genus** | 97 | 4 | 96.04% | 87 genera |
| **Species** | 91 | 10 | 90.10% | 90 unique species |

---

## 8. Species-Level Label Availability & Unresolved Sequences

Of the 101 reference barcodes:
- **91 barcodes** have a verified species-level annotation:
  - 89 species have exactly 1 barcode.
  - 1 species (*Ophidion marginatum*, Striped cusk-eel) has 2 distinct barcode variants (`SEQ_060` and `SEQ_061`, length 106 bp, 1 nucleotide difference).
- **10 barcodes** lack species-level annotations in the reference database:
  - **6 barcodes** are resolved to **Genus**:
    - `SEQ_024`: Genus *Hemiramphus* (Family Hemiramphidae)
    - `SEQ_043`: Genus *Ictalurus* (Family Ictaluridae)
    - `SEQ_078`: Genus *Leucoraja* (Family Rajidae, Class Elasmobranchii)
    - `SEQ_079`: Genus *Psenopsis* (Family Centrolophidae)
    - `SEQ_094`: Genus *Thunnus* (Family Scombridae)
    - `SEQ_099`: Genus *Urophycis* (Family Phycidae)
  - **4 barcodes** are resolved only to **Family**:
    - `SEQ_015`: Family *Clupeidae*
    - `SEQ_072`: Family *Pleuronectidae*
    - `SEQ_085`: Family *Sciaenidae*
    - `SEQ_088`: Family *Scombridae*

> [!IMPORTANT]
> These 10 unresolved sequences are **fully preserved** in `reference_barcodes.csv` and `taxonomy_reference.csv`. They are biologically valid DNA sequences from field eDNA samples where the short 12S V5 marker cannot distinguish between congeneric or confamilial sister taxa. In future phases, they will serve as benchmarks for genus/family-level classification and out-of-distribution evaluation.

---

## 9. Duplicate & Collision Analysis

- **Exact Duplicate Rows in Raw Data**: 0.
- **Sequence Multiplicity**: Every sequence occurred 136 times in the raw table due to matrix combination of station and primer.
- **Cross-Taxon Sequence Collisions**: **0**. No single DNA sequence maps to conflicting taxonomic assignments.
- **Deduplicated Output Table**: Contains exactly 0 duplicate sequences (101 unique sequences in 101 rows).

---

## 10. Sequence-to-Taxonomy Consistency Verification

A programmatic consistency assertion was run across all 13,736 rows:
$$\forall s \in \text{Unique Sequences}, \quad |\text{unique}(\text{Taxonomy}(s))| = 1$$
Every sequence yielded a variance of 0 for Kingdom, Phylum, Class, Order, Family, Genus, Species, and Scientific Name.

---

## 11. Feature Separation: Permitted vs. Excluded Attributes

To prevent data leakage, target leakage, and shortcut learning, strict boundaries are established:

### Permitted ML Features (DNA-Derived Only):
- Sequence length (`sequence_length`)
- Canonical base counts (`A_count`, `T_count`, `G_count`, `C_count`)
- Compositional ratios (e.g., `GC_content`, AT/GC ratio)
- *Future Phase 2 features*: k-mer frequency vectors (k = 3, 4, 5), nucleotide transition/transversion frequencies, and entropy.

### Strictly Excluded Columns (Never Passed to Model Inputs):
1. **Taxonomic Labels (Ground Truths)**:
   `class`, `order`, `family`, `genus`, `species`, `scientificName`, `taxonRank`, `taxonomicStatus`.
2. **Specimen & Occurrence Identifiers**:
   `gbifid`, `gbifID`, `occurrenceID`, `eventID`, `sampname`, `datasetKey`, `datasetName`, `license`, `sequence_key`, `sequence_id`.
3. **Spatial & Temporal Environmental Features**:
   `countryCode`, `continent`, `stateProvince`, `locality`, `waterBody`, `decimalLatitude`, `decimalLongitude`, `year`, `month`, `day`, `eventDate`, `depth`.
4. **Experimental & Sequencing Protocol Artifacts**:
   `targetgene`, `targetsubfragment`, `ampliconsize`, `pcrprimernameforward`, `pcrprimernamereverse`, `pcrprimers`, `pcrprimerreference`, `seqmeth`, `concentration`, `concentrationunit`, `contaminationassessment`, `nuclacidext`, `nuclacidamp`, `otudb`, `samplingProtocol`, `sampleSizeValue`, `sampleSizeUnit`, `basisOfRecord`.
5. **Ecological Detection & Abundance Measures**:
   `occurrenceStatus`, `organismQuantity`, `organismQuantityType`, `iucnRedListCategory`.
6. **Precomputed Pipeline Flags**:
   `sequence_length_bp`, `sequence_is_valid_acgt`, `has_species_label`, `has_class_label`.

---

## 12. Anomalies Discovered

1. **Matrix Multiplication Structure**:
   The raw dataset was not 13,736 biological specimens; rather, it was a 68-station × 2-primer survey matrix evaluating 101 reference sequences. Deduplication was mandatory to prevent 100% data leakage.
2. **Two Barcodes for One Species**:
   *Ophidion marginatum* has two distinct 106 bp barcodes differing at a single nucleotide position. Both are valid biological haplotypes and are preserved.
3. **Empty Column**:
   `datasetName` was 100% NaN in the raw CSV.

---

## 13. Assumptions & Methodological Decisions

1. **Target Population**: BioScan is treated as a specialized marine/estuarine fish biodiversity identification platform (MiFish 12S V5 marker) rather than universal planetary biodiversity.
2. **Unresolved Species Preservation**: The 10 sequences without species names are retained with `NaN` in `species` rather than imputing artificial labels, enabling hierarchical modeling at Class, Order, and Family levels.
3. **Deduplication Priority**: Only the deduplicated 101 reference barcodes are used for training DNA sequence prediction models.
