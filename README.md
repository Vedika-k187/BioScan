# BioScan 🧬🌊

## AI-Powered DNA Biodiversity Identification & Monitoring

BioScan is an AI-powered environmental DNA (eDNA) platform for **DNA barcode identification, taxonomic classification, unknown-sequence detection, and biodiversity monitoring**.

It combines machine learning, sequence similarity, hierarchical taxonomy, confidence analysis, and ecological data into an interactive platform designed for transparent, evidence-aware biodiversity analysis.

---

## 🚀 What BioScan Does

Given a DNA sequence or FASTA file, BioScan can:

- 🧬 Validate and quality-check DNA sequences
- 🔎 Identify DNA against a reference barcode library
- 🌳 Predict taxonomy from **Class → Order → Family → Genus → Species**
- 🧠 Combine machine-learning predictions with sequence-level evidence
- 📏 Find the closest reference sequences using normalized Levenshtein distance
- 🚨 Detect sequences that fall outside the reference library
- 📊 Provide interpretable confidence and similarity evidence
- 🗺️ Connect DNA detections with marine biodiversity survey records
- 📈 Analyze station-level biodiversity and diversity metrics
- 📁 Process multiple sequences through FASTA batch analysis

---

## 💡 The Key Innovation: Unknown Detection

Traditional closed-set classifiers are forced to select one of their known classes, even when an input sequence is not represented in the training library.

BioScan addresses this using an independent **Distance Veto**.

After the machine-learning prediction, BioScan compares the input sequence against its reference barcodes using normalized Levenshtein distance.

If the closest reference sequence is sufficiently divergent:

```text
Minimum normalized distance > 15%
              ↓
     OUT OF REFERENCE
              ↓
No forced species prediction
