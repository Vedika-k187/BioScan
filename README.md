# 🧬 BioScan
### AI-Powered DNA Biodiversity Identification & Monitoring System

BioScan is an AI-powered eDNA analysis system that combines hierarchical machine learning, DNA similarity search, and unknown-sequence detection for biodiversity identification and monitoring.

---

## 1. Problem

Environmental DNA (eDNA) enables biodiversity detection from genetic material collected from environmental samples.

However, conventional DNA classifiers have a major limitation:

- They are **closed-set** and are forced to predict one of the species they were trained on.
- An unfamiliar DNA sequence can therefore receive a **high-confidence but incorrect species prediction**.
- Limited reference sequences make species-level identification difficult.
- A species prediction alone does not show how strongly the DNA is supported by the reference library.

**The key challenge:**  
How can we identify DNA while also determining whether the sequence is actually supported by the available reference data?

---

## 2. Solution

BioScan combines **machine learning + sequence-level evidence**.

Instead of relying only on classifier probability, BioScan:

1. Validates the DNA sequence.
2. Extracts DNA and k-mer features.
3. Predicts taxonomy hierarchically from **Class → Order → Family → Genus → Species**.
4. Searches for the closest reference DNA sequences.
5. Calculates sequence similarity using normalized Levenshtein distance.
6. Applies a **Distance Veto** to detect sequences outside the reference library.
7. Provides the final prediction together with supporting evidence.

This makes BioScan a **reference-aware DNA identification system rather than a simple closed-set classifier**.

---
## 3. System Architecture

![BioScan System Architecture](docs/bioscan_architecture.png)

**Core Flow:**  
DNA / FASTA Input → Sequence QC → Feature Engineering → Hierarchical ML + Similarity Search → Evidence Engine → Distance Veto → Final Prediction / Out of Reference → Biodiversity Dashboard
