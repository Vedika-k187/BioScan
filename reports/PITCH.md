# BioScan Project Pitch

**Project**: BioScan — AI-Powered DNA Biodiversity Identification & Monitoring System  
**Tagline**: *From Environmental DNA to Transparent Taxonomic Evidence, Unknown Detection, and Biodiversity Intelligence.*  

---

## 1. 30-Second Elevator Pitch

> *"Environmental DNA monitoring should revolutionize marine conservation, but current AI classifiers act like dangerous black boxes: they force a single species name and assign 99% confidence even to novel, uncataloged organisms.  
> **BioScan** solves this. Built on 12S rDNA barcode data, BioScan combines parent-restricted hierarchical machine learning with normalized Levenshtein sequence retrieval and an independent **Distance Veto**. When a sequence is uncataloged, BioScan refuses to hallucinate a species—it flags it as `out_of_reference` and suppresses false geographic projections. With 115 passing tests and an interactive Streamlit dashboard, BioScan brings scientific transparency and reproducibility to modern biomonitoring."*

---

## 2. 60-Second Hackathon Stage Pitch

> *"Good afternoon, judges. Today, ocean conservationists collect water samples containing billions of DNA fragments from passing marine life. But when you feed those sequences into standard machine learning models, two catastrophic failure modes occur:  
> First, they provide zero scientific justification—just an isolated species name without rank resolution or edit distance metrics.  
> Second, and worse: because closed-set classifiers must sum softmax probabilities to 1.0, they assign high confidence even to completely unknown or out-of-reference organisms.  
> 
> We built **BioScan** to bridge the gap between machine learning and rigorous ecological science.  
> BioScan executes an end-to-end, multi-stage pipeline: strict biological quality control, k-mer feature extraction, parent-restricted hierarchical prediction across five taxonomic ranks, and pairwise normalized Levenshtein retrieval against cataloged reference barcodes.  
> 
> Our key innovation is the **Distance Veto**: if a query sequence diverges by more than 15% from cataloged barcodes, BioScan strictly overrides the classifier and assigns `out_of_reference`. It suppresses geographic distribution projection and warns the ecologist.  
> 
> BioScan also integrates 2019 New Jersey Ocean Trawl Survey occurrence metadata, mapping station-level abundance-weighted Shannon diversity. Tested against 115 unit and integration tests with zero regressions, BioScan is an open-source, hackathon-ready intelligence platform."*

---

## 3. 2-Minute Keynote & Technical Deep-Dive Pitch

> *"Judges and fellow researchers: Marine biodiversity is declining at an unprecedented rate, and non-invasive environmental DNA (eDNA) is our best sensor. However, the software layer analyzing eDNA has lagged behind the genomics. Ecologists are trapped between computationally intensive BLAST searches and opaque black-box deep learning models that hallucinate species on uncataloged data.  
> 
> Consider the real-world dataset structure: in our reference catalog of 101 unique 12S rDNA barcodes, **89 of the 90 species are singletons**. Standard k-fold cross-validation at the species level is a statistical illusion in this regime. Any system claiming '100% generalized species accuracy' on singletons is confusing reference library re-identification with real-world biology.  
> 
> **BioScan** was engineered with scientific conservatism at its core:  
> 1. **Biological QC & Zero-Leakage Features**: Sequences are strictly validated for canonical nucleotides $\{A,C,G,T\}$. Metadata such as coordinates and cruise dates are strictly barred from entering the feature space.  
> 2. **Hierarchical Taxonomy**: Instead of a flat classifier, BioScan predicts Class $\rightarrow$ Order $\rightarrow$ Family $\rightarrow$ Genus $\rightarrow$ Species conditionally, preserving partial taxonomy when reference labels are unassigned.  
> 3. **The Distance Veto**: Standard closed-set models cannot say 'I don't know.' BioScan pairs ML with normalized Levenshtein search. If $d_{\text{min}} > 0.150$, the Distance Veto triggers, strictly categorizing the sequence as `out_of_reference`.  
> 4. **Survey-Bounded Biodiversity Intelligence**: For verified reference matches, BioScan connects detections to 36 offshore New Jersey sampling stations, computing abundance-weighted Shannon diversity ($H_{\text{reads}}$) while explicitly warning users that detections represent survey events rather than global species limits.  
> 
> All of this is delivered through an interactive, multi-page Streamlit dashboard featuring single sequence analysis, fault-tolerant batch FASTA processing, and dynamic Plotly maps.  
> 
> With 115 automated tests passed and raw dataset hash integrity preserved, BioScan is a reliable, reproducible, and scientifically grounded product ready for immediate evaluation."*
