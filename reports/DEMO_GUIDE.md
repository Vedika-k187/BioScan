# BioScan 3–5 Minute Hackathon Live Demonstration Script

**Target Audience**: Hackathon Judges, Marine Ecologists, ML Engineers  
**Time Limit**: 3 to 5 Minutes  
**Command to Launch**: `streamlit run app/app.py`  

---

## Preparation (30 Seconds Before Demo)
1. Open terminal in workspace root.
2. Launch the dashboard:
   ```bash
   streamlit run app/app.py
   ```
3. Open browser at `http://localhost:8501`. Ensure the sidebar is expanded on the **🔬 Identify DNA** tab.

---

## Live Demo Workflow

### Act 1: The Core Problem & A Known Reference Match (90 Seconds)

**What to Say:**
> *"Traditional environmental DNA pipelines either return a black-box species name with zero confidence context, or suffer from closed-set overconfidence on novel taxa. BioScan solves this with a transparent, multi-stage evidence engine. Let's see it in action."*

**What to Do:**
1. Point out the capability banner at the top of the landing page: **12S rDNA V5 teleost and elasmobranch library**.
2. Click the dropdown: **"Select a pre-loaded reference sequence or synthetic test case"**.
3. Choose: **`Exact Reference — Ophidion marginatum (Striped cusk-eel)`**.
4. Click **🚀 Identify Sequence**.

**What to Point Out:**
- **Biological QC Panel**: Sequence is 106 bp, 100% valid $\{A, C, G, T\}$ bases, 48.1% GC content.
- **Identification Card**: Status shows `🟢 Reference-Library Match`. Predicted species: *Ophidion marginatum*, Class: *Teleostei*, Order: *Ophidiiformes*. Note the explicit tag: `"species_prediction_scope: reference-library constrained"`.
- **Taxonomic Hierarchy**: Visual step progression from Kingdom down to Species.
- **Confidence & Evidence**: High probability ($>95\%$), wide margin, near-zero entropy, distance $0.000$, and 100% taxonomic concordance across ranks.
- **Top-5 Closest References**: Rank 1 highlighted with ⭐ Exact Match.
- **Survey Occurrence Map**: Shows stations in the New Jersey Continental Shelf where this barcode was detected during the 2019 survey, with marker size proportional to eDNA amplification read count.
- **Scientific Caveat**: Point out the notice explaining that this represents survey sampling, not global biological species range.

---

### Act 2: Unknown DNA & The Distance Veto (60 Seconds)

**What to Say:**
> *"Now here is what happens when a standard closed-set model encounters an unknown or divergent organism: it hallucinates a known fish with 99% probability. BioScan prevents this failure with an independent sequence edit distance veto."*

**What to Do:**
1. In the example dropdown, select: **`Out-of-Reference Sequence (Divergent Non-Library Barcode)`**.
2. Click **🚀 Identify Sequence**.

**What to Point Out:**
- **Status Alert**: Prominent crimson banner: `🔴 Out of Reference Library`.
- **Mandatory Scientific Warning**:
  > *"Sequence appears outside the current reference library. This does not prove that the sequence represents a new species. It indicates that the sequence is sufficiently distant from the current BioScan reference library."*
- **The Distance Veto**: The sequence edit distance is $>0.150$ ($15\%$ divergence). Even if the underlying neural net or tree model outputs a class probability, the Distance Veto **overrides** classifier confidence.
- **Geographic Suppression**: Point out that the map is **strictly suppressed** (`NO_PROJECTION_PERMITTED`). BioScan refuses to invent a habitat for an unverified taxon.

---

### Act 3: High-Throughput Batch FASTA Analysis (60 Seconds)

**What to Say:**
> *"In real biomonitoring, researchers process hundreds of eDNA reads at once. BioScan provides fault-tolerant batch classification with sequence-level isolation."*

**What to Do:**
1. In the sidebar navigation, click: **📊 Batch Analysis**.
2. Click the button: **🧪 Load Sample Demonstration Batch**.

**What to Point Out:**
- **Fault Isolation**: The 5-sequence batch contains an exact match, a variant, a low-confidence sample, an out-of-reference sequence, and an invalid record containing ambiguous `N` bases.
- Show that the invalid sequence is cleanly flagged as `QC Failed` **without terminating or crashing the other 4 sequences**.
- **Interactive KPI Cards & Charts**: Point out the Status Distribution donut chart and Predicted Class breakdown.
- **Filtered Table**: Filter by Status $\rightarrow$ `out_of_reference`.
- Click **📥 Download Batch Results (CSV)** to show flat tabular export ready for field reporting.

---

### Act 4: Biodiversity & Survey Intelligence Explorer (45 Seconds)

**What to Say:**
> *"BioScan doesn't just identify sequences; it aggregates eDNA occurrences across geographic space to calculate ecological diversity metrics."*

**What to Do:**
1. In the sidebar navigation, click: **🌍 Biodiversity Explorer**.

**What to Point Out:**
- **Reference Catalog KPIs**: 101 barcodes, 90 species, 87 genera, 66 families, 30 orders, 4 classes.
- **Sampling Stations Map**: 36 distinct sampling stations from the 2019 survey, colored by abundance-weighted Shannon diversity ($H_{\text{reads}}$) with circles scaled by species richness.
- **Depth Correlation Plot**: Scatter plot analyzing station depth vs. Shannon diversity.
- **Scientific Notice**: Clarify that read count represents operational amplification signal, not direct biomass.

---

### Act 5: Wrap-up & Judge Takeaway (15 Seconds)

**What to Say:**
> *"To summarize: BioScan is not a black box claiming 100% universal accuracy. It is a scientifically conservative, reproducible, multi-tiered AI evidence engine with 115 passing tests and verified bit-for-bit raw data integrity. Thank you!"*
