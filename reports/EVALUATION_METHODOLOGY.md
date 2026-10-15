# BioScan Evaluation Methodology and Extension Plan

Generated results: `reports/evaluation/evaluation_results.json`
Regenerate: `PYTHONPATH=. python src/evaluation/run_model_evaluation.py`
Dashboard: sidebar -> **Model Evaluation** (reads the JSON only; nothing is hardcoded).

## 1. Principles

1. Evaluation is done on **unique normalised sequences**, never on the 13,736 duplicated survey rows.
2. Only **held-out predictions** are scored. Training-set accuracy is never reported as generalisation.
3. Evaluation code trains throw-away models inside folds and never modifies `models/*.joblib`
   or the production training code.
4. A rank without enough held-out data shows **"Insufficient data for reliable evaluation"**
   and no numbers (minimum: 20 evaluable sequences and 2 evaluable classes).
5. The Distance Veto, QC, evidence engine and abstention logic are only **read**, never changed.

## 2. Taxonomy classifiers

- Method: leave-one-sequence-out over unique sequences. A sequence is evaluated only if another
  sequence with the same label stays in training; otherwise its true class is absent from training
  and a closed-set model cannot predict it. Such sequences are **counted and reported**, not hidden.
- Scaler is fitted on each training fold only. The split is deterministic (no RNG).
- Feature set per rank is the one used by the production artefact (`models/taxonomy_<rank>_model.joblib`).
  KNN (k=1, Manhattan), Random Forest (100 trees) and XGBoost (100 trees, depth 3) are compared with
  the same settings as Phase 5.
- Macro metrics average over **evaluated classes** (classes present in the held-out ground truth).
- `Top-1 accuracy` is the only top-k reported; top-3/5 are not meaningful with so few classes per rank.
- Not used: AUPRC, Brier score, expected calibration error. Production KNN uses k=1 so its
  probabilities are 0/1, and per-class support is too small for reliable calibration estimates.

## 3. Unknown-DNA (out-of-reference) detection

Positive class: **out-of-reference**.

| Category | Construction | True label | Establishes | Does not establish |
|---|---|---|---|---|
| Known (exact) | 101 reference sequences vs full library | known | Identical sequences are accepted | Anything about generalisation (distance = 0) |
| Altered known | 1, 2 and 5 random substitutions per reference (>= 1 guaranteed) | known (by construction) | Tolerance to substitution errors | Indels, chimeras, real read errors |
| Held-out genus | Genus removed from library, then queried | out-of-reference at genus level | Behaviour on taxa with no own reference | That the species differs from every library species |
| External non-fish | 106-bp windows of NCBI mammal/bird/reptile/amphibian 12S records | out-of-reference | Rejection of distant vertebrate DNA | Real amplicon reads, non-vertebrates, unknown fish |

Calibration/test separation: reference sequences, genera and external organisms are each split
50/50 with a fixed seed. A threshold is chosen (Youden's J) on the calibration half only and
reported on the test half. **The production thresholds (0.03 / 0.15) were originally chosen in an
earlier experiment that saw all of these sequences, so their test-half numbers are optimistic.**

Decision rule evaluated: the production `synthesize_evidence_and_status` with class/order
concordance assumed true, so only the distance rules are exercised. Reported coverage is therefore
an upper bound and classifier probabilities are not evaluated here.

## 4. Metrics that must not be over-interpreted

- Class level: only 2 of 4 classes (Teleostei, Elasmobranchii) can be held out; the other two are
  singletons. Near-perfect scores reflect an easy, almost binary task.
- Order/family: mean support per class is ~5 and ~3 test sequences; per-class values are unstable.
- Genus and species: not evaluated (insufficient data). Species output is reference matching.
- Held-out-genus results measure distance to the closest library relative, not true novelty.
- Non-fish negatives are windows of full-length records, not amplicon reads.

## 5. Extension plan (NOT implemented)

| Need | Why it matters | Requirement |
|---|---|---|
| More unique, independently sourced sequences per species | Species-level evaluation needs >= 2 independent sequences per species | New sequences from MitoFish/NCBI/BOLD with vouchered identifications |
| External validation on unseen taxa | Current "unknown" tests derive from the same 101 sequences | A separate reference set from a different survey and region |
| Trusted search benchmark (BLAST/vsearch) | Establish whether the ML adds anything over standard search | Run BLAST on the same held-out queries and compare accuracy and abstention |
| Threshold and confidence calibration | Current thresholds come from ~200 queries | Larger labelled known/unknown sets; fit and test on disjoint sets; report intervals |
| FASTQ-to-ASV preprocessing | Real eDNA data are noisy reads, not clean amplicons | Primer trimming, denoising (DADA2/Deblur/vsearch), chimera removal, ASV table input |
