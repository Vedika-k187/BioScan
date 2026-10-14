"""
src/prediction/run_phase6_evaluation.py
Phase 6: Empirical Unknown Detection, Similarity Search, and Calibration Pipeline

Executes:
1. Leave-One-Reference-Out (LORO) similarity and confidence evaluation across all 101 reference barcodes.
2. Controlled pseudo-unknown simulation across varying divergence levels (0% to 30%).
3. Generates figures:
   - similarity_distance_distribution.png
   - confidence_distribution.png
   - threshold_analysis.png
4. Logs results to:
   - reports/phase6_similarity_results.csv
   - reports/phase6_unknown_detection_results.csv
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.similarity.reference_search import (
    ReferenceSearcher,
    levenshtein_distance,
    normalized_edit_distance
)
from src.prediction.predictor import BioScanPredictor, validate_and_clean_sequence
from src.prediction.confidence import (
    THRESHOLD_CLOSE_SIMILARITY,
    THRESHOLD_OUT_OF_REFERENCE
)
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
REF_BARCODES_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")

random.seed(42)
np.random.seed(42)

def mutate_sequence(seq: str, divergence_rate: float) -> str:
    """Introduce controlled nucleotide substitutions/indels to simulate phylogenetic divergence."""
    bases = ["A", "C", "G", "T"]
    seq_list = list(seq)
    n_mutations = max(1, int(len(seq) * divergence_rate)) if divergence_rate > 0 else 0
    indices = random.sample(range(len(seq)), min(n_mutations, len(seq)))
    
    for idx in indices:
        curr = seq_list[idx]
        other_bases = [b for b in bases if b != curr]
        seq_list[idx] = random.choice(other_bases)
    return "".join(seq_list)

def run_phase6_experiments():
    print("=" * 70, flush=True)
    print("PHASE 6: Unknown Detection & Similarity Search Evaluation", flush=True)
    print("=" * 70, flush=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    ref_df = pd.read_csv(REF_BARCODES_PATH)
    searcher = ReferenceSearcher(REF_BARCODES_PATH)
    predictor = BioScanPredictor()

    # -------------------------------------------------------------
    # 1. Leave-One-Reference-Out (LORO) Similarity Evaluation
    # -------------------------------------------------------------
    print("\n[Experiment 1] Running Leave-One-Reference-Out Similarity Search (N=101)...", flush=True)
    loro_records = []
    
    for idx, row in ref_df.iterrows():
        seq_id = row["sequence_id"]
        seq = row["sequence_normalized"]

        # Search library strictly excluding self
        matches = searcher.search(seq, top_k=5, exclude_id=seq_id)
        nearest = matches[0]
        second = matches[1] if len(matches) > 1 else matches[0]

        d_min = nearest["distance"]
        d_2 = second["distance"]
        margin_d = round(d_2 - d_min, 4)

        # Evaluate taxonomic concordance with nearest reference
        matches_class = (row["class"] == nearest["class"]) if pd.notnull(row["class"]) and pd.notnull(nearest["class"]) else None
        matches_order = (row["order"] == nearest["order"]) if pd.notnull(row["order"]) and pd.notnull(nearest["order"]) else None
        matches_family = (row["family"] == nearest["family"]) if pd.notnull(row["family"]) and pd.notnull(nearest["family"]) else None
        matches_genus = (row["genus"] == nearest["genus"]) if pd.notnull(row["genus"]) and pd.notnull(nearest["genus"]) else None
        matches_species = (row["species"] == nearest["species"]) if pd.notnull(row["species"]) and pd.notnull(nearest["species"]) and nearest["species"] != "Not available in reference data" else None

        rec = {
            "query_sequence_id": seq_id,
            "query_length": len(seq),
            "nearest_ref_id": nearest["sequence_id"],
            "nearest_distance": d_min,
            "nearest_similarity": nearest["similarity"],
            "second_ref_distance": d_2,
            "distance_margin": margin_d,
            "nearest_edit_dist": nearest["edit_distance"],
            "matches_class": matches_class,
            "matches_order": matches_order,
            "matches_family": matches_family,
            "matches_genus": matches_genus,
            "matches_species": matches_species,
            "nearest_class": nearest["class"],
            "nearest_order": nearest["order"],
            "nearest_family": nearest["family"],
            "nearest_genus": nearest["genus"],
            "nearest_species": nearest["species"]
        }
        loro_records.append(rec)

    sim_df = pd.DataFrame(loro_records)
    sim_path = os.path.join(REPORTS_DIR, "phase6_similarity_results.csv")
    sim_df.to_csv(sim_path, index=False)
    print(f"  Saved LORO similarity search results to: {sim_path}", flush=True)

    # -------------------------------------------------------------
    # 2. Controlled Pseudo-Unknown Simulation
    # -------------------------------------------------------------
    print("\n[Experiment 2] Running Controlled Pseudo-Unknown Divergence Simulation...", flush=True)
    # Divergence levels: 0% (exact reference), 2%, 5%, 10%, 15%, 20%, 30%
    divergence_levels = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    sample_pool = ref_df.sample(20, random_state=42)

    unknown_records = []
    for div in divergence_levels:
        for idx, row in sample_pool.iterrows():
            orig_seq = row["sequence_normalized"]
            test_seq = mutate_sequence(orig_seq, div) if div > 0 else orig_seq

            # Run predictor API
            pred_result = predictor.predict(test_seq, top_k_similar=5)

            rec = {
                "source_sequence_id": row["sequence_id"],
                "simulated_divergence": div,
                "simulated_divergence_pct": int(div * 100),
                "actual_nearest_distance": pred_result["evidence"]["nearest_distance"],
                "actual_nearest_similarity": pred_result["evidence"]["nearest_similarity"],
                "is_exact_match": pred_result["evidence"]["is_exact_match"],
                "status": pred_result["status"],
                "confidence_tier": pred_result["confidence_tier"],
                "model_top1_prob": pred_result["evidence"]["model_top1_prob"],
                "model_prob_margin": pred_result["evidence"]["model_prob_margin"],
                "class_agreement": pred_result["evidence"]["class_agreement"],
                "order_agreement": pred_result["evidence"]["order_agreement"],
                "predicted_class": pred_result["predicted_taxonomy"]["class"],
                "predicted_order": pred_result["predicted_taxonomy"]["order"],
                "predicted_family": pred_result["predicted_taxonomy"]["family"],
                "predicted_genus": pred_result["predicted_taxonomy"]["genus"],
                "predicted_species": pred_result["predicted_taxonomy"]["species"]
            }
            unknown_records.append(rec)

    unk_df = pd.DataFrame(unknown_records)
    unk_path = os.path.join(REPORTS_DIR, "phase6_unknown_detection_results.csv")
    unk_df.to_csv(unk_path, index=False)
    print(f"  Saved pseudo-unknown simulation results to: {unk_path}", flush=True)

    # Summarize detection rates across divergence levels
    div_summary = unk_df.groupby("simulated_divergence_pct")["status"].value_counts(normalize=True).unstack().fillna(0) * 100
    print("\n--- Out-of-Reference Detection Rate by Divergence Level ---", flush=True)
    print(div_summary.round(1), flush=True)

    # -------------------------------------------------------------
    # 3. Figures & Calibration Plots
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    # Figure 1: Similarity Distance Distribution (LORO)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(sim_df["nearest_distance"], bins=15, kde=True, color="#2b5c8f", ax=ax)
    ax.axvline(THRESHOLD_CLOSE_SIMILARITY, color="green", linestyle="--", linewidth=1.8, label=f"Known Reference Boundary (d <= {THRESHOLD_CLOSE_SIMILARITY})")
    ax.axvline(THRESHOLD_OUT_OF_REFERENCE, color="red", linestyle="--", linewidth=1.8, label=f"Out-of-Reference Boundary (d > {THRESHOLD_OUT_OF_REFERENCE})")
    ax.set_title("Nearest Reference Distance Distribution (Leave-One-Reference-Out)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Normalized Edit Distance to Nearest Reference Barcode", fontsize=10)
    ax.set_ylabel("Reference Sequence Count", fontsize=10)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "similarity_distance_distribution.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"\n[Figures] Saved distance distribution to: {fig1_path}", flush=True)

    # Figure 2: Model Confidence vs Distance Margin
    fig, ax = plt.subplots(figsize=(7, 4.5))
    palette = {"known_reference_like": "#2ca02c", "low_confidence": "#ff7f0e", "out_of_reference": "#d62728"}
    sns.scatterplot(
        data=unk_df,
        x="actual_nearest_distance",
        y="model_top1_prob",
        hue="status",
        palette=palette,
        s=60,
        ax=ax
    )
    ax.axvline(THRESHOLD_OUT_OF_REFERENCE, color="red", linestyle=":", label="Out-of-Ref Threshold (0.15)")
    ax.set_title("Confidence Space: Reference Distance vs Model Probability", fontsize=12, fontweight="bold")
    ax.set_xlabel("Normalized Distance to Nearest Reference", fontsize=10)
    ax.set_ylabel("Class Level Top-1 Model Probability", fontsize=10)
    ax.legend(title="Detected Status", loc="lower left", frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "confidence_distribution.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"[Figures] Saved confidence distribution to: {fig2_path}", flush=True)

    # Figure 3: Threshold Analysis Curve
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plot_df = div_summary.reset_index()
    for col, color in [("known_reference_like", "#2ca02c"), ("low_confidence", "#ff7f0e"), ("out_of_reference", "#d62728")]:
        if col in plot_df.columns:
            ax.plot(plot_df["simulated_divergence_pct"], plot_df[col], marker="o", linewidth=2.2, label=col.replace("_", " ").title(), color=color)
    ax.set_title("Status Classification Across Simulated Sequence Divergence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Simulated Sequence Divergence (%)", fontsize=10)
    ax.set_ylabel("Percentage of Sequences (%)", fontsize=10)
    ax.set_ylim(-5, 105)
    ax.set_xticks(plot_df["simulated_divergence_pct"])
    ax.legend(title="Assigned Status", loc="center right", frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "threshold_analysis.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"[Figures] Saved threshold analysis curve to: {fig3_path}", flush=True)

    print("\nPhase 6 experiments and calibration complete!", flush=True)

if __name__ == "__main__":
    run_phase6_experiments()
