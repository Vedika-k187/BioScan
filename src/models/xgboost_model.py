"""
BioScan Phase 4: XGBoost Species Classification Pipeline
Module: src/models/xgboost_model.py

Evaluates XGBoost across 9 DNA feature representations under:
- Full Reference Library Retrieval (Top-1, Top-3, Top-5 accuracy)
- Strict Leave-One-Out Holdout Evaluation (distinguishing structurally impossible singletons from evaluable multi-instance classes)
- Detailed probability distribution and confidence margin analysis

Extracts feature importances (gain), saves the best XGBoost baseline, and exports reports.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

REF_BARCODES_PATH = os.path.join("data", "processed", "reference_barcodes.csv")
PROCESSED_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models")
REPORTS_DIR = os.path.join("reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

FEATURE_SETS = {
    "Basic": os.path.join(PROCESSED_DIR, "features_basic.csv"),
    "k=2": os.path.join(PROCESSED_DIR, "features_k2.csv"),
    "k=3": os.path.join(PROCESSED_DIR, "features_k3.csv"),
    "k=4": os.path.join(PROCESSED_DIR, "features_k4.csv"),
    "k=5": os.path.join(PROCESSED_DIR, "features_k5.csv"),
    "Basic + k=2": os.path.join(PROCESSED_DIR, "features_basic_k2.csv"),
    "Basic + k=3": os.path.join(PROCESSED_DIR, "features_basic_k3.csv"),
    "Basic + k=4": os.path.join(PROCESSED_DIR, "features_basic_k4.csv"),
    "Basic + k=5": os.path.join(PROCESSED_DIR, "features_basic_k5.csv"),
}

def load_data(feature_set_name):
    """
    Load feature matrix for the 91 species-labeled sequences.
    Returns X_df, y_raw, y_encoded, label_encoder, sequence_ids, feature_cols.
    """
    ref_df = pd.read_csv(REF_BARCODES_PATH)
    labeled_mask = ref_df["species"].notnull()
    labeled_ref = ref_df[labeled_mask].copy()

    feature_path = FEATURE_SETS[feature_set_name]
    feat_df = pd.read_csv(feature_path)

    merged = pd.merge(labeled_ref[["sequence_id", "species"]], feat_df, on="sequence_id")
    y_raw = merged["species"].values
    seq_ids = merged["sequence_id"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)

    feature_cols = [c for c in feat_df.columns if c != "sequence_id"]
    X_df = merged[feature_cols]

    return X_df, y_raw, y_encoded, le, seq_ids, feature_cols

def compute_top_k(y_true_encoded, prob_matrix, k):
    """Calculate top-k accuracy from predicted probability matrix."""
    correct = 0
    for true_idx, probs in zip(y_true_encoded, prob_matrix):
        top_k_indices = np.argsort(probs)[::-1][:k]
        if true_idx in top_k_indices:
            correct += 1
    return correct / len(y_true_encoded)

def analyze_probabilities(prob_matrix):
    """
    Compute confidence distribution metrics:
    - mean max probability
    - mean top-1 vs top-2 probability margin
    - mean entropy of predicted distribution
    """
    max_probs = np.max(prob_matrix, axis=1)
    
    # Sort descending across classes for each sample
    sorted_probs = np.sort(prob_matrix, axis=1)[:, ::-1]
    top1_probs = sorted_probs[:, 0]
    top2_probs = sorted_probs[:, 1]
    margins = top1_probs - top2_probs
    
    # Shannon entropy of probability distribution (in bits)
    # Clip to avoid log(0)
    clipped = np.clip(prob_matrix, 1e-12, 1.0)
    entropy = -np.sum(clipped * np.log2(clipped), axis=1)
    
    return {
        "mean_max_prob": float(np.mean(max_probs)),
        "std_max_prob": float(np.std(max_probs)),
        "mean_margin": float(np.mean(margins)),
        "mean_entropy": float(np.mean(entropy))
    }

def run_phase4_experiments():
    print("[Phase 4] Starting XGBoost Species Classification Experiments...", flush=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Hyperparameter search grid
    n_estimators_list = [100, 200]
    max_depth_list = [3, 5]
    learning_rate_list = [0.05, 0.1]
    subsample = 0.8
    colsample_bytree = 0.8
    random_state = 42

    results = []
    trained_models = {}

    ref_df = pd.read_csv(REF_BARCODES_PATH)
    labeled_mask = ref_df["species"].notnull()
    labeled_ref = ref_df[labeled_mask]
    sp_counts = labeled_ref["species"].value_counts()
    
    # Detect multi-instance vs singleton classes
    multi_species = sp_counts[sp_counts > 1].index.tolist()
    evaluable_indices = np.where(labeled_ref["species"].isin(multi_species))[0]
    print(f"    Total labeled sequences: {len(labeled_ref)}", flush=True)
    print(f"    Total species classes: {len(sp_counts)}", flush=True)
    print(f"    Evaluable holdout species: {multi_species} (indices: {evaluable_indices.tolist()})", flush=True)
    print(f"    Structurally impossible singleton species: {len(sp_counts) - len(multi_species)} (89 sequences)", flush=True)

    for feat_name in FEATURE_SETS.keys():
        print(f"\n--- Feature Representation: {feat_name} ---", flush=True)
        X_df, y_raw, y_encoded, le, seq_ids, feature_cols = load_data(feat_name)
        X = X_df.values
        num_classes = len(le.classes_)

        for n_est in n_estimators_list:
            for depth in max_depth_list:
                for lr in learning_rate_list:
                    cfg_str = f"n_est={n_est}, depth={depth}, lr={lr}"
                    
                    # 1. Full Reference Library Model (Indexing / Retrieval)
                    clf = xgb.XGBClassifier(
                        n_estimators=n_est,
                        max_depth=depth,
                        learning_rate=lr,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        random_state=random_state,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        n_jobs=-1
                    )
                    clf.fit(X, y_encoded)
                    ref_probs = clf.predict_proba(X)
                    ref_preds = np.argmax(ref_probs, axis=1)

                    ref_top1 = compute_top_k(y_encoded, ref_probs, k=1)
                    ref_top3 = compute_top_k(y_encoded, ref_probs, k=3)
                    ref_top5 = compute_top_k(y_encoded, ref_probs, k=5)
                    
                    prob_stats = analyze_probabilities(ref_probs)

                    # 2. Leave-One-Out Holdout Evaluation on Evaluable Species (Ophidion marginatum)
                    evaluable_correct = 0
                    evaluable_probs = []
                    for e_idx in evaluable_indices:
                        train_idx = [i for i in range(len(y_encoded)) if i != e_idx]
                        holdout_clf = xgb.XGBClassifier(
                            n_estimators=n_est,
                            max_depth=depth,
                            learning_rate=lr,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            random_state=random_state,
                            objective="multi:softprob",
                            eval_metric="mlogloss",
                            n_jobs=-1
                        )
                        holdout_clf.fit(X[train_idx], y_encoded[train_idx])
                        h_prob = holdout_clf.predict_proba(X[[e_idx]])[0]
                        evaluable_probs.append(h_prob)
                        if np.argmax(h_prob) == y_encoded[e_idx]:
                            evaluable_correct += 1

                    evaluable_acc = evaluable_correct / len(evaluable_indices)
                    
                    # Total LOOCV accuracy across all 91 samples:
                    # Singletons are mathematically 0 correct; evaluable contributes evaluable_correct
                    loocv_total_acc = evaluable_correct / len(y_encoded)

                    rec = {
                        "model": "XGBoost",
                        "feature_set": feat_name,
                        "n_features": X.shape[1],
                        "n_estimators": n_est,
                        "max_depth": depth,
                        "learning_rate": lr,
                        "subsample": subsample,
                        "colsample_bytree": colsample_bytree,
                        "config": cfg_str,
                        "ref_top1_accuracy": round(ref_top1, 4),
                        "ref_top3_accuracy": round(ref_top3, 4),
                        "ref_top5_accuracy": round(ref_top5, 4),
                        "evaluable_holdout_accuracy": round(evaluable_acc, 4),
                        "evaluable_holdout_samples": len(evaluable_indices),
                        "loocv_total_accuracy": round(loocv_total_acc, 4),
                        "loocv_total_samples": len(y_encoded),
                        "mean_max_prob": round(prob_stats["mean_max_prob"], 4),
                        "mean_prob_margin": round(prob_stats["mean_margin"], 4),
                        "mean_prob_entropy": round(prob_stats["mean_entropy"], 4),
                    }
                    results.append(rec)
                    trained_models[f"XGB_{feat_name}_{cfg_str}"] = (clf, feature_cols, le, rec)
                    print(f"    {cfg_str:30} | Ref Top-1={rec['ref_top1_accuracy']:.4f} | Top-3={rec['ref_top3_accuracy']:.4f} | EvalHoldout={rec['evaluable_holdout_accuracy']:.4f} | Margin={rec['mean_prob_margin']:.4f}", flush=True)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(REPORTS_DIR, "phase4_xgboost_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[Results] Saved all 72 experimental results to: {results_path}", flush=True)

    # -------------------------------------------------------------
    # 3. Model Selection
    # -------------------------------------------------------------
    # Rank by: Ref Top-1, then Top-3, then smaller feature dimension
    sorted_df = results_df.sort_values(
        by=["ref_top1_accuracy", "ref_top3_accuracy", "n_features"],
        ascending=[False, False, True]
    )
    best_rec = sorted_df.iloc[0]
    best_key = f"XGB_{best_rec['feature_set']}_{best_rec['config']}"
    best_model, best_cols, best_le, _ = trained_models[best_key]

    print("\n[Best Model Selected]")
    print(f"    Feature Representation: {best_rec['feature_set']} ({best_rec['n_features']} features)")
    print(f"    Configuration: {best_rec['config']}")
    print(f"    Ref Top-1: {best_rec['ref_top1_accuracy']:.4f}, Ref Top-3: {best_rec['ref_top3_accuracy']:.4f}")
    print(f"    Mean Confidence Margin: {best_rec['mean_prob_margin']:.4f}")

    # Save best model artifact
    model_save_path = os.path.join(MODELS_DIR, "xgboost_species.joblib")
    joblib.dump({
        "model": best_model,
        "feature_set": best_rec["feature_set"],
        "feature_columns": best_cols,
        "label_encoder": best_le,
        "target_classes": best_le.classes_.tolist(),
        "config": best_rec["config"],
        "metrics": best_rec,
        "random_state": random_state,
        "xgboost_version": xgb.__version__
    }, model_save_path)
    print(f"    Saved model artifact to: {model_save_path}", flush=True)

    # -------------------------------------------------------------
    # 4. Feature Importance Extraction (Gain)
    # -------------------------------------------------------------
    # Extract feature importance by gain
    booster = best_model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    
    # Map f0, f1... or feature names to importance
    fi_list = []
    for i, col in enumerate(best_cols):
        # XGBoost booster score can use col name or f{i}
        val = score_gain.get(col, score_gain.get(f"f{i}", 0.0))
        fi_list.append({"feature": col, "importance_gain": float(val)})

    fi_df = pd.DataFrame(fi_list).sort_values(by="importance_gain", ascending=False)
    fi_path = os.path.join(REPORTS_DIR, "xgboost_feature_importance.csv")
    fi_df.to_csv(fi_path, index=False)
    print(f"    Saved feature importance to: {fi_path}", flush=True)

    # -------------------------------------------------------------
    # 5. Visualizations
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot: Top-20 XGBoost Feature Importances (Gain)
    top20_fi = fi_df.head(20)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=top20_fi, x="importance_gain", y="feature", palette="mako", ax=ax)
    ax.set_title(f"Top 20 Features by Model-Derived Gain in XGBoost ({best_rec['feature_set']})", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Average Gain per Split (Model-Derived Importance)", fontsize=10)
    ax.set_ylabel("DNA Feature", fontsize=10)
    for i, v in enumerate(top20_fi["importance_gain"]):
        ax.text(v + (top20_fi["importance_gain"].max() * 0.01), i, f"{v:.4f}", va="center", fontsize=8.5)
    plt.tight_layout()
    fi_plot_path = os.path.join(FIGURES_DIR, "xgboost_feature_importance.png")
    fig.savefig(fi_plot_path, dpi=200)
    plt.close(fig)
    print(f"    Saved plot to: {fi_plot_path}", flush=True)

    return {
        "best_record": best_rec,
        "results_count": len(results),
        "fi_top5": top20_fi.head(5).to_dict(orient="records")
    }

if __name__ == "__main__":
    res = run_phase4_experiments()
    print("\nPhase 4 XGBoost Experiments completed successfully.", flush=True)
