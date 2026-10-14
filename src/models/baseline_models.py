"""
BioScan Phase 3: Baseline Machine Learning Pipeline
Module: src/models/baseline_models.py

Evaluates baseline supervised species classification models:
1. K-Nearest Neighbors (KNN) with StandardScaler in scikit-learn Pipeline
2. Random Forest Classifier

Evaluates across 9 distinct DNA feature representations using:
- Leave-One-Out Cross-Validation (LOOCV) for strict out-of-sample holdout
- Reference Library Top-1, Top-3, and Top-5 retrieval accuracy

Saves baseline models, feature importances, and evaluation reports.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Suppress all scikit-learn warnings regarding class counts > 50% of samples
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
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

def load_species_modeling_data(feature_set_name):
    """
    Load feature matrix filtered strictly to the 91 species-labeled sequences.
    Excludes all metadata and taxonomy from X.
    """
    ref_df = pd.read_csv(REF_BARCODES_PATH)
    labeled_mask = ref_df["species"].notnull()
    labeled_ref = ref_df[labeled_mask].copy()

    feature_path = FEATURE_SETS[feature_set_name]
    feat_df = pd.read_csv(feature_path)

    # Merge on sequence_id to ensure exact row alignment
    merged = pd.merge(labeled_ref[["sequence_id", "species"]], feat_df, on="sequence_id")
    
    y = merged["species"].values
    seq_ids = merged["sequence_id"].values
    
    # Drop non-feature columns
    feature_cols = [c for c in feat_df.columns if c != "sequence_id"]
    X_df = merged[feature_cols]
    
    return X_df, y, seq_ids, feature_cols

def compute_top_k_accuracy(y_true, y_probs, classes, k):
    """
    Calculate top-k accuracy from predicted class probability distributions.
    """
    correct = 0
    for true_label, prob_dist in zip(y_true, y_probs):
        top_k_indices = np.argsort(prob_dist)[::-1][:k]
        top_k_classes = [classes[idx] for idx in top_k_indices]
        if true_label in top_k_classes:
            correct += 1
    return correct / len(y_true)

def evaluate_loocv(model_factory, X, y):
    """
    Execute Leave-One-Out Cross-Validation (LOOCV) without data leakage.
    Returns out-of-fold accuracy, macro F1, and weighted F1.
    """
    loo = LeaveOneOut()
    y_preds = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Instantiate fresh model pipeline for the fold
        model = model_factory()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        y_preds.append(pred)

    y_preds = np.array(y_preds)
    
    acc = accuracy_score(y, y_preds)
    macro_f1 = f1_score(y, y_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y, y_preds, average="weighted", zero_division=0)
    
    return {
        "loocv_accuracy": acc,
        "loocv_macro_f1": macro_f1,
        "loocv_weighted_f1": weighted_f1,
        "y_preds": y_preds
    }

def evaluate_reference_retrieval(model_factory, X, y):
    """
    Fit on the full reference library and evaluate top-1, top-3, and top-5 retrieval.
    """
    model = model_factory()
    model.fit(X, y)
    probs = model.predict_proba(X)
    classes = model.classes_
    
    top1 = compute_top_k_accuracy(y, probs, classes, k=1)
    top3 = compute_top_k_accuracy(y, probs, classes, k=3)
    top5 = compute_top_k_accuracy(y, probs, classes, k=5)
    
    return top1, top3, top5, model

def run_phase3_experiment():
    print("[Phase 3] Starting Baseline Machine Learning Experiments...", flush=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Verify class distribution
    ref_df = pd.read_csv(REF_BARCODES_PATH)
    labeled_ref = ref_df[ref_df["species"].notnull()]
    sp_counts = labeled_ref["species"].value_counts()
    
    print(f"          Total reference barcodes: {len(ref_df)}", flush=True)
    print(f"          Species-labeled barcodes: {len(labeled_ref)}", flush=True)
    print(f"          Distinct species: {len(sp_counts)}", flush=True)
    print(f"          Species with 1 sequence: {(sp_counts == 1).sum()} ({(sp_counts == 1).sum()/len(sp_counts)*100:.1f}%)", flush=True)
    print(f"          Species with >1 sequences: {(sp_counts > 1).sum()}", flush=True)
    print("          LOOCV selected as valid holdout evaluation scheme.", flush=True)

    results_records = []
    trained_models = {}

    # Define hyperparameter configs
    knn_configs = [1, 3, 5]
    rf_configs = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 300, "max_depth": None},
    ]

    for feat_name in FEATURE_SETS.keys():
        print(f"\n--- Evaluating Feature Representation: {feat_name} ---", flush=True)
        X_df, y, seq_ids, feature_cols = load_species_modeling_data(feat_name)
        X = X_df.values
        n_features = X.shape[1]
        print(f"    Dimensions: {X.shape[0]} samples x {n_features} features", flush=True)

        # -------------------------------------------------------------
        # 1. Evaluate KNN Configurations
        # -------------------------------------------------------------
        for k_val in knn_configs:
            def knn_factory(k=k_val):
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean"))
                ])
            
            # LOOCV
            loocv_res = evaluate_loocv(knn_factory, X, y)
            
            # Reference retrieval
            top1, top3, top5, fitted_model = evaluate_reference_retrieval(knn_factory, X, y)
            
            config_str = f"n_neighbors={k_val}"
            rec = {
                "model": "KNN",
                "feature_set": feat_name,
                "n_features": n_features,
                "config": config_str,
                "cv_strategy": "LOOCV",
                "loocv_accuracy": round(loocv_res["loocv_accuracy"], 4),
                "loocv_macro_f1": round(loocv_res["loocv_macro_f1"], 4),
                "loocv_weighted_f1": round(loocv_res["loocv_weighted_f1"], 4),
                "ref_top1_accuracy": round(top1, 4),
                "ref_top3_accuracy": round(top3, 4),
                "ref_top5_accuracy": round(top5, 4),
            }
            results_records.append(rec)
            trained_models[f"KNN_{feat_name}_{config_str}"] = (fitted_model, feature_cols, rec)
            print(f"    KNN ({config_str}): LOOCV Acc={rec['loocv_accuracy']}, Ref Top-1={rec['ref_top1_accuracy']}, Top-3={rec['ref_top3_accuracy']}", flush=True)

        # -------------------------------------------------------------
        # 2. Evaluate Random Forest Configurations
        # -------------------------------------------------------------
        for rp in rf_configs:
            n_est = rp["n_estimators"]
            m_depth = rp["max_depth"]
            
            def rf_factory(n=n_est, d=m_depth):
                return RandomForestClassifier(
                    n_estimators=n,
                    max_depth=d,
                    random_state=42,
                    n_jobs=1
                )
            
            # LOOCV
            loocv_res = evaluate_loocv(rf_factory, X, y)
            
            # Reference retrieval
            top1, top3, top5, fitted_model = evaluate_reference_retrieval(rf_factory, X, y)
            
            config_str = f"n_est={n_est}, max_depth={m_depth}"
            rec = {
                "model": "Random Forest",
                "feature_set": feat_name,
                "n_features": n_features,
                "config": config_str,
                "cv_strategy": "LOOCV",
                "loocv_accuracy": round(loocv_res["loocv_accuracy"], 4),
                "loocv_macro_f1": round(loocv_res["loocv_macro_f1"], 4),
                "loocv_weighted_f1": round(loocv_res["loocv_weighted_f1"], 4),
                "ref_top1_accuracy": round(top1, 4),
                "ref_top3_accuracy": round(top3, 4),
                "ref_top5_accuracy": round(top5, 4),
            }
            results_records.append(rec)
            trained_models[f"RF_{feat_name}_{config_str}"] = (fitted_model, feature_cols, rec)
            print(f"    RF ({config_str}): LOOCV Acc={rec['loocv_accuracy']}, Ref Top-1={rec['ref_top1_accuracy']}, Top-3={rec['ref_top3_accuracy']}", flush=True)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results_records)
    results_path = os.path.join(REPORTS_DIR, "phase3_model_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[Results] Saved complete results table to: {results_path}", flush=True)

    # -------------------------------------------------------------
    # 3. Select Best Baseline Models
    # -------------------------------------------------------------
    knn_results = results_df[results_df["model"] == "KNN"].sort_values(
        by=["ref_top1_accuracy", "loocv_accuracy", "n_features"], ascending=[False, False, True]
    )
    best_knn_rec = knn_results.iloc[0]
    best_knn_key = f"KNN_{best_knn_rec['feature_set']}_{best_knn_rec['config']}"
    best_knn_model, best_knn_features, _ = trained_models[best_knn_key]
    
    knn_model_path = os.path.join(MODELS_DIR, "baseline_knn.joblib")
    joblib.dump({
        "pipeline": best_knn_model,
        "feature_set": best_knn_rec["feature_set"],
        "feature_columns": best_knn_features,
        "config": best_knn_rec["config"],
        "metrics": best_knn_rec.to_dict(),
        "random_state": 42
    }, knn_model_path)
    print(f"          Saved best KNN model: {knn_model_path}", flush=True)

    rf_results = results_df[results_df["model"] == "Random Forest"].sort_values(
        by=["ref_top1_accuracy", "ref_top3_accuracy", "loocv_accuracy", "n_features"], ascending=[False, False, False, True]
    )
    best_rf_rec = rf_results.iloc[0]
    best_rf_key = f"RF_{best_rf_rec['feature_set']}_{best_rf_rec['config']}"
    best_rf_model, best_rf_features, _ = trained_models[best_rf_key]

    rf_model_path = os.path.join(MODELS_DIR, "baseline_random_forest.joblib")
    joblib.dump({
        "pipeline": best_rf_model,
        "feature_set": best_rf_rec["feature_set"],
        "feature_columns": best_rf_features,
        "config": best_rf_rec["config"],
        "metrics": best_rf_rec.to_dict(),
        "random_state": 42
    }, rf_model_path)
    print(f"          Saved best RF model: {rf_model_path}", flush=True)

    # -------------------------------------------------------------
    # 4. Extract Feature Importance for Best Random Forest
    # -------------------------------------------------------------
    rf_estimator = best_rf_model
    importances = rf_estimator.feature_importances_
    fi_df = pd.DataFrame({
        "feature": best_rf_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False)
    
    fi_csv_path = os.path.join(REPORTS_DIR, "random_forest_feature_importance.csv")
    fi_df.to_csv(fi_csv_path, index=False)
    print(f"          Saved feature importances to: {fi_csv_path}", flush=True)

    # -------------------------------------------------------------
    # 5. Visualizations
    # -------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot 1: Model Comparison (Top-1 Retrieval Accuracy across representations)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_df = results_df[
        (results_df["config"].isin(["n_neighbors=1", "n_est=100, max_depth=None"]))
    ].copy()
    sns.barplot(data=plot_df, x="feature_set", y="ref_top1_accuracy", hue="model", palette="Set2", ax=ax)
    ax.set_title("Species Reference Top-1 Accuracy Across DNA Representations", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("DNA Feature Representation", fontsize=10)
    ax.set_ylabel("Top-1 Retrieval Accuracy", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.tick_params(axis="x", rotation=25)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points")
    plt.tight_layout()
    comp_plot_path = os.path.join(FIGURES_DIR, "model_comparison.png")
    fig.savefig(comp_plot_path, dpi=200)
    plt.close(fig)
    print(f"          Saved plot: {comp_plot_path}", flush=True)

    # Plot 2: Top-20 Random Forest Feature Importances
    fig, ax = plt.subplots(figsize=(9, 5.5))
    top20_fi = fi_df.head(20)
    sns.barplot(data=top20_fi, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(f"Top 20 Features in Best Random Forest ({best_rf_rec['feature_set']})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Gini Feature Importance", fontsize=10)
    ax.set_ylabel("DNA Feature", fontsize=10)
    for i, v in enumerate(top20_fi["importance"]):
        ax.text(v + (top20_fi["importance"].max() * 0.01), i, f"{v:.4f}", va="center", fontsize=8.5)
    plt.tight_layout()
    fi_plot_path = os.path.join(FIGURES_DIR, "random_forest_feature_importance.png")
    fig.savefig(fi_plot_path, dpi=200)
    plt.close(fig)
    print(f"          Saved plot: {fi_plot_path}", flush=True)

    return {
        "best_knn": best_knn_rec.to_dict(),
        "best_rf": best_rf_rec.to_dict(),
        "top20_features": fi_df.head(20).to_dict(orient="records")
    }

if __name__ == "__main__":
    res = run_phase3_experiment()
    print("\nPhase 3 Experiments completed successfully.", flush=True)
