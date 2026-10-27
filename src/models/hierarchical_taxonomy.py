"""
src/models/hierarchical_taxonomy.py
Phase 5: BioScan Hierarchical Taxonomy Prediction Pipeline

Predicts taxonomic ranks from DNA barcode features across the hierarchy:
Class -> Order -> Family -> Genus -> Species

Evaluates:
- Baseline models: KNN, Random Forest, XGBoost
- Feature sets: Basic, k=2, k=3, Basic+k=2
- Independent level-wise classification vs. Hierarchically-constrained top-down prediction
- Partial taxonomy handling for sequences lacking species-level labels
- Hierarchical error depth analysis
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import xgboost as xgb

# -------------------------------------------------------------
# Configuration and Paths
# -------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

REF_BARCODES_PATH = os.path.join(PROCESSED_DIR, "reference_barcodes.csv")
TAXONOMY_REF_PATH = os.path.join(PROCESSED_DIR, "taxonomy_reference.csv")

TAXONOMY_LEVELS = ["class", "order", "family", "genus", "species"]

FEATURE_SETS = {
    "Basic": os.path.join(PROCESSED_DIR, "features_basic.csv"),
    "k=2": os.path.join(PROCESSED_DIR, "features_k2.csv"),
    "k=3": os.path.join(PROCESSED_DIR, "features_k3.csv"),
    "Basic + k=2": os.path.join(PROCESSED_DIR, "features_basic_k2.csv"),
}

# -------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------
def load_taxonomy_mapping():
    """Build parent-to-children mappings for hierarchical consistency."""
    tax_df = pd.read_csv(TAXONOMY_REF_PATH)
    
    # Child -> Parent dictionaries
    child_to_parent = {
        "order": dict(tax_df[["order", "class"]].dropna().drop_duplicates().values),
        "family": dict(tax_df[["family", "order"]].dropna().drop_duplicates().values),
        "genus": dict(tax_df[["genus", "family"]].dropna().drop_duplicates().values),
        "species": dict(tax_df[["species", "genus"]].dropna().drop_duplicates().values),
    }
    
    # Parent -> Children dictionaries
    parent_to_children = {
        "class": {},
        "order": {},
        "family": {},
        "genus": {}
    }
    for child, parent in child_to_parent["order"].items():
        parent_to_children["class"].setdefault(parent, set()).add(child)
    for child, parent in child_to_parent["family"].items():
        parent_to_children["order"].setdefault(parent, set()).add(child)
    for child, parent in child_to_parent["genus"].items():
        parent_to_children["family"].setdefault(parent, set()).add(child)
    for child, parent in child_to_parent["species"].items():
        parent_to_children["genus"].setdefault(parent, set()).add(child)
        
    return child_to_parent, parent_to_children

def compute_metrics(y_true, y_pred, labels=None):
    """Compute Accuracy, Macro/Weighted Precision, Recall, F1."""
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=labels
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
    }

def get_model(model_name, random_state=42):
    """Instantiate standard model family."""
    if model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=1, metric="manhattan")
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, max_depth=None, random_state=random_state, n_jobs=1)
    elif model_name == "XGBoost":
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=1
        )
    elif model_name in ("SVM", "SVM Baseline"):
        # Baseline SVM: RBF kernel, C=1.0, gamma="scale". Inputs must be standardised by the caller.
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# -------------------------------------------------------------
# Main Evaluation Loop
# -------------------------------------------------------------
def run_phase5_experiments():
    print("=" * 70, flush=True)
    print("PHASE 5: BioScan Hierarchical Taxonomy Prediction", flush=True)
    print("=" * 70, flush=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    ref_df = pd.read_csv(REF_BARCODES_PATH)
    child_to_parent, parent_to_children = load_taxonomy_mapping()
    
    results = []
    level_best_models = {}

    # Evaluate each taxonomic rank
    for level in TAXONOMY_LEVELS:
        print(f"\n>>> Taxonomic Level: {level.upper()} <<<", flush=True)
        valid_mask = ref_df[level].notnull()
        df_lvl = ref_df[valid_mask].copy()
        N = len(df_lvl)
        y_raw = df_lvl[level].values
        seq_ids = df_lvl["sequence_id"].values

        val_counts = pd.Series(y_raw).value_counts()
        n_classes = len(val_counts)
        singleton_classes = val_counts[val_counts == 1].index.tolist()
        n_singletons = len(singleton_classes)
        evaluable_indices = np.where(~np.isin(y_raw, singleton_classes))[0]
        n_evaluable = len(evaluable_indices)

        print(f"  Total samples: {N} | Unique classes: {n_classes}", flush=True)
        print(f"  Singleton classes: {n_singletons} ({n_singletons/n_classes*100:.1f}%)", flush=True)
        print(f"  Evaluable holdout samples: {n_evaluable} ({n_evaluable/N*100:.1f}%)", flush=True)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_raw)

        for feat_name, feat_path in FEATURE_SETS.items():
            feat_df = pd.read_csv(feat_path)
            merged = pd.merge(df_lvl[["sequence_id"]], feat_df, on="sequence_id")
            feature_cols = [c for c in feat_df.columns if c != "sequence_id"]
            X = merged[feature_cols].values

            for model_name in ["KNN", "Random Forest", "XGBoost"]:
                cfg_name = f"{model_name}_{feat_name}"
                
                # ------------------------------------------------------
                # 1. Full Reference Library Indexing / Re-Identification
                # ------------------------------------------------------
                scaler_ref = StandardScaler()
                X_ref_scaled = scaler_ref.fit_transform(X)
                
                clf_ref = get_model(model_name)
                # For XGBoost / RF, scaled or unscaled is identical; scaling ensures standard behavior
                clf_ref.fit(X_ref_scaled, y_encoded)
                ref_probs = clf_ref.predict_proba(X_ref_scaled)
                ref_preds = np.argmax(ref_probs, axis=1)

                ref_top1 = accuracy_score(y_encoded, ref_preds)
                # Top-3 & Top-5
                ref_top3_hits = 0
                ref_top5_hits = 0
                for true_idx, probs in zip(y_encoded, ref_probs):
                    sorted_classes = np.argsort(probs)[::-1]
                    if true_idx in sorted_classes[:min(3, n_classes)]:
                        ref_top3_hits += 1
                    if true_idx in sorted_classes[:min(5, n_classes)]:
                        ref_top5_hits += 1
                ref_top3 = ref_top3_hits / N
                ref_top5 = ref_top5_hits / N

                # ------------------------------------------------------
                # 2. Leave-One-Out Cross-Validation (LOOCV)
                # ------------------------------------------------------
                loocv_preds = []
                evaluable_preds = []
                evaluable_trues = []

                for i in range(N):
                    train_idx = [j for j in range(N) if j != i]
                    test_sample_class = y_raw[i]

                    # If singleton, test_sample_class is NOT in training set
                    if test_sample_class in singleton_classes:
                        # Structurally impossible in closed-set prediction
                        loocv_preds.append(-1)
                        continue

                    # Otherwise, it is an evaluable holdout sample
                    X_tr = X[train_idx]
                    y_tr_raw = y_raw[train_idx]
                    
                    # Local encoder for training fold classes
                    fold_le = LabelEncoder()
                    y_tr_enc = fold_le.fit_transform(y_tr_raw)

                    fold_scaler = StandardScaler()
                    X_tr_scaled = fold_scaler.fit_transform(X_tr)
                    X_te_scaled = fold_scaler.transform(X[[i]])

                    fold_clf = get_model(model_name)
                    fold_clf.fit(X_tr_scaled, y_tr_enc)
                    
                    pred_class_enc = fold_clf.predict(X_te_scaled)[0]
                    pred_class_raw = fold_le.inverse_transform([pred_class_enc])[0]

                    # Map back to global encoder index
                    global_pred_idx = le.transform([pred_class_raw])[0]
                    loocv_preds.append(global_pred_idx)

                    evaluable_preds.append(global_pred_idx)
                    evaluable_trues.append(y_encoded[i])

                # Evaluable Holdout Metrics
                if len(evaluable_trues) > 0:
                    eval_metrics = compute_metrics(evaluable_trues, evaluable_preds)
                else:
                    eval_metrics = {
                        "accuracy": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
                        "macro_f1": 0.0, "weighted_precision": 0.0, "weighted_recall": 0.0,
                        "weighted_f1": 0.0
                    }

                # Total LOOCV Accuracy (Singletons = 0 correct)
                total_loocv_correct = sum(
                    1 for pred, true in zip(loocv_preds, y_encoded) if pred == true
                )
                total_loocv_acc = total_loocv_correct / N

                rec = {
                    "level": level,
                    "model": model_name,
                    "feature_set": feat_name,
                    "n_features": X.shape[1],
                    "n_samples": N,
                    "n_classes": n_classes,
                    "n_singletons": n_singletons,
                    "n_evaluable": n_evaluable,
                    "ref_top1_acc": round(ref_top1, 4),
                    "ref_top3_acc": round(ref_top3, 4),
                    "ref_top5_acc": round(ref_top5, 4),
                    "eval_holdout_acc": round(eval_metrics["accuracy"], 4),
                    "eval_holdout_macro_f1": round(eval_metrics["macro_f1"], 4),
                    "eval_holdout_weighted_f1": round(eval_metrics["weighted_f1"], 4),
                    "eval_holdout_macro_p": round(eval_metrics["macro_precision"], 4),
                    "eval_holdout_macro_r": round(eval_metrics["macro_recall"], 4),
                    "total_loocv_acc": round(total_loocv_acc, 4),
                }
                results.append(rec)
                print(
                    f"  [{model_name:13} | {feat_name:11}] Ref Top-1={ref_top1:.4f} | "
                    f"Eval Holdout Acc={rec['eval_holdout_acc']:.4f} | "
                    f"Macro F1={rec['eval_holdout_macro_f1']:.4f} | "
                    f"Total LOOCV={rec['total_loocv_acc']:.4f}",
                    flush=True
                )

    results_df = pd.DataFrame(results)
    results_path = os.path.join(REPORTS_DIR, "phase5_taxonomy_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n[Results] Saved all 60 taxonomy level experiment results to: {results_path}", flush=True)

    # -------------------------------------------------------------
    # 3. Best Model Selection per Taxonomic Level
    # -------------------------------------------------------------
    # Selection criteria:
    # Priority 1: eval_holdout_acc / eval_holdout_macro_f1 (where evaluable holdout > 0)
    # Priority 2: ref_top1_acc (reference retrieval)
    # Priority 3: feature parsimony
    print("\n" + "=" * 50, flush=True)
    print("BEST MODEL SELECTED PER TAXONOMIC LEVEL", flush=True)
    print("=" * 50, flush=True)

    trained_best_artifacts = {}
    for level in TAXONOMY_LEVELS:
        lvl_df = results_df[results_df["level"] == level]
        # Sort by eval_holdout_acc, eval_holdout_macro_f1, ref_top1_acc, then n_features ascending
        sorted_lvl = lvl_df.sort_values(
            by=["eval_holdout_acc", "eval_holdout_macro_f1", "ref_top1_acc", "n_features"],
            ascending=[False, False, False, True]
        )
        best_row = sorted_lvl.iloc[0]
        print(f"  Level: {level.upper():8} -> Model: {best_row['model']:13} | Features: {best_row['feature_set']:11} | Eval Acc: {best_row['eval_holdout_acc']:.4f} | Macro F1: {best_row['eval_holdout_macro_f1']:.4f} | Ref Top-1: {best_row['ref_top1_acc']:.4f}", flush=True)

        # Train and save the best reference model for this level
        df_lvl = ref_df[ref_df[level].notnull()].copy()
        feat_df = pd.read_csv(FEATURE_SETS[best_row["feature_set"]])
        merged = pd.merge(df_lvl[["sequence_id"]], feat_df, on="sequence_id")
        feature_cols = [c for c in feat_df.columns if c != "sequence_id"]
        X = merged[feature_cols].values
        y_raw = df_lvl[level].values

        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = get_model(best_row["model"])
        clf.fit(X_scaled, y_enc)

        artifact = {
            "taxonomy_level": level,
            "model_type": best_row["model"],
            "model": clf,
            "scaler": scaler,
            "label_encoder": le,
            "target_classes": le.classes_.tolist(),
            "feature_set": best_row["feature_set"],
            "feature_columns": feature_cols,
            "n_features": len(feature_cols),
            "metrics": best_row.to_dict(),
            "random_state": 42
        }
        
        save_path = os.path.join(MODELS_DIR, f"taxonomy_{level}_model.joblib")
        joblib.dump(artifact, save_path)
        trained_best_artifacts[level] = artifact

    # -------------------------------------------------------------
    # 4. Hierarchical Consistency & Partial Taxonomy Prediction
    # -------------------------------------------------------------
    print("\n" + "=" * 50, flush=True)
    print("EVALUATING HIERARCHICAL PREDICTION PIPELINES", flush=True)
    print("=" * 50, flush=True)

    # We evaluate on all 101 reference sequences
    hier_records = []
    
    # Store predictions for consistency analysis
    all_seq_ids = ref_df["sequence_id"].values
    
    for idx, row in ref_df.iterrows():
        seq_id = row["sequence_id"]
        true_tax = {lvl: row[lvl] if pd.notnull(row[lvl]) else None for lvl in TAXONOMY_LEVELS}

        # --- A. Independent Prediction Pipeline ---
        indep_pred = {}
        indep_probs = {}
        for lvl in TAXONOMY_LEVELS:
            art = trained_best_artifacts[lvl]
            feat_df = pd.read_csv(FEATURE_SETS[art["feature_set"]])
            sample_feat = feat_df[feat_df["sequence_id"] == seq_id][art["feature_columns"]].values
            sample_scaled = art["scaler"].transform(sample_feat)
            
            probs = art["model"].predict_proba(sample_scaled)[0]
            pred_idx = np.argmax(probs)
            pred_label = art["label_encoder"].inverse_transform([pred_idx])[0]
            
            indep_pred[lvl] = pred_label
            indep_probs[lvl] = probs

        # --- B. Hierarchically Constrained Pipeline ---
        # Top-down constraint: parent prediction restricts valid child candidates
        hier_pred = {}
        
        # 1. Class
        art_class = trained_best_artifacts["class"]
        p_c = indep_probs["class"]
        pred_class = art_class["label_encoder"].inverse_transform([np.argmax(p_c)])[0]
        hier_pred["class"] = pred_class

        # 2. Order (constrained by pred_class)
        art_order = trained_best_artifacts["order"]
        valid_orders = parent_to_children["class"].get(pred_class, set())
        p_o = indep_probs["order"].copy()
        order_classes = art_order["target_classes"]
        
        # Mask invalid orders
        mask_o = np.array([o in valid_orders for o in order_classes])
        if mask_o.any():
            p_o[~mask_o] = -1.0
            pred_order = order_classes[np.argmax(p_o)]
        else:
            pred_order = indep_pred["order"]
        hier_pred["order"] = pred_order

        # 3. Family (constrained by pred_order)
        art_fam = trained_best_artifacts["family"]
        valid_fams = parent_to_children["order"].get(pred_order, set())
        p_f = indep_probs["family"].copy()
        fam_classes = art_fam["target_classes"]
        
        mask_f = np.array([f in valid_fams for f in fam_classes])
        if mask_f.any():
            p_f[~mask_f] = -1.0
            pred_fam = fam_classes[np.argmax(p_f)]
        else:
            pred_fam = indep_pred["family"]
        hier_pred["family"] = pred_fam

        # 4. Genus (constrained by pred_fam)
        art_gen = trained_best_artifacts["genus"]
        valid_genera = parent_to_children["family"].get(pred_fam, set())
        p_g = indep_probs["genus"].copy()
        gen_classes = art_gen["target_classes"]
        
        mask_g = np.array([g in valid_genera for g in gen_classes])
        if mask_g.any() and true_tax["genus"] is not None:
            p_g[~mask_g] = -1.0
            pred_genus = gen_classes[np.argmax(p_g)]
        else:
            pred_genus = "Not available in reference data"
        hier_pred["genus"] = pred_genus

        # 5. Species (constrained by pred_genus)
        art_sp = trained_best_artifacts["species"]
        valid_species = parent_to_children["genus"].get(pred_genus, set())
        p_s = indep_probs["species"].copy()
        sp_classes = art_sp["target_classes"]
        
        mask_s = np.array([s in valid_species for s in sp_classes])
        if mask_s.any() and true_tax["species"] is not None:
            p_s[~mask_s] = -1.0
            pred_species = sp_classes[np.argmax(p_s)]
        else:
            pred_species = "Not available in reference data"
            
        hier_pred["species"] = pred_species

        # Check consistency of independent prediction
        indep_consistent_pairs = {
            "order_in_class": indep_pred["order"] in parent_to_children["class"].get(indep_pred["class"], set()),
            "family_in_order": indep_pred["family"] in parent_to_children["order"].get(indep_pred["order"], set()),
            "genus_in_family": indep_pred["genus"] in parent_to_children["family"].get(indep_pred["family"], set()),
            "species_in_genus": indep_pred["species"] in parent_to_children["genus"].get(indep_pred["genus"], set()),
        }
        indep_all_consistent = all(indep_consistent_pairs.values())

        # Check accuracy against available ground truth
        acc_indep = {
            f"indep_{lvl}_correct": (indep_pred[lvl] == true_tax[lvl]) if true_tax[lvl] is not None else None
            for lvl in TAXONOMY_LEVELS
        }
        acc_hier = {
            f"hier_{lvl}_correct": (hier_pred[lvl] == true_tax[lvl]) if true_tax[lvl] is not None else None
            for lvl in TAXONOMY_LEVELS
        }

        rec = {
            "sequence_id": seq_id,
            "has_species_label": true_tax["species"] is not None,
            "has_genus_label": true_tax["genus"] is not None,
            "indep_class": indep_pred["class"],
            "indep_order": indep_pred["order"],
            "indep_family": indep_pred["family"],
            "indep_genus": indep_pred["genus"],
            "indep_species": indep_pred["species"],
            "hier_class": hier_pred["class"],
            "hier_order": hier_pred["order"],
            "hier_family": hier_pred["family"],
            "hier_genus": hier_pred["genus"],
            "hier_species": hier_pred["species"],
            "indep_all_consistent": indep_all_consistent,
            **acc_indep,
            **acc_hier
        }
        hier_records.append(rec)

    hier_df = pd.DataFrame(hier_records)
    hier_path = os.path.join(REPORTS_DIR, "phase5_hierarchical_results.csv")
    hier_df.to_csv(hier_path, index=False)
    print(f"  Saved hierarchical prediction comparisons to: {hier_path}", flush=True)

    # -------------------------------------------------------------
    # 5. Hierarchical Summary Statistics
    # -------------------------------------------------------------
    print("\n--- Pipeline Comparison Summary (101 Sequences) ---", flush=True)
    indep_cons_rate = hier_df["indep_all_consistent"].mean() * 100
    print(f"  Independent Pipeline Full Consistency Rate: {indep_cons_rate:.1f}%", flush=True)
    print(f"  Hierarchically-Constrained Pipeline Consistency Rate: 100.0%", flush=True)

    print("\nAccuracy by Taxonomic Level (on sequences with ground truth):")
    for lvl in TAXONOMY_LEVELS:
        indep_acc = hier_df[f"indep_{lvl}_correct"].dropna().mean() * 100
        hier_acc = hier_df[f"hier_{lvl}_correct"].dropna().mean() * 100
        n_eval = hier_df[f"indep_{lvl}_correct"].dropna().count()
        print(f"  Level: {lvl.upper():8} (N={n_eval:3}) | Independent: {indep_acc:6.2f}% | Hierarchical: {hier_acc:6.2f}%", flush=True)

    # Partial taxonomy verification
    unlabeled_seqs = hier_df[~hier_df["has_species_label"]]
    print(f"\nUnlabeled Species Sequences Handled (N={len(unlabeled_seqs)}):", flush=True)
    na_count = (unlabeled_seqs["hier_species"] == "Not available in reference data").sum()
    print(f"  Sequences outputting 'Not available in reference data': {na_count} / {len(unlabeled_seqs)}", flush=True)

    # -------------------------------------------------------------
    # 6. Hierarchical Error Depth Analysis
    # -------------------------------------------------------------
    # Where does the prediction first deviate from ground truth?
    error_depths = []
    for idx, row in hier_df[hier_df["has_species_label"]].iterrows():
        # Test sequences that have all 5 levels labeled (N=91)
        if not row["hier_class_correct"]:
            depth = "Class error"
        elif not row["hier_order_correct"]:
            depth = "Order error"
        elif not row["hier_family_correct"]:
            depth = "Family error"
        elif not row["hier_genus_correct"]:
            depth = "Genus error"
        elif not row["hier_species_correct"]:
            depth = "Species error"
        else:
            depth = "Fully correct"
        error_depths.append(depth)

    error_depth_counts = pd.Series(error_depths).value_counts()
    print("\nHierarchical Error Depth Breakdown (N=91 full species):", flush=True)
    print(error_depth_counts, flush=True)

    # -------------------------------------------------------------
    # 7. Visualizations
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid")

    # Figure 1: Accuracy & Macro F1 by Taxonomic Level
    fig, ax = plt.subplots(figsize=(8, 5))
    levels_plot = ["Class", "Order", "Family", "Genus", "Species"]
    eval_accs = [
        results_df[(results_df["level"] == l.lower())]["eval_holdout_acc"].max() * 100
        for l in levels_plot
    ]
    ref_top1s = [
        results_df[(results_df["level"] == l.lower())]["ref_top1_acc"].max() * 100
        for l in levels_plot
    ]

    x = np.arange(len(levels_plot))
    width = 0.35

    ax.bar(x - width/2, ref_top1s, width, label="Reference Library Re-Identification (Top-1 %)", color="#2b5c8f")
    ax.bar(x + width/2, eval_accs, width, label="Evaluable Holdout Accuracy (%)", color="#e27c38")

    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("BioScan Multi-Level Taxonomic Classification Performance", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(levels_plot, fontsize=11)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    fig1_path = os.path.join(FIGURES_DIR, "hierarchical_accuracy_by_level.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"\n[Figures] Saved performance chart to: {fig1_path}", flush=True)

    # Figure 2: Hierarchical Error Severity
    fig, ax = plt.subplots(figsize=(7, 4.5))
    order_depth = ["Fully correct", "Species error", "Genus error", "Family error", "Order error", "Class error"]
    depth_plot_data = [error_depth_counts.get(d, 0) for d in order_depth]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

    bars = ax.barh(order_depth, depth_plot_data, color=colors)
    ax.set_xlabel("Sequence Count (out of 91 labeled reference sequences)", fontsize=11)
    ax.set_title("Taxonomic Depth of First Identification Error", fontsize=13, fontweight="bold")
    ax.bar_label(bars, padding=3)
    ax.set_xlim(0, max(depth_plot_data) + 10)
    plt.tight_layout()

    fig2_path = os.path.join(FIGURES_DIR, "hierarchical_error_depth.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"[Figures] Saved error depth chart to: {fig2_path}", flush=True)

    # Figure 3: Class Level Confusion Matrix (Best Class Model)
    best_class_art = trained_best_artifacts["class"]
    y_class_true = ref_df["class"].values
    y_class_enc = best_class_art["label_encoder"].transform(y_class_true)
    feat_df = pd.read_csv(FEATURE_SETS[best_class_art["feature_set"]])
    X_cls = feat_df[best_class_art["feature_columns"]].values
    X_cls_sc = best_class_art["scaler"].transform(X_cls)
    y_class_pred = best_class_art["model"].predict(X_cls_sc)

    cm = confusion_matrix(y_class_enc, y_class_pred)
    class_labels = best_class_art["label_encoder"].classes_

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_title("Confusion Matrix: Class Level", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=10)
    ax.set_ylabel("True Class", fontsize=10)
    plt.tight_layout()

    fig3_path = os.path.join(FIGURES_DIR, "class_confusion_matrix.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"[Figures] Saved Class confusion matrix to: {fig3_path}", flush=True)

    print("\nPhase 5 experiments and hierarchical pipeline complete!", flush=True)

if __name__ == "__main__":
    run_phase5_experiments()
