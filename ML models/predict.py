#!/usr/bin/env python3
# predict.py — Multiclass classification (Healthy=0), samples = individual files (Healthy or Patho)
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# On réutilise des utilitaires de analysis.py
from analysis import (
    read_patients_scores, LIST_PATH,
    load_measure, pid4
)

# --------------------------
# Feature extraction (single file)
# --------------------------
def extract_features_single(df, min_cov=0.3):
    """
    Extrait des features d'un SEUL fichier (Healthy ou Patho) :
    - pour chaque fréquence avec couverture >= min_cov, calcule mean et std.
    Retourne une Series (ou None si rien d'exploitable).
    """
    if df is None or df.empty:
        return None

    cols = [c for c in df.columns if c != "t"]
    feats = {}

    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").astype(float).values
        cov = np.isfinite(x).mean()
        if cov < min_cov:
            continue
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        feats[f"{c}__mean"] = float(np.mean(x))
        feats[f"{c}__std"]  = float(np.std(x))

    if not feats:
        return None
    return pd.Series(feats)

# --------------------------
# Dataset builder (with logging)
# --------------------------
def build_dataset_individuals_with_log(scores_df, min_cov=0.3):
    """
    Construit un dataset où CHAQUE fichier (Healthy ou Patho) = 1 individu.
      - Label Healthy = 0
      - Label Patho   = score du patient (1..5)
    Retourne: X (DataFrame), y (Series), sids (list[str]), pids (list[int]), logs (list[dict])
    sids: "0007_H" / "0007_P"
    logs: [{pid, kind, reason, extra}, ...]
    """
    rows, labels, sids, pids, logs = [], [], [], [], []

    for pid, score in scores_df[["patient", "score"]].itertuples(index=False):
        # --- HEALTHY ---
        df_h = load_measure(pid, "Healthy")
        if df_h is None:
            logs.append({"pid": int(pid), "kind": "H", "reason": "missing_file"})
        else:
            fs_h = extract_features_single(df_h, min_cov=min_cov)
            if fs_h is None or fs_h.empty:
                logs.append({"pid": int(pid), "kind": "H", "reason": "insufficient_coverage_or_all_nan"})
            else:
                rows.append(fs_h); labels.append(0); sids.append(f"{pid4(pid)}_H"); pids.append(int(pid))

        # --- PATHO ---
        df_p = load_measure(pid, "Patho")
        if df_p is None:
            logs.append({"pid": int(pid), "kind": "P", "reason": "missing_file"})
        else:
            fs_p = extract_features_single(df_p, min_cov=min_cov)
            if fs_p is None or fs_p.empty:
                logs.append({"pid": int(pid), "kind": "P", "reason": "insufficient_coverage_or_all_nan"})
            else:
                y_val = pd.to_numeric(score, errors="coerce")
                if pd.isna(y_val):
                    logs.append({"pid": int(pid), "kind": "P", "reason": "non_numeric_score", "extra": str(score)})
                else:
                    rows.append(fs_p); labels.append(int(y_val)); sids.append(f"{pid4(pid)}_P"); pids.append(int(pid))

    if not rows:
        return None, None, None, None, logs

    X = pd.DataFrame(rows)
    y = pd.Series(labels, name="score")

    # Nettoyage numérique
    X = X.replace({np.inf: np.nan, -np.inf: np.nan})
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        return None, None, None, None, logs
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    return X, y, sids, pids, logs

# --------------------------
# Classifiers
# --------------------------
def make_classifier(name: str, args, n_features: int, pca_n: int = 0, k_best_n: int = 0) -> Pipeline:
    """
    Pipeline classification multi-classes.
    - Models: 'xgb' (XGBoost), 'lgbm' (LightGBM)
    - Optionnels: SelectKBest (f_classif), PCA (avec standardisation)
    """
    steps = []
    name = name.lower()

    if k_best_n and k_best_n > 0:
        k = min(k_best_n, n_features)
        steps.append(("select", SelectKBest(score_func=f_classif, k=k)))

    if pca_n and pca_n > 0:
        steps = [("scaler", StandardScaler())] + steps + [
            ("pca", PCA(n_components=pca_n, random_state=args.seed))
        ]

    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise SystemExit("XGBoost n'est pas installé. Fais: pip install xgboost") from e
        model = XGBClassifier(
            n_estimators=args.xgb_estimators,
            max_depth=args.xgb_max_depth,
            learning_rate=args.xgb_learning_rate,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            min_child_weight=args.xgb_min_child_weight,
            reg_alpha=args.xgb_reg_alpha,
            reg_lambda=args.xgb_reg_lambda,
            objective="multi:softprob",
            tree_method="hist",
            n_jobs=-1,
            random_state=args.seed,
        )
    elif name == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise SystemExit("LightGBM n'est pas installé. Fais: pip install lightgbm") from e
        model = LGBMClassifier(
            n_estimators=args.lgbm_estimators,
            max_depth=args.lgbm_max_depth,
            learning_rate=args.lgbm_learning_rate,
            subsample=args.lgbm_subsample,
            colsample_bytree=args.lgbm_colsample_bytree,
            min_child_samples=args.lgbm_min_child_samples,
            reg_alpha=args.lgbm_reg_alpha,
            reg_lambda=args.lgbm_reg_lambda,
            n_jobs=-1,
            random_state=args.seed,
        )
    else:
        raise ValueError("Modèle inconnu (utilise 'xgb' ou 'lgbm').")

    steps.append(("model", model))
    return Pipeline(steps)

# --------------------------
# ROC
# --------------------------
def plot_multiclass_roc(proba_all, y_true_all, classes_all, *, model_name="", config_label="", save_fig=""):
    from sklearn.metrics import roc_curve, auc

    K = len(classes_all)
    y_bin = label_binarize(y_true_all, classes=list(range(K)))

    color_cycle = ["#00cfff", "#ffa500", "#2e8b57", "#dc143c", "#800080",
                   "#8b4513", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    plt.figure(figsize=(8, 6))
    for c in range(K):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, c], proba_all[:, c])
            auc_c = auc(fpr, tpr)
        except Exception:
            fpr, tpr, auc_c = np.array([0, 1]), np.array([0, 1]), np.nan
        color = color_cycle[c % len(color_cycle)]
        plt.plot(
            fpr, tpr, lw=2, color=color,
            label=f"{model_name} — classe {classes_all[c]} (AUC = {auc_c:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate (Recall)")

    title = "ROC multi-classes"
    if model_name:
        title += f" — {model_name}"
    if config_label:
        title += f" [{config_label}]"
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    if not save_fig:
        os.makedirs("reports", exist_ok=True)
        def slug(s: str) -> str:
            return (s or "base").lower().replace(" ", "").replace("=", "").replace("+", "_").replace(":", "")
        save_fig = f"reports/roc_{slug(model_name)}_{slug(config_label)}.png"

    plt.savefig(save_fig, dpi=150)
    print(f"ROC figure saved to: {save_fig}")

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Multiclass (Healthy=0) — samples = files — Group-LOOCV par patient, ROC + classification report"
    )

    # Modèle
    parser.add_argument("--model", type=str, default="xgb", choices=["xgb", "lgbm"])
    parser.add_argument("--seed", type=int, default=0)

    # Sélection / PCA
    parser.add_argument("--k-best", type=int, default=0, help="SelectKBest (0 = off)")
    parser.add_argument("--pca-components", type=int, default=0, help="PCA components (0 = off)")

    # Filtres / logs
    parser.add_argument("--min-cov", type=float, default=0.3,
                        help="Couverture minimale par fréquence pour garder une feature [0..1]")
    parser.add_argument("--ignore", type=str, nargs="*", default=[],
                        help="IDs patients à ignorer (ex: 7 12 ou 0007 0012)")
    parser.add_argument("--log-skips", type=str, default="", help="Chemin CSV pour enregistrer les échantillons skippés")

    # Sorties
    parser.add_argument("--save-fig", type=str, default="", help="Chemin pour sauvegarder la figure ROC (sinon auto)")
    parser.add_argument("--save-report", type=str, default="", help="Chemin pour sauvegarder le classification report")

    # XGBoost
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=3)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.7)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.7)
    parser.add_argument("--xgb-min-child-weight", type=float, default=2.0)
    parser.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0)

    # LightGBM
    parser.add_argument("--lgbm-estimators", type=int, default=300)
    parser.add_argument("--lgbm-max-depth", type=int, default=-1)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lgbm-subsample", type=float, default=0.8)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=5)
    parser.add_argument("--lgbm-reg-alpha", type=float, default=0.0)
    parser.add_argument("--lgbm-reg-lambda", type=float, default=1.0)

    args = parser.parse_args()

    # Charger données (Healthy=0 ; Patho=score 1..5)
    scores_df = read_patients_scores(LIST_PATH)
    X, y, sids, pids, logs = build_dataset_individuals_with_log(scores_df, min_cov=args.min_cov)
    if args.log_skips:
        os.makedirs(os.path.dirname(args.log_skips) or ".", exist_ok=True)
        pd.DataFrame(logs).to_csv(args.log_skips, index=False)
        print(f"[info] Logs des échantillons skippés -> {args.log_skips}")

    if X is None or len(X) < 2:
        raise SystemExit("Pas assez de données (X None ou < 2 échantillons).")

    # Appliquer --ignore (par patient-id)
    if args.ignore:
        nums = []
        for tok in args.ignore:
            try:
                nums.append(int(str(tok)))
            except Exception:
                print(f"[warn] --ignore: token ignoré: {tok}")
        ignore_set = set(nums)
        if ignore_set:
            mask = [pid not in ignore_set for pid in pids]
            X = X.loc[mask].reset_index(drop=True)
            y = y.loc[mask].reset_index(drop=True)
            sids = [sid for sid, on in zip(sids, mask) if on]
            pids = [pid for pid, on in zip(pids, mask) if on]
            print(f"Ignored patients: {sorted(ignore_set)}")

    print(f"X shape: {X.shape} | #samples: {len(sids)} | #patients: {len(set(pids))}")

    # Vérifier les classes présentes vs attendues {0,1,2,3,4,5}
    present_raw = sorted(set(int(v) for v in y))
    counts_raw  = dict(Counter(int(v) for v in y))
    expected = set([0,1,2,3,4,5])
    missing = sorted(expected - set(present_raw))
    print(f"Classes présentes (labels bruts): {present_raw}")
    print(f"Comptes par classe: {counts_raw}")
    if missing:
        print(f"⚠️ Classe(s) absente(s) du dataset après filtres: {missing}")
        print("   -> relâcher --min-cov, vérifier fichiers, ou ajuster --ignore si besoin.")

    # Encodage
    le = LabelEncoder()
    y_enc = le.fit_transform(y.values)
    classes_all = le.classes_
    K = len(classes_all)
    if K < 2:
        raise SystemExit("Il faut au moins 2 classes pour évaluer.")

    # Group-LOOCV par patient
    logo = LeaveOneGroupOut()
    groups = np.array(pids)

    y_true_all, y_pred_all, proba_all = [], [], []
    k_best_n = min(args.k_best, X.shape[1]) if args.k_best > 0 else 0

    for train_idx, test_idx in logo.split(X, y_enc, groups=groups):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr_global, y_te_global = y_enc[train_idx], y_enc[test_idx]

        # Remap classes du fold -> 0..k_fold-1
        fold_classes = np.unique(y_tr_global)
        global_to_fold = {g: i for i, g in enumerate(fold_classes)}
        fold_to_global = {i: g for i, g in enumerate(fold_classes)}
        y_tr_fold = np.array([global_to_fold[g] for g in y_tr_global], dtype=int)

        # PCA safe
        pca_n = 0
        if args.pca_components and args.pca_components > 0:
            pca_n = min(args.pca_components, X_tr.shape[1], max(1, X_tr.shape[0] - 1))

        pipe = make_classifier(args.model, args, n_features=X.shape[1],
                               pca_n=pca_n, k_best_n=k_best_n)
        pipe.fit(X_tr, y_tr_fold)

        proba_fold = pipe.predict_proba(X_te)  # (n_test, k_fold)
        for i in range(proba_fold.shape[0]):
            vec = np.zeros(K, dtype=float)
            for j in range(proba_fold.shape[1]):
                gcls = fold_to_global[j]
                vec[gcls] = proba_fold[i, j]
            proba_all.append(vec)
            y_true_all.append(int(y_te_global[i]))
            y_pred_all.append(int(vec.argmax()))

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    proba_all = np.vstack(proba_all)

    # Report
    report_txt = classification_report(
        y_true_all, y_pred_all,
        labels=list(range(K)),
        target_names=[str(c) for c in classes_all],
        zero_division=0
    )
    print("\n=== Classification report (Group-LOOCV par patient) ===")
    print(report_txt)
    if args.save_report:
        os.makedirs(os.path.dirname(args.save_report) or ".", exist_ok=True)
        with open(args.save_report, "w", encoding="utf-8") as f:
            f.write(report_txt)
        print(f"[info] Report sauvegardé -> {args.save_report}")

    # ROC par classe
    friendly_model = "XGBoost" if args.model == "xgb" else "LightGBM"
    parts = []
    if args.k_best and args.k_best > 0: parts.append(f"k={k_best_n}")
    if args.pca_components and args.pca_components > 0: parts.append(f"PCA={args.pca_components}")
    config_label = " + ".join(parts) if parts else "base"

    plot_multiclass_roc(
        proba_all, y_true_all, classes_all,
        model_name=friendly_model,
        config_label=config_label,
        save_fig=args.save_fig
    )

if __name__ == "__main__":
    main()
