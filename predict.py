#!/usr/bin/env python3
# predict.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor

import argparse

# on réutilise ton extraction de features
from analysis import build_dataset, read_patients_scores, LIST_PATH


def make_model(name: str, args, n_features: int) -> Pipeline:
    """
    Fabrique un pipeline selon le modèle demandé.
    - Linear models: StandardScaler (+ SelectKBest/PCA si demandé)
    - RandomForest: pas de scaler (inutile), SelectKBest/PCA optionnels
    """
    steps = []
    name = name.lower()

    # Sélection de features (optionnelle)
    if args.k_best and args.k_best > 0:
        k = min(args.k_best, n_features)
        steps.append(("select", SelectKBest(score_func=f_regression, k=k)))

    # PCA (optionnel)
    if args.pca_components and args.pca_components > 0:
        # Pour les modèles linéaires on normalise avant PCA
        if name in ("ridge", "lasso", "elastic", "pls"):
            steps = [("scaler", StandardScaler())] + steps + [("pca", PCA(n_components=args.pca_components, random_state=0))]
        else:
            # PCA avant RandomForest (peu courant, mais tu peux l'activer si tu veux réduire la dimension)
            steps = steps + [("pca", PCA(n_components=args.pca_components, random_state=0))]
    else:
        # Pas de PCA : on scale seulement les linéaires
        if name in ("ridge", "lasso", "elastic"):
            steps = [("scaler", StandardScaler())] + steps

    # Modèle
    if name == "ridge":
        model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    elif name == "lasso":
        model = LassoCV(cv=5, random_state=0, n_alphas=100, max_iter=10000)
    elif name == "elastic":
        model = ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8, 0.95], cv=5, max_iter=20000, random_state=0)
    elif name == "pls":
        # PLS gère bien n << p ; standardisation faite plus haut si PCA pas demandé
        # n_components borné par min(n_samples-1, n_features). On sécurise :
        n_comp = min(args.pls_components, max(1, n_features))
        model = PLSRegression(n_components=n_comp, scale=(args.pca_components == 0))
    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=args.rf_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            random_state=0,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Modèle inconnu: {name}")

    steps.append(("model", model))
    return Pipeline(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ridge",
                        choices=["ridge", "lasso", "elastic", "pls", "rf"],
                        help="Choix du modèle")
    parser.add_argument("--ignore", type=int, nargs="*", default=[],
                        help="Liste d'IDs patients à ignorer (ex: --ignore 7 12)")
    parser.add_argument("--k-best", type=int, default=0,
                        help="Sélectionne les k meilleures features (0 = désactivé)")
    parser.add_argument("--pca-components", type=int, default=0,
                        help="Nombre de composantes PCA (0 = désactivé)")
    parser.add_argument("--pls-components", type=int, default=2,
                        help="Composantes PLS (si --model pls)")
    parser.add_argument("--rf-estimators", type=int, default=300,
                        help="RandomForest: n_estimators")
    parser.add_argument("--rf-max-depth", type=int, default=None,
                        help="RandomForest: max_depth (None = illimité)")
    parser.add_argument("--rf-min-samples-leaf", type=int, default=1,
                        help="RandomForest: min_samples_leaf")

    parser.add_argument("--clip-min", type=float, default=None,
                        help="Borne basse pour les prédictions (ex: 0)")
    parser.add_argument("--clip-max", type=float, default=None,
                        help="Borne haute pour les prédictions (ex: 5)")
    parser.add_argument("--annotate", action="store_true",
                        help="Annoter les points sur le scatter")
    args = parser.parse_args()

    # charger données
    scores_df = read_patients_scores(LIST_PATH)
    X, y, ids = build_dataset(scores_df)
    if X is None or len(ids) < 3:
        print("Pas assez de données pour entraîner (min 3 patients).")
        return

    # convertir '0007' -> 7, '7' -> 7
    ignore_set = set(int(s) for s in args.ignore)  # gère str numériques

    if ignore_set:
        mask = [pid not in ignore_set for pid in ids]
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        ids = [pid for pid in ids if pid not in ignore_set]
        print(f"Ignored patients: {sorted(ignore_set)}")
        print(f"X shape after ignore: {X.shape} | #patients: {len(ids)}")

    if len(ids) < 3:
        print("Trop de patients exclus. Minimum 3 requis.")
        return


    print(f"X shape: {X.shape} | #patients: {len(ids)} | modèle: {args.model}")

    # pipeline
    model = make_model(args.model, args, n_features=X.shape[1])

    # LOOCV
    loo = LeaveOneOut()
    y_true, y_pred, ids_tested = [], [], []
    y_base = []  # baseline: moyenne du train

    for train_idx, test_idx in loo.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(Xtr, ytr)

        pred = model.predict(Xte)[0]
        # bornage des prédictions (optionnel)
        if args.clip_min is not None:
            pred = max(args.clip_min, pred)
        if args.clip_max is not None:
            pred = min(args.clip_max, pred)

        y_true.append(yte.values[0])
        y_pred.append(pred)
        ids_tested.append(ids[test_idx[0]])
        y_base.append(float(np.mean(ytr)))

    # métriques
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_base = np.sqrt(mean_squared_error(y_true, y_base))
    print("Baseline RMSE (moyenne du train):", rmse_base)
    print("Model    R²:", r2)
    print("Model  RMSE:", rmse)

    # tableau de résultats
    res = pd.DataFrame({"patient": ids_tested, "y_true": y_true, "y_pred": y_pred})
    print("\n", res.sort_values("patient").to_string(index=False))

    # scatter réel vs prédit
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.8)
    mn = min(min(y_true), min(y_pred)); mx = max(max(y_true), max(y_pred))
    plt.plot([mn, mx], [mn, mx], "r--")
    if args.annotate:
        for i, pid in enumerate(ids_tested):
            plt.annotate(str(pid), (y_true[i], y_pred[i]),
                         textcoords="offset points", xytext=(5,5), fontsize=8, alpha=0.9)
    plt.xlabel("Score réel")
    plt.ylabel("Score prédit")
    plt.title(f"LOOCV – {args.model.upper()} (R²={r2:.3f}, RMSE={rmse:.3f})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
