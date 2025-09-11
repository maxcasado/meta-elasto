#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = "Skin"
RES_DIR = os.path.join(BASE_DIR, "Resultados")
LIST_PATH = os.path.join(BASE_DIR, "0list.xlsx")

def read_patients_scores(path):
    # lit les 2 premières colonnes, qu'il y ait un header ou pas
    df = pd.read_excel(path, header=None, usecols=[0, 1], names=["patient", "score"])
    # convertit en numérique; l'en-tête "Patient" deviendra NaN
    df["patient"] = pd.to_numeric(df["patient"], errors="coerce")
    df["score"]   = pd.to_numeric(df["score"],   errors="coerce")
    # enlève les lignes sans patient (en-tête, lignes vides, etc.)
    df = df.dropna(subset=["patient"])
    df["patient"] = df["patient"].astype(int)
    return df


def pid4(n):
    return f"{int(n):04d}"

def load_measure(pid, kind):
    fname = f"Elastome_{pid4(pid)}_{kind}_angle_1.csv"
    fpath = os.path.join(RES_DIR, fname)
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    # index temps
    df = df.rename_axis("t").reset_index()
    # numerique + NaN pour tout le reste
    for c in df.columns:
        if c != "t":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    return df


def common_ordered_columns(df_h, df_p, min_cov=0.3):
    # colonnes communes, hors 't'
    cols_h = [c for c in df_h.columns if c != "t"]
    cols_p = [c for c in df_p.columns if c != "t"]
    cols = [c for c in cols_h if c in cols_p]

    def coverage(df, c):
        v = df[c].values
        return np.isfinite(v).sum() / len(v)  # proportion de valeurs finies

    scored = []
    kept = []
    for c in cols:
        r = 0.5*coverage(df_h, c) + 0.5*coverage(df_p, c)
        scored.append((c, r))
        if coverage(df_h, c) >= min_cov and coverage(df_p, c) >= min_cov:
            kept.append(c)

    # trie pour tracer d'abord les plus “propres”
    kept_sorted = sorted(kept, key=lambda k: -dict(scored)[k])
    return kept_sorted

def select_cols_for_lines(cols, k=6):
    return cols[:k]

def plot_patient(pid, scores_df, k_lines=6):
    df_h = load_measure(pid, "Healthy")
    df_p = load_measure(pid, "Patho")
    if df_h is None or df_p is None:
        print(f"Mesures manquantes pour patient {pid4(pid)}")
        return
    cols = common_ordered_columns(df_h, df_p)
    if not cols:
        print(f"Aucune colonne commune exploitable pour patient {pid4(pid)}")
        return
    cols_lines = select_cols_for_lines(cols, k_lines)
    fig = plt.figure(figsize=(14,9))
    gs = fig.add_gridspec(3, 2, height_ratios=[2,1.5,1.5], hspace=0.35, wspace=0.15)
    ax_lines = fig.add_subplot(gs[0,:])
    for c in cols_lines:
        y_h = df_h[c].astype(float)
        y_p = df_p[c].astype(float)
        t = df_h["t"].values
        ax_lines.plot(t, y_h, linewidth=1, alpha=0.9, label=f"{c} Healthy")
        ax_lines.plot(t, y_p, linewidth=1, linestyle="--", alpha=0.9, label=f"{c} Patho")
    sc = scores_df.loc[scores_df["patient"]==pid, "score"]
    score = sc.values[0] if len(sc) else np.nan
    ax_lines.set_title(f"Patient {pid4(pid)} – Score={score}")
    ax_lines.set_xlabel("Temps (index)")
    ax_lines.set_ylabel("Amplitude")
    ax_lines.legend(ncols=3, fontsize=8)
    ax_h = fig.add_subplot(gs[1,0])
    ax_p = fig.add_subplot(gs[1,1])
    def heatmap(ax, df, title):
        M = df[cols].astype(float).values
        mask = np.isnan(M)
        M = np.ma.array(M, mask=mask)
        im = ax.imshow(M, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Fréquences")
        ax.set_ylabel("Temps")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([c.replace("Hz","") for c in cols], rotation=90, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    heatmap(ax_h, df_h, "Healthy heatmap")
    heatmap(ax_p, df_p, "Patho heatmap")
    ax_delta = fig.add_subplot(gs[2,:])
    means_h = pd.Series({c: df_h[c].astype(float).mean(skipna=True) for c in cols})
    means_p = pd.Series({c: df_p[c].astype(float).mean(skipna=True) for c in cols})
    delta = (means_p - means_h)
    ax_delta.bar(range(len(cols)), delta.values)
    ax_delta.set_xticks(range(len(cols)))
    ax_delta.set_xticklabels([c.replace("Hz","") for c in cols], rotation=90, fontsize=8)
    ax_delta.set_title("Moyenne(Patho) - Moyenne(Healthy) par fréquence")
    ax_delta.set_xlabel("Fréquences")
    ax_delta.set_ylabel("Delta")
    plt.tight_layout()
    plt.show()

def build_features_for_patient(pid):
    df_h = load_measure(pid, "Healthy")
    df_p = load_measure(pid, "Patho")
    if df_h is None or df_p is None:
        return None
    cols = common_ordered_columns(df_h, df_p)
    if not cols:
        return None
    feats = {}    
    for c in cols:
        xh = df_h[c].astype(float).values
        xp = df_p[c].astype(float).values
        # on ne garde que les valeurs finies
        xh = xh[np.isfinite(xh)]
        xp = xp[np.isfinite(xp)]
        # si l'une des deux est vide, on skip cette fréquence
        if len(xh) == 0 or len(xp) == 0:
            continue
        mh, sh = np.mean(xh), np.std(xh)
        mp, sp = np.mean(xp), np.std(xp)
        feats[f"{c}__H_mean"] = mh
        feats[f"{c}__H_std"]  = sh
        feats[f"{c}__P_mean"] = mp
        feats[f"{c}__P_std"]  = sp
        feats[f"{c}__Delta_mean"] = mp - mh
        feats[f"{c}__Delta_std"]  = sp - sh
    return pd.Series(feats)

def build_dataset(scores_df):
    rows, y, ids = [], [], []

    for pid in scores_df["patient"].tolist():
        s = build_features_for_patient(pid)
        if s is None or s.empty:
            continue
        rows.append(s)
        # score du patient
        score_val = scores_df.loc[scores_df["patient"] == pid, "score"]
        y.append(pd.to_numeric(score_val.iloc[0], errors="coerce") if len(score_val) else np.nan)
        ids.append(pid)

    if not rows:
        return None, None, None

    X = pd.DataFrame(rows)
    y = pd.Series(y, name="score")
    #replace ±inf by NaN
    X = X.replace({np.inf: np.nan, -np.inf: np.nan})
    # drop NaN columns
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        return None, None, None
    #numerical cast
    X = X.apply(pd.to_numeric, errors="coerce")
    # imputation by medians
    X = X.fillna(X.median(numeric_only=True))

    return X, y, ids

def plot_pca_overview(scores_df):
    X, y, ids = build_dataset(scores_df)
    if X is None:
        print("Aucune donnée exploitable pour la PCA")
        return
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(Z[:,0], Z[:,1], c=y.values, cmap="viridis")
    for i, pid in enumerate(ids):
        ax.annotate(pid4(pid), (Z[i,0], Z[i,1]), fontsize=8, alpha=0.8)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Score")
    ax.set_title("PCA 2D des patients (features agrégés Healthy/Patho)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=int, default=None)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--k_lines", type=int, default=6)
    args = parser.parse_args()
    scores_df = read_patients_scores(LIST_PATH)
    if args.patient is not None:
        plot_patient(args.patient, scores_df, k_lines=args.k_lines)
    if args.pca:
        plot_pca_overview(scores_df)
    if args.patient is None and not args.pca:
        if len(scores_df):
            plot_patient(int(scores_df.iloc[0]["patient"]), scores_df, k_lines=args.k_lines)

if __name__ == "__main__":
    main()
