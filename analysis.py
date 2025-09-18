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
    # enlève les lignes sans patient (en-tête, lignes vides)
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
        print(f"Lacking measures for patient {pid4(pid)}")
        return
    cols = common_ordered_columns(df_h, df_p)
    if not cols:
        print(f"No shared column exploitable for patient {pid4(pid)}")
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
    ax_lines.set_xlabel("Time (index)")
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
        ax.set_xlabel("Frequences")
        ax.set_ylabel("Time")
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
    ax_delta.set_title("Mean (Patho) - Mean (Healthy) by frequency")
    ax_delta.set_xlabel("Frequency")
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
        print("No data availablee for PCA")
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
    ax.set_title("patients 2D PCA (aggregated features Healthy/Patho)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.show()

def _pearsonr_safe(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    n = min(len(x), len(y))
    if n < 3:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def build_long_table(scores_df, min_cov=0.3):
    """Table longue patient×fréquence avec H_mean, P_mean, Delta_mean et couvertures."""
    rows = []
    for pid, score in scores_df[["patient","score"]].itertuples(index=False):
        df_h = load_measure(pid, "Healthy")
        df_p = load_measure(pid, "Patho")
        if df_h is None or df_p is None:
            continue
        cols = [c for c in df_h.columns if c != "t" and c in df_p.columns]
        for c in cols:
            xh = pd.to_numeric(df_h[c], errors="coerce").astype(float).values
            xp = pd.to_numeric(df_p[c], errors="coerce").astype(float).values
            cov_h = np.isfinite(xh).mean()
            cov_p = np.isfinite(xp).mean()
            if cov_h < min_cov or cov_p < min_cov:
                continue
            mh = np.nanmean(xh)
            mp = np.nanmean(xp)
            rows.append({
                "patient": int(pid),
                "pid4": pid4(pid),
                "score": float(score),
                "freq": c,
                "cov_h": cov_h,
                "cov_p": cov_p,
                "H_mean": mh,
                "P_mean": mp,
                "Delta_mean": (mp - mh),
            })
    return pd.DataFrame(rows)

def summarize_by_frequency(long_df):
    """Agrégation par fréquence (couverture, |Δ| moyen, corr(Δ,score))."""
    if long_df.empty:
        return pd.DataFrame()
    base = long_df.groupby("freq").agg(
        n_patients=("patient", "nunique"),
        cov_h_mean=("cov_h", "mean"),
        cov_p_mean=("cov_p", "mean"),
        delta_mean_avg=("Delta_mean", "mean"),
        delta_mean_std=("Delta_mean", "std"),
        delta_abs_mean=("Delta_mean", lambda x: np.mean(np.abs(x))),
    )
    rs = []
    for f, g in long_df.groupby("freq"):
        rs.append((f, _pearsonr_safe(g["Delta_mean"].values, g["score"].values)))
    corr_df = pd.DataFrame(rs, columns=["freq","r_score_delta"]).set_index("freq")
    return base.join(corr_df)

def build_delta_matrix(long_df, order="abs_delta"):
    """Pivot patients × fréquences avec Delta_mean; colonnes ordonnées par pertinence."""
    if long_df.empty:
        return pd.DataFrame()
    mat = long_df.pivot_table(index="patient", columns="freq", values="Delta_mean", aggfunc="mean")
    mat = mat.sort_index()

    # Ordonner les colonnes (fréquences) par pertinence
    summ = summarize_by_frequency(long_df)
    if order == "corr":
        cols = summ.reindex(summ["r_score_delta"].abs().sort_values(ascending=False).index).index.tolist()
    else:  # "abs_delta" par défaut
        cols = summ.sort_values(["delta_abs_mean","n_patients"], ascending=[False, False]).index.tolist()
    mat = mat.reindex(columns=cols)
    return mat

def plot_global_heatmap(delta_mat, scores_df, clip_q=0.98, outpath=None):
    """Heatmap patients×fréquences des Δ moyens avec clipping robuste."""
    if delta_mat.empty:
        print("Heatmap: no data.")
        return
    V = delta_mat.values
    finite = V[np.isfinite(V)]
    if finite.size == 0:
        print("Heatmap: all values are NaN.")
        return
    vmax = np.quantile(np.abs(finite), clip_q)
    vmin = -vmax

    # Construire labels de ligne: [0007] s=3
    score_map = dict(scores_df[["patient","score"]].values)
    row_labels = [f"{pid4(r)} (s={int(score_map.get(r, np.nan)) if np.isfinite(score_map.get(r, np.nan)) else 'NaN'})"
                  for r in delta_mat.index]

    fig, ax = plt.subplots(figsize=(max(10, delta_mat.shape[1]*0.35), max(6, delta_mat.shape[0]*0.35)))
    im = ax.imshow(np.ma.array(V, mask=~np.isfinite(V)), aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax, cmap="coolwarm", interpolation="nearest")
    ax.set_title("Δ means (Patho − Healthy) global heatmap ")
    ax.set_xlabel("Frequences")
    ax.set_ylabel("Patients")
    ax.set_yticks(range(delta_mat.shape[0]))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xticks(range(delta_mat.shape[1]))
    ax.set_xticklabels([c.replace("Hz","") for c in delta_mat.columns], rotation=90, fontsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Δ mean (Patho − Healthy)")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
        plt.close(fig)
    else:
        plt.show()

def summarize_by_patient(long_df, topk=5):
    """Métriques par patient (+ top fréquences par |Δ|)."""
    if long_df.empty:
        return pd.DataFrame()
    rows = []
    for pid, g in long_df.groupby("patient"):
        top = (g.assign(abs_delta=lambda d: d["Delta_mean"].abs())
                 .sort_values("abs_delta", ascending=False)
                 .head(topk))
        rows.append({
            "patient": int(pid),
            "pid4": pid4(pid),
            "score": float(g["score"].iloc[0]) if len(g) else np.nan,
            "n_freq": int(len(g)),
            "cov_h_mean": float(g["cov_h"].mean()),
            "cov_p_mean": float(g["cov_p"].mean()),
            "delta_mean_avg": float(g["Delta_mean"].mean()),
            "delta_abs_mean": float(g["Delta_mean"].abs().mean()),
            "top_freq_by_abs_delta": ", ".join(top["freq"].tolist()),
            "top_abs_delta_values": ", ".join([f"{v:.3g}" for v in top["Delta_mean"].tolist()]),
        })
    return pd.DataFrame(rows).sort_values("patient").reset_index(drop=True)

def analyze_all(scores_df, outdir="reports", min_cov=0.3, top=15,
                save_plots=False, heatmap=True, heatmap_clip=0.98, topk_patient=5):
    os.makedirs(outdir, exist_ok=True)

    long_df = build_long_table(scores_df, min_cov=min_cov)
    if long_df.empty:
        print("No exploitable data on the whole dataset.")
        return

    summary = summarize_by_frequency(long_df)

    # Sauvegardes de base
    long_path = os.path.join(outdir, "per_patient_frequency.csv")
    sum_path  = os.path.join(outdir, "per_frequency_summary.csv")
    long_df.to_csv(long_path, index=False)
    summary.to_csv(sum_path)
    print(f"> Saved: {long_path}")
    print(f"> Saved: {sum_path}")

    # Heatmap Δ (patients × fréquences)
    delta_mat = build_delta_matrix(long_df, order="abs_delta")
    delta_path = os.path.join(outdir, "delta_matrix.csv")
    delta_mat.to_csv(delta_path)
    print(f"> Saved: {delta_path}")

    if heatmap:
        figpath = os.path.join(outdir, "dataset_heatmap.png") if save_plots else None
        plot_global_heatmap(delta_mat, scores_df, clip_q=heatmap_clip, outpath=figpath)
        if figpath:
            print(f"> Figure saved: {figpath}")

    # Résumé par patient
    by_patient = summarize_by_patient(long_df, topk=topk_patient)
    by_patient_path = os.path.join(outdir, "per_patient_summary.csv")
    by_patient.to_csv(by_patient_path, index=False)
    print(f"> Sauvé: {by_patient_path}")

    # Tops (console)
    top_abs = summary.dropna(subset=["delta_abs_mean"]).sort_values("delta_abs_mean", ascending=False).head(top)
    top_corr = summary.dropna(subset=["r_score_delta"]).reindex(summary["r_score_delta"].abs().sort_values(ascending=False).index).head(top)

    print("\nTop frequency by |Δ| moyen (Patho-Healthy):")
    print(top_abs[["delta_abs_mean","n_patients","cov_h_mean","cov_p_mean"]].round(4).to_string())

    print("\nTop frequency by |corr(Δ, score)| :")
    print(top_corr[["r_score_delta","n_patients","cov_h_mean","cov_p_mean"]].round(4).to_string())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=int, default=None)
    parser.add_argument("--pca", action="store_true")
    parser.add_argument("--k_lines", type=int, default=6)

    parser.add_argument("--summary", action="store_true",
                        help="Analyse l'ensemble du dataset et génère un résumé (CSV + heatmap + per-patient).")
    parser.add_argument("--outdir", type=str, default="reports", help="Dossier de sortie")
    parser.add_argument("--min_cov", type=float, default=0.3, help="Couverture minimale (0–1) par série pour garder une fréquence")
    parser.add_argument("--top", type=int, default=15, help="Combien de fréquences afficher dans les tops")
    parser.add_argument("--save-plots", action="store_true", help="Sauver les figures au lieu d'afficher")
    parser.add_argument("--no-heatmap", action="store_true", help="Ne pas générer la heatmap globale")
    parser.add_argument("--heatmap-clip", type=float, default=0.98, help="Quantile de clipping pour la heatmap (0.9–1.0)")
    parser.add_argument("--topk-patient", type=int, default=5, help="Top-k fréquences listées par patient")





    args = parser.parse_args()
    scores_df = read_patients_scores(LIST_PATH)
    if args.summary:
        analyze_all(scores_df,
                    outdir=args.outdir,
                    min_cov=args.min_cov,
                    top=args.top,
                    save_plots=bool(args.save_plots),
                    heatmap=(not args.no_heatmap),
                    heatmap_clip=args.heatmap_clip,
                    topk_patient=args.topk_patient)
        return

    if args.patient is not None:
        plot_patient(args.patient, scores_df, k_lines=args.k_lines)
    if args.pca:
        plot_pca_overview(scores_df)
    if args.patient is None and not args.pca:
        if len(scores_df):
            plot_patient(int(scores_df.iloc[0]["patient"]), scores_df, k_lines=args.k_lines)

if __name__ == "__main__":
    main()
