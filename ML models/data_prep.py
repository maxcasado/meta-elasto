#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_prep.py — Data extraction + cleaning + configurable outlier/NaN handling for the Skin dataset.

Main features:
- Read patient scores (Skin/0list.xlsx)
- Load Healthy / Patho measurements (Skin/Resultados/Elastome_XXXX_*.csv)
- Configurable cleaning:
    * NaN: none | median | mean | ffill | bfill | interpolate
    * Outliers: none | zscore | iqr | clip
- Post-cleaning minimum coverage filter (min_cov)
- Export cleaned CSVs to outdir/clean
- Save a cleaning summary (CSV) + optional cleaned long table export

Dependencies: numpy, pandas (matplotlib/scikit-learn not required here).
"""

import os
import argparse
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

# --- Default paths, compatible with your current project layout ---
BASE_DIR = "Skin"
RES_DIR = os.path.join(BASE_DIR, "Resultados")
LIST_PATH = os.path.join(BASE_DIR, "0list.xlsx")

# --------- Basic utils ---------
def pid4(n: int) -> str:
    return f"{int(n):04d}"

def read_patients_scores(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=None, usecols=[0, 1], names=["patient", "score"])
    df["patient"] = pd.to_numeric(df["patient"], errors="coerce")
    df["score"]   = pd.to_numeric(df["score"],   errors="coerce")
    df = df.dropna(subset=["patient"])
    df["patient"] = df["patient"].astype(int)
    return df

def load_measure(pid: int, kind: str) -> Optional[pd.DataFrame]:
    """
    kind ∈ {"Healthy", "Patho"}
    """
    fname = f"Elastome_{pid4(pid)}_{kind}_angle_1.csv"
    fpath = os.path.join(RES_DIR, fname)
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    df = df.rename_axis("t").reset_index()  # index -> column "t"
    # numeric cast + NaN for non-numeric
    for c in df.columns:
        if c != "t":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace({np.inf: np.nan, -np.inf: np.nan})
    return df

# --------- Cleaning: NaNs + Outliers ---------
def _nan_impute(s: pd.Series, method: str, interpolate_limit: Optional[int]) -> pd.Series:
    if method == "none":
        return s
    if method == "median":
        return s.fillna(np.nanmedian(s.values))
    if method == "mean":
        return s.fillna(np.nanmean(s.values))
    if method == "ffill":
        return s.ffill()
    if method == "bfill":
        return s.bfill()
    if method == "interpolate":
        return s.interpolate(method="linear",
                             limit=interpolate_limit,
                             limit_direction="both")
    return s

def _mark_outliers(s: pd.Series, method: str,
                   z_thresh: float = 3.0,
                   iqr_k: float = 1.5,
                   clip_low_q: float = 0.01,
                   clip_high_q: float = 0.99) -> Tuple[pd.Series, np.ndarray]:
    """
    Returns (s_out, mask_outliers)
    - zscore: |(x - mean)/std| > z_thresh
    - iqr:    x < Q1 - k*IQR or x > Q3 + k*IQR
    - clip:   hard-clip to [low_q, high_q]
    """
    x = s.to_numpy(dtype=float)
    mask_nan = ~np.isfinite(x)

    if method == "none":
        return s, np.zeros_like(x, dtype=bool)

    if method == "zscore":
        finite = x[~mask_nan]
        if finite.size < 3 or np.nanstd(finite) == 0:
            return s, np.zeros_like(x, dtype=bool)
        mu = np.nanmean(finite)
        sd = np.nanstd(finite)
        z = (x - mu) / sd
        mask = np.abs(z) > z_thresh
        # Replace outliers with NaN (to be imputed)
        x[mask] = np.nan
        return pd.Series(x, index=s.index), mask

    if method == "iqr":
        finite = x[~mask_nan]
        if finite.size < 3:
            return s, np.zeros_like(x, dtype=bool)
        q1 = np.nanpercentile(finite, 25)
        q3 = np.nanpercentile(finite, 75)
        iqr = q3 - q1
        low = q1 - iqr_k * iqr
        high = q3 + iqr_k * iqr
        mask = (x < low) | (x > high)
        x[mask] = np.nan
        return pd.Series(x, index=s.index), mask

    if method == "clip":
        finite = x[~mask_nan]
        if finite.size < 3:
            return s, np.zeros_like(x, dtype=bool)
        lo = np.nanquantile(finite, clip_low_q)
        hi = np.nanquantile(finite, clip_high_q)
        before = x.copy()
        x = np.clip(x, lo, hi)
        mask = (before != x) & np.isfinite(before)
        return pd.Series(x, index=s.index), mask

    return s, np.zeros_like(x, dtype=bool)

def clean_series(
    s: pd.Series,
    nan_method: str = "interpolate",
    outlier_method: str = "iqr",
    interpolate_limit: Optional[int] = None,
    z_thresh: float = 3.0,
    iqr_k: float = 1.5,
    clip_low_q: float = 0.01,
    clip_high_q: float = 0.99
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Cleaning pipeline for one series:
      1) outlier handling (-> NaN or clipping)
      2) NaN imputation
    Returns (s_clean, stats)
    """
    x0 = s.copy()
    stats = {
        "n": len(s),
        "n_nan_before": int(s.isna().sum()),
        "outlier_method": outlier_method,
        "nan_method": nan_method,
        "outliers_flagged": 0,
    }

    # Outliers
    s1, mask_out = _mark_outliers(
        x0, method=outlier_method, z_thresh=z_thresh, iqr_k=iqr_k,
        clip_low_q=clip_low_q, clip_high_q=clip_high_q
    )
    stats["outliers_flagged"] = int(mask_out.sum())

    # NaN imputation
    s2 = _nan_impute(s1, method=nan_method, interpolate_limit=interpolate_limit)
    stats["n_nan_after"] = int(s2.isna().sum())

    return s2, stats

def clean_dataframe(
    df: pd.DataFrame,
    nan_method: str,
    outlier_method: str,
    interpolate_limit: Optional[int],
    z_thresh: float,
    iqr_k: float,
    clip_low_q: float,
    clip_high_q: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply clean_series to every column != 't'.
    Returns (df_clean, stats_df) where stats_df holds per-column stats.
    """
    assert "t" in df.columns, "Column 't' must exist."
    stats_rows = []
    df_clean = df.copy()
    for c in df.columns:
        if c == "t":
            continue
        s_clean, st = clean_series(
            df[c],
            nan_method=nan_method,
            outlier_method=outlier_method,
            interpolate_limit=interpolate_limit,
            z_thresh=z_thresh,
            iqr_k=iqr_k,
            clip_low_q=clip_low_q,
            clip_high_q=clip_high_q
        )
        df_clean[c] = s_clean
        st.update({"column": c})
        stats_rows.append(st)
    stats_df = pd.DataFrame(stats_rows).set_index("column") if stats_rows else pd.DataFrame()
    return df_clean, stats_df

def common_columns(df_h: pd.DataFrame, df_p: pd.DataFrame) -> List[str]:
    cols_h = [c for c in df_h.columns if c != "t"]
    cols_p = [c for c in df_p.columns if c != "t"]
    return [c for c in cols_h if c in cols_p]

def coverage(series: pd.Series) -> float:
    v = series.to_numpy()
    return np.isfinite(v).mean()

# --------- Per-patient pipeline ---------
def process_patient(
    pid: int,
    outdir_clean: str,
    min_cov: float,
    nan_method: str,
    outlier_method: str,
    interpolate_limit: Optional[int],
    z_thresh: float,
    iqr_k: float,
    clip_low_q: float,
    clip_high_q: float
) -> Dict[str, object]:
    """
    Clean Healthy & Patho for one patient, keep common columns,
    filter by post-cleaning coverage, export cleaned CSVs.
    Returns a dict with patient-level stats.
    """
    df_h_raw = load_measure(pid, "Healthy")
    df_p_raw = load_measure(pid, "Patho")
    status = {
        "patient": pid,
        "pid4": pid4(pid),
        "has_healthy": df_h_raw is not None,
        "has_patho": df_p_raw is not None,
        "n_cols_common_before": 0,
        "n_cols_kept_after": 0
    }
    if df_h_raw is None or df_p_raw is None:
        return status

    # Clean separately
    df_h_clean, stats_h = clean_dataframe(
        df_h_raw, nan_method, outlier_method, interpolate_limit,
        z_thresh, iqr_k, clip_low_q, clip_high_q
    )
    df_p_clean, stats_p = clean_dataframe(
        df_p_raw, nan_method, outlier_method, interpolate_limit,
        z_thresh, iqr_k, clip_low_q, clip_high_q
    )

    # Column intersection
    cols = common_columns(df_h_clean, df_p_clean)
    status["n_cols_common_before"] = len(cols)

    # Coverage filter
    kept = []
    for c in cols:
        cov_h = coverage(df_h_clean[c])
        cov_p = coverage(df_p_clean[c])
        if cov_h >= min_cov and cov_p >= min_cov:
            kept.append(c)

    status["n_cols_kept_after"] = len(kept)

    # Reduce to kept columns + 't'
    keep_cols = ["t"] + kept
    df_h_final = df_h_clean[keep_cols].copy()
    df_p_final = df_p_clean[keep_cols].copy()

    # Exports
    os.makedirs(outdir_clean, exist_ok=True)
    f_h = os.path.join(outdir_clean, f"Elastome_{pid4(pid)}_Healthy_clean.csv")
    f_p = os.path.join(outdir_clean, f"Elastome_{pid4(pid)}_Patho_clean.csv")
    df_h_final.to_csv(f_h, index=False)
    df_p_final.to_csv(f_p, index=False)

    status.update({
        "path_healthy_clean": f_h,
        "path_patho_clean": f_p
    })
    return status

# --------- Optional cleaned long table ---------
def build_clean_long_table(scores_df: pd.DataFrame, outdir_clean: str, min_cov: float) -> pd.DataFrame:
    """
    Build a long table (patient × frequency) from the *_clean.csv exports,
    with coverage and H/P means.
    """
    rows = []
    for pid in scores_df["patient"].tolist():
        pid_s = pid4(pid)
        f_h = os.path.join(outdir_clean, f"Elastome_{pid_s}_Healthy_clean.csv")
        f_p = os.path.join(outdir_clean, f"Elastome_{pid_s}_Patho_clean.csv")
        if not (os.path.exists(f_h) and os.path.exists(f_p)):
            continue
        df_h = pd.read_csv(f_h)
        df_p = pd.read_csv(f_p)
        cols = [c for c in df_h.columns if c != "t" and c in df_p.columns]
        for c in cols:
            xh = pd.to_numeric(df_h[c], errors="coerce").astype(float).values
            xp = pd.to_numeric(df_p[c], errors="coerce").astype(float).values
            cov_h = np.isfinite(xh).mean()
            cov_p = np.isfinite(xp).mean()
            if cov_h < min_cov or cov_p < min_cov:
                continue
            rows.append({
                "patient": int(pid),
                "pid4": pid_s,
                "score": float(scores_df.loc[scores_df["patient"] == pid, "score"].iloc[0]) if len(scores_df) else np.nan,
                "freq": c,
                "cov_h": cov_h,
                "cov_p": cov_p,
                "H_mean": float(np.nanmean(xh)),
                "P_mean": float(np.nanmean(xp)),
                "Delta_mean": float(np.nanmean(xp) - np.nanmean(xh)),
            })
    return pd.DataFrame(rows)

# --------- CLI ---------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Data extraction & cleaning for Skin dataset")
    # Patient selection
    p.add_argument("--only-patient", type=int, default=None,
                   help="If provided, process only this patient ID.")
    # Minimum coverage after cleaning
    p.add_argument("--min-cov", type=float, default=0.3,
                   help="Minimum coverage (0–1) to keep a frequency after cleaning.")
    # NaN methods
    p.add_argument("--nan-method", type=str, default="interpolate",
                   choices=["none", "median", "mean", "ffill", "bfill", "interpolate"],
                   help="NaN imputation method.")
    p.add_argument("--interpolate-limit", type=int, default=None,
                   help="Max gap length for interpolation (None = unlimited).")
    # Outlier methods
    p.add_argument("--outlier-method", type=str, default="iqr",
                   choices=["none", "zscore", "iqr", "clip"],
                   help="Outlier handling method.")
    p.add_argument("--z-thresh", type=float, default=3.0,
                   help="Z-score threshold (outlier-method=zscore).")
    p.add_argument("--iqr-k", type=float, default=1.5,
                   help="IQR multiplier (outlier-method=iqr).")
    p.add_argument("--clip-low-q", type=float, default=0.01,
                   help="Lower quantile for clipping (outlier-method=clip).")
    p.add_argument("--clip-high-q", type=float, default=0.99,
                   help="Upper quantile for clipping (outlier-method=clip).")
    # Outputs
    p.add_argument("--outdir", type=str, default="reports",
                   help="Main output folder.")
    p.add_argument("--export-long", action="store_true",
                   help="Also build and export a cleaned long table.")
    return p.parse_args()

def main():
    args = parse_args()

    # Prepare outputs
    outdir = args.outdir
    outdir_clean = os.path.join(outdir, "clean")
    os.makedirs(outdir_clean, exist_ok=True)

    # Read patient list
    scores_df = read_patients_scores(LIST_PATH)
    if scores_df.empty:
        print(f"[!] No patients found in {LIST_PATH}")
        return

    # Selection
    pids = [args.only_patient] if args.only_patient is not None else scores_df["patient"].tolist()

    # Global stats
    all_status = []

    # Processing
    for pid in pids:
        st = process_patient(
            pid=pid,
            outdir_clean=outdir_clean,
            min_cov=args.min_cov,
            nan_method=args.nan_method,
            outlier_method=args.outlier_method,
            interpolate_limit=args.interpolate_limit,
            z_thresh=args.z_thresh,
            iqr_k=args.iqr_k,
            clip_low_q=args.clip_low_q,
            clip_high_q=args.clip_high_q
        )
        all_status.append(st)

    # Export per-patient status
    status_df = pd.DataFrame(all_status)
    status_path = os.path.join(outdir, "cleaning_status_by_patient.csv")
    status_df.to_csv(status_path, index=False)
    print(f"> Saved: {status_path}")

    # Optional cleaned long table
    if args.export_long:
        long_df = build_clean_long_table(scores_df, outdir_clean, min_cov=args.min_cov)
        long_path = os.path.join(outdir, "clean_long_table.csv")
        long_df.to_csv(long_path, index=False)
        print(f"> Saved: {long_path}")

    print("\n[OK] Cleaning completed.")
    print(f"- NaN method     : {args.nan_method}")
    print(f"- Outlier method : {args.outlier_method}")
    print(f"- min_cov        : {args.min_cov}")
    print(f"- Outputs in     : {outdir}")

if __name__ == "__main__":
    main()
