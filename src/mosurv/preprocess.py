"""Per-layer unit transforms and outcome-blind QC helpers.

Only sample-wise / fixed transforms live here (no cross-sample statistics), so they are safe to
apply globally. Cross-sample steps (filters, imputation, scaling) belong inside CV pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def log2cpm_from_log2count1(X: pd.DataFrame, prior: float = 1.0):
    """X: genes x samples holding log2(count+1). Returns (log2(CPM+prior) float32, library sizes)."""
    counts = np.clip(np.exp2(X.to_numpy(dtype=np.float64)) - 1.0, 0.0, None)
    lib = counts.sum(axis=0)
    out = np.log2(counts / lib * 1e6 + prior).astype(np.float32)
    return pd.DataFrame(out, index=X.index, columns=X.columns), pd.Series(lib, index=X.columns, name="libsize")


def beta_to_m(beta: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Beta -> M-value, clipping to [eps, 1-eps]. NaN is preserved."""
    b = np.clip(beta, eps, 1.0 - eps)
    return np.log2(b / (1.0 - b)).astype(np.float32)


def pca_scores(F: pd.DataFrame, k: int = 5):
    """F: features x samples. Returns (scores samples x k, fraction of variance for first k PCs)."""
    X = F.to_numpy(dtype=np.float64).T
    X = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    return U[:, :k] * S[:k], var[:k]


def r2_by_group(y, groups) -> float:
    """Fraction of variance of y explained by a categorical grouping (one-way ANOVA R^2)."""
    y = np.asarray(y, dtype=float)
    g = pd.Series(np.asarray(groups, dtype=object))
    ok = g.notna().to_numpy()
    y, g = y[ok], g[ok].reset_index(drop=True)
    sst = ((y - y.mean()) ** 2).sum()
    means = pd.Series(y).groupby(g).agg(["mean", "size"])
    ssb = (means["size"] * (means["mean"] - y.mean()) ** 2).sum()
    return float(ssb / sst) if sst > 0 else float("nan")


def robust_outliers(y, thresh: float = 4.0) -> int:
    y = np.asarray(y, dtype=float)
    mad = 1.4826 * np.median(np.abs(y - np.median(y)))
    return int((np.abs(y - np.median(y)) > thresh * mad).sum()) if mad > 0 else 0


AUTOSOMES = {f"chr{i}" for i in range(1, 23)}


def autosomal_probes(probemap: pd.DataFrame) -> pd.Index:
    """Probes the annotation places on chr1-22 (fixed, outcome-blind rule).

    probemap needs columns '#id' and 'chrom'. Probes absent from the map, or without a chromosome,
    are NOT returned (their chromosome cannot be verified, so sex-linked probes could hide there).
    """
    ok = probemap["chrom"].isin(AUTOSOMES)
    return pd.Index(probemap.loc[ok, "#id"].unique())
