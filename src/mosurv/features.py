"""In-fold feature handling: prevalence / missingness filters, univariate Cox score screening, scaling.

Every prep class is fitted on the TRAINING rows of the full matrix X (rows = index array) and then applied to
any rows. Nothing here sees held-out samples or outcomes during fit.
"""
from __future__ import annotations

import numpy as np


def cox_score_z(X, time, event, block: int = 20000) -> np.ndarray:
    """Univariate Cox score-test z for each column of X (n x p), Breslow ties, beta=0.

    z_j = U_j / sqrt(I_j) with U = sum over events of (x_i - mean of x over risk set) and
    I = sum over events of the risk-set variance of x. O(n p), vectorised, chunked over columns.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    n, p = X.shape
    order = np.argsort(time, kind="mergesort")
    t, e = time[order], event[order]
    k0 = np.searchsorted(t, t[e], side="left")      # first index of each event's tied block
    nr = (n - k0).astype(float)[:, None]
    z = np.empty(p)
    for a in range(0, p, block):
        Xs = np.asarray(X[:, a:a + block], dtype=np.float64)[order]
        S1 = np.cumsum(Xs[::-1], axis=0)[::-1]
        S2 = np.cumsum((Xs ** 2)[::-1], axis=0)[::-1]
        m1, m2 = S1[k0] / nr, S2[k0] / nr
        U = (Xs[e] - m1).sum(axis=0)
        info = (m2 - m1 ** 2).sum(axis=0)
        z[a:a + block] = U / np.sqrt(np.maximum(info, 1e-12))
    return z


class OmicsPrep:
    """prevalence filter -> top-k by |Cox score z| -> standardise. For complete (no-NaN) matrices."""

    def __init__(self, min_value: float, min_frac: float, k: int):
        self.min_value, self.min_frac, self.k = min_value, min_frac, k

    def fit(self, X, rows, y):
        Xr = np.asarray(X[rows])
        keep = np.flatnonzero((Xr > self.min_value).mean(axis=0) >= self.min_frac)
        z = np.nan_to_num(cox_score_z(Xr[:, keep], y["time"], y["event"]))
        top = np.argsort(-np.abs(z))[: self.k]
        self.cols_ = keep[top]
        sub = Xr[:, self.cols_].astype(np.float64)
        self.mu_, self.sd_ = sub.mean(axis=0), sub.std(axis=0) + 1e-8
        self.n_kept_prevalence_ = len(keep)
        return self

    def transform(self, X, rows):
        return (np.asarray(X[rows])[:, self.cols_].astype(np.float64) - self.mu_) / self.sd_


class MethPrep:
    """Block-wise: drop probes with > max_missing NaN in train -> median-impute -> top-k by |Cox z| -> standardise.

    Works column-block by column-block so no full copy of the (large) matrix is ever made.
    """

    def __init__(self, max_missing: float, k: int, block: int = 10000):
        self.max_missing, self.k, self.block = max_missing, k, block

    def fit(self, X, rows, y):
        p = X.shape[1]
        zs, idxs, n_kept = [], [], 0
        for a in range(0, p, self.block):
            B = np.asarray(X[rows, a:a + self.block], dtype=np.float64)
            nan = np.isnan(B)
            ok = nan.mean(axis=0) <= self.max_missing
            if not ok.any():
                continue
            B, nan = B[:, ok], nan[:, ok]
            B = np.where(nan, np.nanmedian(B, axis=0), B)
            z = np.nan_to_num(cox_score_z(B, y["time"], y["event"]))
            n_kept += B.shape[1]
            top = np.argsort(-np.abs(z))[: self.k]
            zs.append(z[top])
            idxs.append(a + np.flatnonzero(ok)[top])
        z, idx = np.concatenate(zs), np.concatenate(idxs)
        top = np.argsort(-np.abs(z))[: self.k]
        self.cols_ = idx[top]
        sub = np.asarray(X[np.ix_(rows, self.cols_)], dtype=np.float64)
        self.med_ = np.nanmedian(sub, axis=0)
        sub = np.where(np.isnan(sub), self.med_, sub)
        self.mu_, self.sd_ = sub.mean(axis=0), sub.std(axis=0) + 1e-8
        self.n_kept_prevalence_ = n_kept
        return self

    def transform(self, X, rows):
        B = np.asarray(X[np.ix_(rows, self.cols_)], dtype=np.float64)
        B = np.where(np.isnan(B), self.med_, B)
        return (B - self.mu_) / self.sd_


class ClinPrep:
    """Clinical covariates (n x p array with NaN allowed): median-impute and standardise on train rows."""

    def fit(self, C, rows, y):
        sub = np.asarray(C[rows], dtype=np.float64)
        self.med_ = np.nanmedian(sub, axis=0)
        sub = np.where(np.isnan(sub), self.med_, sub)
        self.mu_, self.sd_ = sub.mean(axis=0), sub.std(axis=0) + 1e-8
        self.cols_ = np.arange(C.shape[1])
        self.n_kept_prevalence_ = C.shape[1]
        return self

    def transform(self, C, rows):
        sub = np.asarray(C[rows], dtype=np.float64)
        sub = np.where(np.isnan(sub), self.med_, sub)
        return (sub - self.mu_) / self.sd_


class StackPrep:
    """Concatenate several in-fold preps. X is a dict {key: matrix}. Clinical block ('clin') is unpenalised.

    cols_ are global column codes (block offset + original column) so selections can be mapped to names.
    """

    def __init__(self, parts, offsets):
        self.parts, self.offsets = parts, offsets          # parts: [(key, factory)], offsets: {key: int}

    def fit(self, X, rows, y):
        self.preps_, pf, cols, kept = [], [], [], 0
        for key, factory in self.parts:
            p = factory().fit(X[key], rows, y)
            self.preps_.append((key, p))
            pf.append(np.full(len(p.cols_), 0.0 if key == "clin" else 1.0))
            cols.append(self.offsets[key] + p.cols_)
            kept += p.n_kept_prevalence_
        self.penalty_factor_ = np.concatenate(pf)
        self.cols_ = np.concatenate(cols)
        self.n_kept_prevalence_ = kept
        return self

    def transform(self, X, rows):
        return np.hstack([p.transform(X[key], rows) for key, p in self.preps_])
