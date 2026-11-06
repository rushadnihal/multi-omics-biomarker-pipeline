"""Final model fit on all training patients, and scoring of any cohort from a frozen signature file.

Scoring rules (fixed before any external data are scored):
  clinical block : TCGA medians for missing values, TCGA mean/sd, frozen coefficients
  RNA block      : genes by symbol; expression on log scale; per-gene z-score WITHIN the cohort being scored
                   (gene_scaling='cohort'), or with frozen TCGA mean/sd (gene_scaling='frozen', used to verify
                   that the saved file reproduces the TCGA fit)
"""
import numpy as np


def fit_final(Xs, y, rows, prep_factory, l1_grid, inner_folds, n_alphas, alpha_min_ratio, seed):
    """Same tuning procedure as models.nested_fold (inner-CV Harrell C over l1 x alpha), but fit on all `rows`."""
    from sklearn.model_selection import StratifiedKFold

    from mosurv.evaluate import harrell_c
    from mosurv.models import _fit_path

    rows = np.asarray(rows)
    yr = y[rows]
    prep = prep_factory().fit(Xs, rows, yr)
    P = prep.transform(Xs, rows)
    pf = getattr(prep, "penalty_factor_", None)
    paths = {l1: _fit_path(P, yr, l1, None, n_alphas, alpha_min_ratio, pf).alphas_ for l1 in l1_grid}
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    score = {l1: np.full((inner_folds, len(a)), np.nan) for l1, a in paths.items()}
    for f, (itr, iva) in enumerate(inner.split(np.zeros(len(rows)), yr["event"])):
        r_tr, r_va = rows[itr], rows[iva]
        ip = prep_factory().fit(Xs, r_tr, y[r_tr])
        A, B = ip.transform(Xs, r_tr), ip.transform(Xs, r_va)
        ipf = getattr(ip, "penalty_factor_", None)
        for l1, alphas in paths.items():
            try:
                m = _fit_path(A, y[r_tr], l1, alphas, penalty_factor=ipf)
            except Exception:
                continue
            for j, a in enumerate(alphas):
                score[l1][f, j] = harrell_c(y[r_va]["event"], y[r_va]["time"], m.predict(B, alpha=a))
    inner_c, l1, j = max(((np.nanmean(s[:, j]), l1, j) for l1, s in score.items() if not np.isnan(s).all()
                          for j in range(s.shape[1]) if not np.isnan(s[:, j]).all()), key=lambda r: r[0])
    alpha = paths[l1][j]
    final = _fit_path(P, yr, l1, paths[l1], penalty_factor=pf)
    coef = final.coef_[:, int(np.argmin(np.abs(final.alphas_ - alpha)))].copy()
    return dict(prep=prep, coef=coef, l1=float(l1), alpha=float(alpha), inner_c=float(inner_c), lp=P @ coef)


def score(sig, clin, expr, gene_scaling="cohort"):
    """clin: (n x p) array in sig['clinical']['features'] order (NaN allowed); expr: DataFrame genes x samples.
    Returns dict(total, clinical, rna) of linear predictors (higher = higher risk)."""
    c = sig["clinical"]
    x = np.asarray(clin, dtype=float)
    x = np.where(np.isnan(x), np.asarray(c["median"]), x)
    lp_c = ((x - np.asarray(c["mean"])) / np.asarray(c["sd"])) @ np.asarray(c["coef"])
    r = sig["rna"]
    E = expr.loc[r["genes"]].to_numpy(dtype=float).T          # samples x genes
    if gene_scaling == "cohort":
        mu, sd = E.mean(axis=0), E.std(axis=0) + 1e-8
    else:
        mu, sd = np.asarray(r["mean"]), np.asarray(r["sd"])
    lp_r = ((E - mu) / sd) @ np.asarray(r["coef"])
    return dict(total=lp_c + lp_r, clinical=lp_c, rna=lp_r)
