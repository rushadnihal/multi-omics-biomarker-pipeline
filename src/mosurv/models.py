"""Model definitions (sklearn Pipelines: every data-dependent step is fitted inside the fold)."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clinical_cox(features, alpha: float = 1e-4):
    """Small unpenalised-ish Cox (tiny ridge for numerical stability) on clinical covariates.

    features subset of: age, gender, stage, pack_years.
    age: median-impute + scale. pack_years: median-impute + missing indicator + scale.
    gender/stage: most-frequent impute + one-hot.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    parts = []
    if "age" in features:
        parts.append(("age", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), ["age"]))
    if "pack_years" in features:
        parts.append(("smk", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                                       ("sc", StandardScaler())]), ["pack_years"]))
    cats = [c for c in ("gender", "stage") if c in features]
    if cats:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore"))]), cats))
    prep = ColumnTransformer(parts, sparse_threshold=0.0)
    return Pipeline([("prep", prep), ("cox", CoxPHSurvivalAnalysis(alpha=alpha))])


def _fit_path(X, y, l1, alphas=None, n_alphas=30, alpha_min_ratio=0.05, penalty_factor=None):
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    kw = dict(l1_ratio=l1, fit_baseline_model=False, max_iter=100000)
    if alphas is None:
        kw.update(n_alphas=n_alphas, alpha_min_ratio=alpha_min_ratio)
    else:
        kw.update(alphas=np.sort(alphas)[::-1])
    if penalty_factor is not None:
        kw["penalty_factor"] = np.asarray(penalty_factor, dtype=float)   # 0 = unpenalised (always in model)
    return CoxnetSurvivalAnalysis(**kw).fit(X, y)


def nested_fold(X, y, tr, te, prep_factory, l1_grid, inner_folds, n_alphas, alpha_min_ratio, seed, tau_days,
                return_oof=False):
    """One outer fold of nested CV for an omics-only elastic-net Cox. X is the full matrix; tr/te row indices.

    All data-dependent steps (filters, imputation, screening, scaling, tuning) are fitted on training rows only;
    inside the tuning loop they are re-fitted on every inner training split.
    """
    from sklearn.model_selection import StratifiedKFold

    from mosurv.evaluate import harrell_c, uno_c

    ytr, yte = y[tr], y[te]
    prep = prep_factory().fit(X, tr, ytr)
    Ptr = prep.transform(X, tr)
    pf = getattr(prep, "penalty_factor_", None)
    paths, fails = {}, 0
    for l1 in l1_grid:                                  # alpha grids from OUTER-TRAIN data only
        try:
            paths[l1] = _fit_path(Ptr, ytr, l1, None, n_alphas, alpha_min_ratio, pf).alphas_
        except Exception:
            fails += 1
    if not paths:
        return dict(failed=True)
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    score = {l1: np.full((inner_folds, len(a)), np.nan) for l1, a in paths.items()}
    oofs = {l1: np.full((len(tr), len(a)), np.nan) for l1, a in paths.items()}   # inner out-of-fold predictions
    for f, (itr, iva) in enumerate(inner.split(np.zeros(len(tr)), ytr["event"])):
        r_tr, r_va = tr[itr], tr[iva]
        ip = prep_factory().fit(X, r_tr, y[r_tr])        # screening redone inside inner fold
        A, B = ip.transform(X, r_tr), ip.transform(X, r_va)
        ipf = getattr(ip, "penalty_factor_", None)
        for l1, alphas in paths.items():
            try:
                m = _fit_path(A, y[r_tr], l1, alphas, penalty_factor=ipf)
            except Exception:
                fails += 1
                continue
            for j, a in enumerate(alphas):
                pr = m.predict(B, alpha=a)
                oofs[l1][iva, j] = pr
                score[l1][f, j] = harrell_c(y[r_va]["event"], y[r_va]["time"], pr)
    best = max(((np.nanmean(s[:, j]), l1, j) for l1, s in score.items() if not np.isnan(s).all()
                for j in range(s.shape[1]) if not np.isnan(s[:, j]).all()), key=lambda r: r[0])
    inner_c, l1, j = best
    alpha = paths[l1][j]
    final = _fit_path(Ptr, ytr, l1, paths[l1], penalty_factor=pf)
    risk = final.predict(prep.transform(X, te), alpha=alpha)
    coef = final.coef_[:, int(np.argmin(np.abs(final.alphas_ - alpha)))]
    pen = np.ones(len(coef), bool) if pf is None else (np.asarray(pf) > 0)   # selection counts penalised features only
    nz = np.flatnonzero((coef != 0) & pen)
    out = dict(failed=False, harrell=harrell_c(yte["event"], yte["time"], risk),
                uno=uno_c(ytr, yte, risk, tau_days), l1=l1, alpha=float(alpha), inner_c=float(inner_c),
                n_nonzero=int(len(nz)), sel_cols=prep.cols_[nz].tolist(), n_prev=prep.n_kept_prevalence_,
                n_fail=fails, risk=risk.tolist())
    if return_oof:
        out["oof_train"] = oofs[l1][:, j].tolist()   # inner-CV OOF scores of the TRAIN rows at the chosen penalty
    return out


def late_fold(Xs, y, tr, te, factories, l1_grid, inner_folds, n_alphas, alpha_min_ratio, seed, tau_days, meta_alpha=1e-2):
    """Late integration for one outer fold: per-layer tuned elastic-net -> stack with clinical in a small Cox.

    The stacker is trained on the INNER-CV out-of-fold layer scores of the training rows, so it never sees a score
    produced by a model that was fitted on the same patient. Test scores come from per-layer models refit on all
    training rows.
    """
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    from mosurv.evaluate import harrell_c, uno_c
    from mosurv.features import ClinPrep

    layer = {}
    for k, fac in factories.items():
        r = nested_fold(Xs[k], y, tr, te, fac, l1_grid, inner_folds, n_alphas, alpha_min_ratio, seed, tau_days,
                        return_oof=True)
        if r["failed"]:
            return dict(failed=True)
        layer[k] = r
    cp = ClinPrep().fit(Xs["clin"], tr, y[tr])
    Ztr = np.column_stack([cp.transform(Xs["clin"], tr)] + [np.asarray(layer[k]["oof_train"]) for k in factories])
    Zte = np.column_stack([cp.transform(Xs["clin"], te)] + [np.asarray(layer[k]["risk"]) for k in factories])
    mu, sd = Ztr.mean(axis=0), Ztr.std(axis=0) + 1e-8
    meta = CoxPHSurvivalAnalysis(alpha=meta_alpha).fit((Ztr - mu) / sd, y[tr])
    risk = meta.predict((Zte - mu) / sd)
    yte, ytr = y[te], y[tr]
    return dict(failed=False, harrell=harrell_c(yte["event"], yte["time"], risk),
                uno=uno_c(ytr, yte, risk, tau_days), coef=meta.coef_.tolist(), risk=risk.tolist(),
                layer_harrell={k: layer[k]["harrell"] for k in factories},
                layer_inner_c={k: layer[k]["inner_c"] for k in factories},
                n_fail=sum(layer[k]["n_fail"] for k in factories))


def nonlinear_fold(Xs, y, tr, te, prep_factory, nl_cfg, seed, tau_days, n_clin):
    """One outer fold for non-linear comparators on the stacked in-fold prep (clinical block first).

    Prep is fitted once on the training rows; hyperparameters are fixed in config (no tuning).
    """
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest

    from mosurv.evaluate import harrell_c, uno_c

    ytr, yte = y[tr], y[te]
    prep = prep_factory().fit(Xs, tr, ytr)
    Ptr, Pte = prep.transform(Xs, tr), prep.transform(Xs, te)
    specs = {
        "RSF clinical": (RandomSurvivalForest(n_jobs=1, random_state=seed, **nl_cfg["rsf_clin"]), slice(0, n_clin)),
        "RSF clinical+omics": (RandomSurvivalForest(n_jobs=1, random_state=seed, **nl_cfg["rsf_omics"]), slice(None)),
        "GBM clinical+omics": (GradientBoostingSurvivalAnalysis(random_state=seed, **nl_cfg["gbm_omics"]), slice(None)),
    }
    out = {"failed": False, "cols": prep.cols_.tolist()}
    for name, (model, cols) in specs.items():
        model.fit(Ptr[:, cols], ytr)
        risk = model.predict(Pte[:, cols])
        out[name] = dict(harrell=harrell_c(yte["event"], yte["time"], risk), uno=uno_c(ytr, yte, risk, tau_days),
                         risk=risk.tolist())
        if name.startswith("GBM"):
            out[name]["importance"] = model.feature_importances_.tolist()
    return out
