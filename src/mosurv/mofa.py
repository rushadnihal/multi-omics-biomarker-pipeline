"""MOFA+ factors (fitted in-fold, omics views only) -> Cox. Test rows are projected, never used in fitting."""
import numpy as np


def project_factors(views, weights, taus):
    """Posterior-mean-style projection of samples onto factors given loadings.

    Z = (I + sum_v tau_v W_v' W_v)^-1 sum_v tau_v W_v' x_v   (rows = samples).
    views: list of (n x D_v), weights: list of (D_v x K), taus: list of floats.
    """
    K = weights[0].shape[1]
    A = np.eye(K)
    B = 0.0
    for X, W, t in zip(views, weights, taus):
        A = A + t * (W.T @ W)
        B = B + t * (X @ W)
    return np.linalg.solve(A, np.asarray(B).T).T


def fit_mofa(views, view_names, n_factors, seed, iterations=1000, convergence_mode="fast"):
    """Fit MOFA+ on training views (samples x features, already standardised). Returns (weights, taus)."""
    from mofapy2.run.entry_point import entry_point

    n = views[0].shape[0]
    ent = entry_point()
    ent.set_data_options(scale_groups=False, scale_views=False)
    ent.set_data_matrix(
        [[v.astype(np.float64)] for v in views],
        likelihoods=["gaussian"] * len(views),
        views_names=list(view_names),
        groups_names=["g0"],
        samples_names=[[f"s{i}" for i in range(n)]],
        features_names=[[f"{vn}_{j}" for j in range(v.shape[1])] for vn, v in zip(view_names, views)],
    )
    ent.set_model_options(factors=n_factors, spikeslab_weights=True, ard_factors=True, ard_weights=True)
    ent.set_train_options(iter=iterations, convergence_mode=convergence_mode, dropR2=None, gpu_mode=False,
                          seed=int(seed), verbose=False, quiet=True)
    ent.build()
    ent.run()
    W = [np.asarray(w, dtype=float) for w in ent.model.nodes["W"].getExpectation()]
    T = [float(np.nanmean(np.asarray(t, dtype=float))) for t in ent.model.nodes["Tau"].getExpectation()]
    return W, T


def mofa_fold(Xs, y, tr, te, prep_factory, offsets, sizes, omics, cfg, seed, tau_days):
    from sksurv.linear_model import CoxPHSurvivalAnalysis

    from mosurv.evaluate import harrell_c, uno_c

    ytr, yte = y[tr], y[te]
    prep = prep_factory().fit(Xs, tr, ytr)
    Ptr, Pte = prep.transform(Xs, tr), prep.transform(Xs, te)
    codes = np.asarray(prep.cols_)
    idx = {k: np.where((codes >= offsets[k]) & (codes < offsets[k] + sizes[k]))[0] for k in ["clin", *omics]}
    D = np.array([len(idx[k]) for k in omics], dtype=float)
    scale = np.sqrt(D.mean() / D)  # equalise total variance per view
    Vtr = [Ptr[:, idx[k]] * s for k, s in zip(omics, scale)]
    Vte = [Pte[:, idx[k]] * s for k, s in zip(omics, scale)]
    W, T = fit_mofa(Vtr, omics, cfg["n_factors"], seed, cfg["iterations"], cfg["convergence_mode"])
    Ztr, Zte = project_factors(Vtr, W, T), project_factors(Vte, W, T)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-12
    Ztr, Zte = (Ztr - mu) / sd, (Zte - mu) / sd
    out = {"n_per_view": {k: int(len(idx[k])) for k in omics}}
    for name, cols in (("MOFA factors", False), ("clinical + MOFA factors", True)):
        A = np.hstack([Ptr[:, idx["clin"]], Ztr]) if cols else Ztr
        B = np.hstack([Pte[:, idx["clin"]], Zte]) if cols else Zte
        m = CoxPHSurvivalAnalysis(alpha=cfg["cox_alpha"]).fit(A, ytr)
        risk = m.predict(B)
        out[name] = dict(harrell=harrell_c(yte["event"], yte["time"], risk), uno=uno_c(ytr, yte, risk, tau_days),
                         coef=m.coef_[-Ztr.shape[1]:].tolist())
    return out
