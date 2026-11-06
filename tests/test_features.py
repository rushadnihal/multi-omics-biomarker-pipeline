import numpy as np

from mosurv.features import MethPrep, OmicsPrep, cox_score_z


def brute_force_z(x, time, event):
    U, I = 0.0, 0.0
    for i in np.flatnonzero(event):
        r = time >= time[i]
        m = x[r].mean()
        U += x[i] - m
        I += ((x[r] - m) ** 2).mean()
    return U / np.sqrt(I)


def make_y(time, event):
    y = np.zeros(len(time), dtype=[("event", bool), ("time", float)])
    y["event"], y["time"] = event, time
    return y


def test_cox_score_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    n, p = 60, 7
    X = rng.normal(size=(n, p))
    time = rng.integers(1, 15, n).astype(float)          # many ties
    event = rng.random(n) < 0.6
    z = cox_score_z(X, time, event, block=3)
    for j in range(p):
        assert np.isclose(z[j], brute_force_z(X[:, j], time, event))


def test_omicsprep_selects_signal_feature():
    rng = np.random.default_rng(1)
    n = 200
    X = rng.normal(2.0, 1.0, size=(n, 50))
    y = make_y(rng.exponential(np.exp(-X[:, 7])), np.ones(n, bool))
    rows = np.arange(n)
    prep = OmicsPrep(0.0, 0.5, 3).fit(X, rows, y)
    assert 7 in prep.cols_
    out = prep.transform(X, rows)
    assert out.shape == (n, 3) and np.allclose(out.mean(axis=0), 0, atol=1e-6)


def test_methprep_filters_imputes_and_uses_only_train_rows():
    rng = np.random.default_rng(2)
    n, p = 240, 40
    X = rng.normal(0, 1, size=(n, p)).astype(np.float32)
    y = make_y(rng.exponential(np.exp(-X[:, 5])), np.ones(n, bool))
    X[rng.random((n, p)) < 0.03] = np.nan                 # sparse missingness everywhere
    X[:, 11] = np.where(rng.random(n) < 0.5, np.nan, X[:, 11])   # probe 11: ~50% missing -> must be dropped
    X[:, 5] = np.where(rng.random(n) < 0.05, np.nan, X[:, 5])     # signal probe with 5% NaN -> kept
    train, test = np.arange(0, 180), np.arange(180, 240)
    prep = MethPrep(0.10, 4, block=7).fit(X, train, y[train])
    assert 11 not in prep.cols_ and 5 in prep.cols_
    assert prep.n_kept_prevalence_ < p
    Ptr, Pte = prep.transform(X, train), prep.transform(X, test)
    assert not np.isnan(Ptr).any() and not np.isnan(Pte).any()
    assert np.allclose(Ptr.mean(axis=0), 0, atol=1e-6)    # standardised on TRAIN rows
    # changing held-out rows must not change the fit
    X2 = X.copy()
    X2[test] = 99.0
    prep2 = MethPrep(0.10, 4, block=7).fit(X2, train, y[train])
    assert np.array_equal(prep.cols_, prep2.cols_) and np.allclose(prep.mu_, prep2.mu_)


def test_clinprep_and_stackprep():
    from mosurv.features import ClinPrep, StackPrep
    rng = np.random.default_rng(3)
    n = 150
    C = rng.normal(size=(n, 5))
    C[rng.random((n, 5)) < 0.05] = np.nan
    X1 = rng.normal(2.0, 1.0, size=(n, 40))
    y = make_y(rng.exponential(np.exp(-X1[:, 9])), np.ones(n, bool))
    from functools import partial
    parts = [("clin", ClinPrep), ("rna", partial(OmicsPrep, 0.0, 0.5, 4))]
    offsets = {"clin": 0, "rna": 5}
    train = np.arange(100)
    sp = StackPrep(parts, offsets).fit({"clin": C, "rna": X1}, train, y[train])
    out = sp.transform({"clin": C, "rna": X1}, np.arange(100, 150))
    assert out.shape == (50, 9) and not np.isnan(out).any()
    assert list(sp.penalty_factor_) == [0.0] * 5 + [1.0] * 4
    assert (sp.cols_[:5] == np.arange(5)).all() and 5 + 9 in sp.cols_       # omics block offset applied
