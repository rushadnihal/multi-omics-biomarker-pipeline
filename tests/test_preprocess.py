import numpy as np
import pandas as pd

from mosurv.preprocess import autosomal_probes, beta_to_m, log2cpm_from_log2count1, pca_scores, r2_by_group


def test_log2cpm_roundtrip():
    counts = pd.DataFrame({"a": [10, 90, 0], "b": [200, 800, 0]}, index=["g1", "g2", "g3"], dtype=float)
    X = np.log2(counts + 1)
    out, lib = log2cpm_from_log2count1(X)
    assert np.allclose(lib.values, [100, 1000])
    assert np.isclose(out.loc["g1", "a"], np.log2(10 / 100 * 1e6 + 1), atol=1e-4)
    assert np.isclose(out.loc["g1", "b"], np.log2(200 / 1000 * 1e6 + 1), atol=1e-4)


def test_beta_to_m():
    m = beta_to_m(np.array([0.5, 0.75, np.nan, 0.0, 1.0]))
    assert np.isclose(m[0], 0) and np.isclose(m[1], np.log2(3), atol=1e-6) and np.isnan(m[2])
    assert np.isfinite(m[3]) and np.isfinite(m[4])


def test_r2_by_group():
    y = np.array([1.0, 1.0, 5.0, 5.0])
    assert np.isclose(r2_by_group(y, ["a", "a", "b", "b"]), 1.0)
    assert r2_by_group(y, ["a", "b", "a", "b"]) < 1e-9


def test_pca_rank1():
    rng = np.random.default_rng(0)
    v = rng.normal(size=20)
    F = pd.DataFrame(np.outer(rng.normal(size=50), v))
    _, var = pca_scores(F, k=2)
    assert var[0] > 0.999


def test_autosomal_probes():
    pm = pd.DataFrame({"#id": ["cg1", "cg2", "cg3", "cg4", "cg5"],
                       "chrom": ["chr1", "chrX", "chrY", None, "chr22"]})
    assert list(autosomal_probes(pm)) == ["cg1", "cg5"]
