import numpy as np
import pandas as pd

from mosurv.signature import score


def _sig():
    return {"clinical": {"features": ["age", "male"], "median": [60.0, 0.0], "mean": [60.0, 0.5], "sd": [10.0, 0.5],
                         "coef": [0.3, 0.1]},
            "rna": {"genes": ["G1", "G2"], "mean": [0.0, 1.0], "sd": [1.0, 2.0], "coef": [0.5, -0.4]}}


def test_cohort_scaling_ignores_platform_affine_shift():
    rng = np.random.default_rng(1)
    E = pd.DataFrame(rng.normal(size=(2, 50)), index=["G1", "G2"])
    clin = np.column_stack([rng.normal(60, 10, 50), rng.integers(0, 2, 50)])
    a = score(_sig(), clin, E)
    E2 = E.copy()
    E2.loc["G1"] = 3.0 * E.loc["G1"] + 7.0
    E2.loc["G2"] = 0.5 * E.loc["G2"] - 2.0
    b = score(_sig(), clin, E2)
    assert np.allclose(a["rna"], b["rna"], atol=1e-6)
    assert np.allclose(a["total"], a["clinical"] + a["rna"])


def test_frozen_scaling_and_missing_clinical():
    E = pd.DataFrame([[1.0, 2.0], [3.0, 5.0]], index=["G1", "G2"])
    clin = np.array([[np.nan, 1.0], [70.0, np.nan]])
    s = score(_sig(), clin, E, gene_scaling="frozen")
    exp_c = np.array([(0 * 0.3) + ((1 - 0.5) / 0.5) * 0.1, (1.0 * 0.3) + ((0 - 0.5) / 0.5) * 0.1])
    assert np.allclose(s["clinical"], exp_c)
    exp_r = np.array([0.5 * 1.0 - 0.4 * (3.0 - 1.0) / 2.0, 0.5 * 2.0 - 0.4 * (5.0 - 1.0) / 2.0])
    assert np.allclose(s["rna"], exp_r)
