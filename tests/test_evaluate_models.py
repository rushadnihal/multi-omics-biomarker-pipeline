import numpy as np
import pandas as pd
import pytest

from mosurv.evaluate import nb_ci


def test_nb_ci_basic():
    vals = np.array([0.6, 0.62, 0.58, 0.61, 0.59] * 5)
    m, lo, hi, se, p = nb_ci(vals, n_train=350, n_test=90)
    assert lo < m < hi and np.isclose(m, vals.mean())
    # corrected SE must exceed the naive SE
    assert se > vals.std(ddof=1) / np.sqrt(len(vals))
    m0, _, _, _, p0 = nb_ci(np.zeros(25) + np.random.default_rng(0).normal(0, 1e-3, 25), 350, 90)
    assert p0 > 0.05


def test_clinical_cox_fits():
    pytest.importorskip("sksurv")
    from sksurv.util import Surv

    from mosurv.models import clinical_cox
    rng = np.random.default_rng(1)
    n = 120
    X = pd.DataFrame({"age": rng.normal(65, 9, n), "gender": rng.choice(["MALE", "FEMALE"], n).astype(object),
                      "stage": rng.choice(["I", "II", "III", np.nan], n).astype(object),
                      "pack_years": np.where(rng.random(n) < 0.3, np.nan, rng.gamma(2, 15, n))})
    t = rng.exponential(1000, n) + 10
    y = Surv.from_arrays(event=rng.random(n) < 0.6, time=t)
    pipe = clinical_cox(["age", "gender", "stage", "pack_years"]).fit(X, y)
    assert pipe.predict(X).shape == (n,)
