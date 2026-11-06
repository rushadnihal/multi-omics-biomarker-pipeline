"""Metrics and uncertainty for repeated K-fold CV of survival models."""
from __future__ import annotations

import numpy as np
from scipy import stats


def nb_ci(values, n_train: int, n_test: int, level: float = 0.95):
    """Nadeau-Bengio corrected resampled mean, CI, SE and two-sided p (H0: mean == 0).

    values: per-fold scores (or per-fold paired differences) from repeated K-fold CV.
    The 1/J + n_test/n_train variance inflation accounts for overlapping training sets.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    J = len(v)
    m = float(v.mean())
    s2 = float(v.var(ddof=1))
    se = float(np.sqrt((1.0 / J + n_test / n_train) * s2))
    t = stats.t.ppf(0.5 + level / 2, J - 1)
    p = float(2 * (1 - stats.t.cdf(abs(m) / se, J - 1))) if se > 0 else float("nan")
    return m, m - t * se, m + t * se, se, p


def harrell_c(event, time, risk) -> float:
    from sksurv.metrics import concordance_index_censored
    return float(concordance_index_censored(np.asarray(event, bool), np.asarray(time, float), risk)[0])


def uno_c(y_train, y_test, risk, tau_days: float) -> float:
    """Uno's IPCW C-index truncated at tau (clipped inside follow-up); NaN if it cannot be computed."""
    from sksurv.metrics import concordance_index_ipcw
    tau = min(tau_days, float(y_train["time"].max()) - 1.0, float(y_test["time"].max()) - 1.0)
    try:
        return float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])
    except Exception:
        return float("nan")
