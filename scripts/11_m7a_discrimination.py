"""M7a: discrimination of the frozen candidates from out-of-fold risks: Uno C, time-dependent AUC (1/3/5y),
patient-bootstrap CIs (paired differences), KM + HR for median-split risk groups. Exploratory internal view;
the real frozen cutoff is set in M8."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from mosurv.evaluate import uno_c

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = yaml.safe_load(open("config/config.yaml"))
tables, figs, models = Path(cfg["paths"]["tables"]), Path(cfg["paths"]["figures"]), Path(cfg["paths"]["models"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)
n, tau = len(core), cfg["evaluation"]["tau_days"]
stage_ok = core["stage"].notna()
C = np.column_stack([core["age"].astype(float).to_numpy(), (core["gender"] == "MALE").astype(float).to_numpy(),
                     *[np.where(stage_ok, (core["stage"] == s).astype(float), np.nan) for s in ("II", "III", "IV")]])

cvc = cfg["evaluation"]["outer_cv"]
K, R = cvc["n_splits"], cvc["n_repeats"]
splits = list(RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=cfg["seed"]).split(np.zeros(n), event))
oof_clin = np.zeros((n, R))
for i, (tr, te) in enumerate(splits):  # same folds, same prep as the clinical baseline
    med = np.nanmedian(C[tr], 0)
    A, B = np.where(np.isnan(C[tr]), med, C[tr]), np.where(np.isnan(C[te]), med, C[te])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    oof_clin[te, i // K] = CoxPHSurvivalAnalysis(alpha=1e-4).fit((A - mu) / sd, y[tr]).predict((B - mu) / sd)
rna = pd.read_parquet(models / "oof_risk_combo_rna.parquet").reindex(core.index)
assert not rna.isna().any().any(), "OOF index does not align with cohort"
RISK = {"clinical-only": oof_clin, "clinical + RNA": rna.to_numpy()}

TIMES = np.array([365.0, 3 * 365.0, 5 * 365.0])
NAMES = ["Uno C (5y)", "AUC 1y", "AUC 3y", "AUC 5y"]


def metrics(idx, M):
    ys, out = y[idx], []
    for r in range(R):
        try:
            rk = M[idx, r]
            out.append([uno_c(ys, ys, rk, tau), *cumulative_dynamic_auc(ys, ys, rk, TIMES)[0]])
        except Exception:
            out.append([np.nan] * 4)
    return np.nanmean(out, 0)


allidx = np.arange(n)
point = {k: metrics(allidx, M) for k, M in RISK.items()}
rng = np.random.default_rng(cfg["seed"])
draws = [rng.integers(0, n, n) for _ in range(300)]
boot = {k: np.array(Parallel(n_jobs=cfg["single_omics"]["n_jobs"])(delayed(metrics)(d, M) for d in draws)) for k, M in RISK.items()}
rows = []
print(f"== discrimination from OOF risks (mean over {R} repeats; 95% patient-bootstrap CI, 300 draws)")
for k in RISK:
    for j, nm in enumerate(NAMES):
        lo, hi = np.nanpercentile(boot[k][:, j], [2.5, 97.5])
        rows.append(dict(model=k, metric=nm, est=point[k][j], lo=lo, hi=hi))
        print(f"  {k:15s} {nm:11s} {point[k][j]:.3f} [{lo:.3f}, {hi:.3f}]")
d = boot["clinical + RNA"] - boot["clinical-only"]
print("  paired difference (clinical + RNA minus clinical-only):")
for j, nm in enumerate(NAMES):
    lo, hi = np.nanpercentile(d[:, j], [2.5, 97.5])
    est = point["clinical + RNA"][j] - point["clinical-only"][j]
    rows.append(dict(model="difference", metric=nm, est=est, lo=lo, hi=hi))
    print(f"    {nm:11s} {est:+.3f} [{lo:+.3f}, {hi:+.3f}]")
harr = {k: np.mean([concordance_index_censored(event, time, M[:, r])[0] for r in range(R)]) for k, M in RISK.items()}
print("  pooled-OOF Harrell C (sanity vs fold means 0.657 / 0.686):", {k: round(v, 3) for k, v in harr.items()})
pd.DataFrame(rows).to_csv(tables / "m7a_discrimination.tsv", sep="\t", index=False)

print("\n== risk groups: median split of OOF risk (exploratory; frozen cutoff comes in M8)")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (k, M) in zip(axes, RISK.items()):
    z = (M - M.mean(0)) / M.std(0)
    hrs, ps = [], []
    for r in range(R):
        hi = M[:, r] > np.median(M[:, r])
        df = pd.DataFrame({"T": time, "E": event, "high": hi.astype(int)})
        hrs.append(float(np.exp(CoxPHFitter().fit(df, "T", "E").params_["high"])))
        ps.append(logrank_test(time[hi], time[~hi], event[hi], event[~hi]).p_value)
    s = z.mean(1)
    hi = s > np.median(s)
    df = pd.DataFrame({"T": time, "E": event, "high": hi.astype(int)})
    fit = CoxPHFitter().fit(df, "T", "E")
    ci = np.exp(fit.confidence_intervals_.iloc[0].to_numpy())
    print(f"  {k:15s} HR high vs low: per-repeat median {np.median(hrs):.2f} (range {min(hrs):.2f}-{max(hrs):.2f}); "
          f"repeat-averaged score {np.exp(fit.params_['high']):.2f} [{ci[0]:.2f}, {ci[1]:.2f}], log-rank p={logrank_test(time[hi], time[~hi], event[hi], event[~hi]).p_value:.3g}; "
          f"per-repeat log-rank p median {np.median(ps):.3g}")
    for lab, m in (("low risk", ~hi), ("high risk", hi)):
        KaplanMeierFitter().fit(time[m] / 365.25, event[m], label=f"{lab} (n={m.sum()})").plot_survival_function(ax=ax, ci_show=True)
    ax.set_title(k); ax.set_xlabel("years"); ax.set_xlim(0, 10)
axes[0].set_ylabel("overall survival")
plt.tight_layout(); plt.savefig(figs / "km_risk_groups_oof.png", dpi=150)
print("\nsaved results/tables/m7a_discrimination.tsv and results/figures/km_risk_groups_oof.png")
