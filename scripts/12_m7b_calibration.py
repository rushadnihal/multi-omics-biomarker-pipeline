"""M7b: calibration slope, added-value LR test, Brier/IBS vs null, calibration + decision curves, PH test, RMST.
All from out-of-fold linear predictors (one per patient per CV repeat). Survival probabilities come from a
1-covariate Cox on the OOF score, applied identically to both models, so comparisons between them are fair
but absolute calibration is 'recalibrated on OOF' (mildly optimistic). True external calibration is M8."""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import restricted_mean_survival_time
from scipy.stats import chi2
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, integrated_brier_score
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

matplotlib.use("Agg")
import matplotlib.pyplot as plt

cfg = yaml.safe_load(open("config/config.yaml"))
tables, figs, models = Path(cfg["paths"]["tables"]), Path(cfg["paths"]["figures"]), Path(cfg["paths"]["models"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)
n = len(core)
stage_ok = core["stage"].notna()
C = np.column_stack([core["age"].astype(float).to_numpy(), (core["gender"] == "MALE").astype(float).to_numpy(),
                     *[np.where(stage_ok, (core["stage"] == s).astype(float), np.nan) for s in ("II", "III", "IV")]])
K, R = cfg["evaluation"]["outer_cv"]["n_splits"], cfg["evaluation"]["outer_cv"]["n_repeats"]
splits = list(RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=cfg["seed"]).split(np.zeros(n), event))
oof_clin = np.zeros((n, R))
for i, (tr, te) in enumerate(splits):
    med = np.nanmedian(C[tr], 0)
    A, B = np.where(np.isnan(C[tr]), med, C[tr]), np.where(np.isnan(C[te]), med, C[te])
    mu, sd = A.mean(0), A.std(0) + 1e-12
    oof_clin[te, i // K] = CoxPHSurvivalAnalysis(alpha=1e-4).fit((A - mu) / sd, y[tr]).predict((B - mu) / sd)
pd.DataFrame(oof_clin, index=core.index, columns=[f"repeat{r}" for r in range(R)]).to_parquet(models / "oof_risk_clinical.parquet")
rna = pd.read_parquet(models / "oof_risk_combo_rna.parquet").reindex(core.index).to_numpy()
RISK = {"clinical-only": oof_clin, "clinical + RNA": rna}
GRID = np.linspace(365, 1825, 9)
T5 = 1825.0


def km_at(t, e, t0):
    ts, S = kaplan_meier_estimator(e, t)
    i = np.searchsorted(ts, t0, side="right") - 1
    return 1.0 if i < 0 else float(S[i])


def surv_at(m, X, times):
    arr = m.predict_survival_function(X, return_array=True)
    i = np.searchsorted(m.unique_times_, times, side="right") - 1
    return np.where(i[None, :] < 0, 1.0, arr[:, np.clip(i, 0, None)])


# 1) calibration slope of the OOF score (Cox coefficient on the raw linear predictor)
print("== calibration slope of OOF linear predictor (1 = well calibrated; <1 = over-fit)")
for k, M in RISK.items():
    sl = []
    for r in range(R):
        f = CoxPHFitter().fit(pd.DataFrame({"T": time, "E": event, "lp": M[:, r]}), "T", "E")
        sl.append((f.params_["lp"], *f.confidence_intervals_.iloc[0].to_numpy()))
    sl = np.array(sl)
    print(f"  {k:15s} slope median {np.median(sl[:, 0]):.2f} (range {sl[:, 0].min():.2f}-{sl[:, 0].max():.2f}); "
          f"median 95% CI [{np.median(sl[:, 1]):.2f}, {np.median(sl[:, 2]):.2f}]")

# 2) added value: LR test of adding the clinical+RNA OOF score to the clinical-only OOF score (1 df)
print("\n== added value: Cox(clinical OOF) vs Cox(clinical OOF + clinical+RNA OOF), LR test, 1 df")
ps, cs = [], []
for r in range(R):
    d0 = pd.DataFrame({"T": time, "E": event, "clin": oof_clin[:, r]})
    d1 = d0.assign(rna=rna[:, r])
    f0, f1 = CoxPHFitter().fit(d0, "T", "E"), CoxPHFitter().fit(d1, "T", "E")
    ps.append(chi2.sf(2 * (f1.log_likelihood_ - f0.log_likelihood_), 1))
    cs.append(f1.params_["rna"])
print(f"  LR p per repeat: median {np.median(ps):.3g} (range {min(ps):.3g}-{max(ps):.3g}); coef of RNA-model score {np.median(cs):+.2f}")

# 3) Brier / IBS vs no-covariate KM baseline
print("\n== Brier score (lower = better) and integrated Brier 1-5y, vs null KM model; mean over repeats")
ts0, S0 = kaplan_meier_estimator(event, time)
null = np.tile(S0[np.clip(np.searchsorted(ts0, GRID, side="right") - 1, 0, None)], (n, 1))
bn = brier_score(y, y, null, GRID)[1]
ibs_null = integrated_brier_score(y, y, null, GRID)
res = {}
for k, M in RISK.items():
    b, ib = [], []
    for r in range(R):
        m = CoxPHSurvivalAnalysis().fit(M[:, [r]], y)
        est = surv_at(m, M[:, [r]], GRID)
        b.append(brier_score(y, y, est, GRID)[1]); ib.append(integrated_brier_score(y, y, est, GRID))
    res[k] = (np.mean(b, 0), np.mean(ib))
    idx = [0, 4, 8]
    print(f"  {k:15s} Brier 1y/3y/5y: " + " / ".join(f"{res[k][0][i]:.3f}" for i in idx) +
          f"  IBS {res[k][1]:.3f}  (null: " + " / ".join(f"{bn[i]:.3f}" for i in idx) + f"; IBS {ibs_null:.3f})")

# 4) calibration + decision curve at 5y, using repeat-averaged standardised OOF score
print("\n== calibration at 5y (quintiles of predicted 5y event risk; recalibrated on OOF) and decision curve")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
pred = {}
for k, M in RISK.items():
    s = ((M - M.mean(0)) / M.std(0)).mean(1)
    m = CoxPHSurvivalAnalysis().fit(s[:, None], y)
    p = 1 - surv_at(m, s[:, None], np.array([T5]))[:, 0]
    pred[k] = p
    q = pd.qcut(p, 5, labels=False, duplicates="drop")
    px = [p[q == g].mean() for g in np.unique(q)]
    ob = [1 - km_at(time[q == g], event[q == g], T5) for g in np.unique(q)]
    ax[0].plot(px, ob, "o-", label=k)
    print(f"  {k:15s} predicted vs observed 5y event risk by quintile: " + ", ".join(f"{a:.2f}/{b:.2f}" for a, b in zip(px, ob)))
lim = [0, 1]
ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlabel("predicted 5y event risk"); ax[0].set_ylabel("observed (KM)"); ax[0].legend()
ax[0].set_title("calibration (OOF, recalibrated)")
th = np.arange(0.10, 0.71, 0.01)


def nb(p, pt):
    g = p > pt
    if g.sum() == 0:
        return 0.0
    rate = 1 - km_at(time[g], event[g], T5)
    return (rate * g.sum() - (1 - rate) * g.sum() * pt / (1 - pt)) / n


for k, p in pred.items():
    ax[1].plot(th, [nb(p, t) for t in th], label=k)
ax[1].plot(th, [nb(np.ones(n), t) for t in th], "gray", label="treat all")
ax[1].axhline(0, color="k", lw=1, label="treat none")
ax[1].set_ylim(-0.1, None); ax[1].set_xlabel("threshold probability (5y event risk)"); ax[1].set_ylabel("net benefit"); ax[1].legend()
ax[1].set_title("decision curve at 5y")
plt.tight_layout(); plt.savefig(figs / "m7b_calibration_dca.png", dpi=150)
for t in (0.3, 0.4, 0.5):
    print(f"  net benefit at threshold {t:.1f}: " + ", ".join(f"{k} {nb(p, t):.3f}" for k, p in pred.items()) + f", treat-all {nb(np.ones(n), t):.3f}")

# 5) PH test of the score and RMST (5y) between median-split risk groups
print("\n== proportional hazards (Schoenfeld, rank time) for the standardised score, and 5y RMST by risk group")
rng = np.random.default_rng(cfg["seed"])
for k, M in RISK.items():
    s = ((M - M.mean(0)) / M.std(0)).mean(1)
    df = pd.DataFrame({"T": time, "E": event, "score": s})
    f = CoxPHFitter().fit(df, "T", "E")
    p_ph = proportional_hazard_test(f, df, time_transform="rank").summary["p"].iloc[0]
    hi = s > np.median(s)

    def rmst_diff(ix):
        a, b = KaplanMeierFitter(), KaplanMeierFitter()
        a.fit(time[ix & ~hi], event[ix & ~hi]); b.fit(time[ix & hi], event[ix & hi])
        return restricted_mean_survival_time(a, t=T5) - restricted_mean_survival_time(b, t=T5)
    d0 = rmst_diff(np.ones(n, bool))
    bs = [rmst_diff(np.isin(np.arange(n), rng.integers(0, n, n))) for _ in range(200)]
    lo, up = np.percentile(bs, [2.5, 97.5])
    print(f"  {k:15s} PH p={p_ph:.3f}; HR per SD {np.exp(f.params_['score']):.2f}; "
          f"5y RMST low-minus-high {d0:.0f} days [{lo:.0f}, {up:.0f}]")
print("\nsaved results/figures/m7b_calibration_dca.png and results/models/oof_risk_clinical.parquet")
