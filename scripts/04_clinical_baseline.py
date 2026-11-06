"""Milestone 4: clinical baseline. Repeated CV C-index (Harrell, Uno) + descriptive Cox, PH test, KM by stage."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.util import Surv

from mosurv.evaluate import harrell_c, nb_ci, uno_c
from mosurv.models import clinical_cox

cfg = yaml.safe_load(open("config/config.yaml"))
tables, figs = Path(cfg["paths"]["tables"]), Path(cfg["paths"]["figures"])
tables.mkdir(parents=True, exist_ok=True)
figs.mkdir(parents=True, exist_ok=True)
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core").copy()

X = core[["age", "gender", "stage", "pack_years"]].copy()
for c in ("gender", "stage"):                 # NaN (not None) so imputers see missing values
    X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
X["age"] = X["age"].astype(float)
X["pack_years"] = pd.to_numeric(X["pack_years"], errors="coerce").astype(float)
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)
n = len(core)
print(f"core cohort: n={n}, OS events={int(event.sum())}, tau={cfg['evaluation']['tau_days']} d")

models = {
    "age+sex": ["age", "gender"],
    "stage": ["stage"],
    "age+sex+stage": ["age", "gender", "stage"],
    "age+sex+stage+smoking": ["age", "gender", "stage", "pack_years"],
}
cvc = cfg["evaluation"]["outer_cv"]
cv = RepeatedStratifiedKFold(n_splits=cvc["n_splits"], n_repeats=cvc["n_repeats"], random_state=cfg["seed"])
rows = []
for fold, (tr, te) in enumerate(cv.split(X, event)):
    for name, feats in models.items():
        pipe = clinical_cox(feats).fit(X.iloc[tr][feats], y[tr])
        risk = pipe.predict(X.iloc[te][feats])
        rows.append(dict(fold=fold, model=name, harrell=harrell_c(event[te], time[te], risk),
                         uno=uno_c(y[tr], y[te], risk, cfg["evaluation"]["tau_days"])))
res = pd.DataFrame(rows)
res.to_csv(tables / "clinical_baseline_cv_folds.tsv", sep="\t", index=False)
n_tr, n_te = len(tr), len(te)

print(f"\n== Repeated {cvc['n_splits']}x{cvc['n_repeats']} CV, identical folds for all models (n_train~{n_tr}, n_test~{n_te})")
print("mean [95% CI, Nadeau-Bengio] (SD across folds)")
summ = []
for metric in ("harrell", "uno"):
    for name in models:
        v = res[(res.model == name)][metric].to_numpy()
        m, lo, hi, se, _ = nb_ci(v, n_tr, n_te)
        summ.append(dict(metric=metric, model=name, mean=m, ci_lo=lo, ci_hi=hi, sd_folds=np.nanstd(v, ddof=1),
                         n_nan=int(np.isnan(v).sum())))
        print(f"  {metric:8s} {name:24s} {m:.3f} [{lo:.3f}, {hi:.3f}] (SD {np.nanstd(v, ddof=1):.3f})"
              f"{'  NaN folds: ' + str(int(np.isnan(v).sum())) if np.isnan(v).any() else ''}")
pd.DataFrame(summ).to_csv(tables / "clinical_baseline_cv_summary.tsv", sep="\t", index=False)

print("\n== Paired differences (same folds)")
piv = {m: res.pivot(index="fold", columns="model", values=m) for m in ("harrell", "uno")}
for a, b in [("age+sex+stage", "age+sex"), ("age+sex+stage+smoking", "age+sex+stage")]:
    for metric in ("harrell", "uno"):
        d = (piv[metric][a] - piv[metric][b]).to_numpy()
        m, lo, hi, _, p = nb_ci(d, n_tr, n_te)
        print(f"  {metric:8s} {a} - {b}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}")

# ---- descriptive full-data analyses (not used for model selection)
try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test

    d = core.dropna(subset=["stage", "age"]).copy()
    d["age_per10y"] = d["age"].astype(float) / 10
    d["male"] = (d["gender"] == "MALE").astype(float)
    for s in ("II", "III", "IV"):
        d[f"stage_{s}"] = (d["stage"] == s).astype(float)
    cols = ["age_per10y", "male", "stage_II", "stage_III", "stage_IV"]
    dd = d[cols + ["os_time", "os_event"]].astype(float)
    cph = CoxPHFitter().fit(dd, "os_time", "os_event")
    print(f"\n== Full-data Cox (n={len(dd)}, events={int(dd.os_event.sum())}; stage ref = I; descriptive)")
    print(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3).to_string())
    ph = proportional_hazard_test(cph, dd, time_transform="rank").summary[["test_statistic", "p"]]
    print("\n== Proportional-hazards test (Schoenfeld, rank time)")
    print(ph.round(3).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in ("I", "II", "III", "IV"):
        g = d[d.stage == s]
        KaplanMeierFitter().fit(g.os_time / 365.25, g.os_event, label=f"Stage {s} (n={len(g)})").plot_survival_function(ax=ax, ci_show=False)
    lr = multivariate_logrank_test(d.os_time, d.stage, d.os_event)
    ax.set_xlabel("Years")
    ax.set_ylabel("Overall survival")
    ax.set_title(f"TCGA-LUAD core cohort, OS by stage (log-rank p={lr.p_value:.2g})")
    fig.tight_layout()
    fig.savefig(figs / "km_stage.png", dpi=110)
    print(f"\nlog-rank across stages: chi2={lr.test_statistic:.1f}, p={lr.p_value:.3g}; saved results/figures/km_stage.png")
except Exception as e:  # keep the CV results even if a descriptive step fails
    print(f"\n[descriptive analyses failed: {type(e).__name__}: {e}]")
