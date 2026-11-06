"""M8e: score the FROZEN signature_v1 on the external cohorts. No refitting, no tuning, no threshold search.
Run once; re-running requires --rerun (the result table is the record)."""
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import chi2
from sksurv.util import Surv

from mosurv.evaluate import harrell_c, uno_c
from mosurv.signature import score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_SHA = "704878c704c0b13e3c1f4e3fb3e7671bf411eae7e13611f19342a47c8cd48041"
cfg = yaml.safe_load(open("config/config.yaml"))
models_dir, tables, figs = Path(cfg["paths"]["models"]), Path(cfg["paths"]["tables"]), Path(cfg["paths"]["figures"])
processed, interim = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["interim"])
out_tab = tables / "external_validation.tsv"
if out_tab.exists() and "--rerun" not in sys.argv:
    sys.exit(f"{out_tab} already exists: the external test was already run. Use --rerun only to reproduce it unchanged.")
raw = (models_dir / "signature_v1.json").read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA, "signature file differs from the frozen version"
sig = json.loads(raw)
cut, tau = sig["cutoff"]["tcga_median_lp"], cfg["evaluation"]["tau_days"]
rng = np.random.default_rng(cfg["seed"])
print(f"signature_v1 verified (sha256 {EXPECTED_SHA[:12]}...); genes {sig['rna']['genes']}; frozen cutoff {cut:.3f}\n")


def cox_row(T, E, x, extra=None):
    d = pd.DataFrame({"T": T, "E": E, "x": x})
    if extra is not None:
        d = d.join(extra.reset_index(drop=True))   # positional alignment (d has a RangeIndex)
    f = CoxPHFitter().fit(d, "T", "E")
    s = f.summary.loc["x"]
    return float(s["coef"]), float(np.exp(s["coef"])), float(s["exp(coef) lower 95%"]), float(s["exp(coef) upper 95%"]), float(s["p"])


def evaluate(name, T, E, lp, lp_c, lp_r, extra=None, adj_label=""):
    y = Surv.from_arrays(event=E.astype(bool), time=T)
    rows, n = [], len(T)
    print(f"--- {name}: n={n}, events={int(E.sum())}")
    for label, v in (("full signature", lp), ("RNA component only", lp_r), ("clinical component only", lp_c)):
        c = harrell_c(E.astype(bool), T, v)
        bs = []
        for _ in range(1000):
            i = rng.integers(0, n, n)
            if E[i].sum() > 0:
                bs.append(harrell_c(E[i].astype(bool), T[i], v[i]))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        try:
            u = uno_c(y, y, v, tau)
        except Exception:
            u = np.nan
        _, hr, hlo, hhi, p = cox_row(T, E, v / v.std())
        print(f"  {label:24s} Harrell C {c:.3f} [{lo:.3f}, {hi:.3f}]; Uno C(5y) {u:.3f}; HR per SD {hr:.2f} [{hlo:.2f}, {hhi:.2f}] p={p:.3g}")
        rows.append(dict(cohort=name, score=label, harrell=c, harrell_lo=lo, harrell_hi=hi, uno5=u, hr_per_sd=hr, hr_lo=hlo, hr_hi=hhi, p=p))
    slope, _, _, _, _ = cox_row(T, E, lp)
    f = CoxPHFitter().fit(pd.DataFrame({"T": T, "E": E, "x": lp}), "T", "E")
    sl_ci = f.confidence_intervals_.iloc[0].to_numpy()
    print(f"  calibration slope of the full score {slope:.2f} [{sl_ci[0]:.2f}, {sl_ci[1]:.2f}]")
    hi_f = lp > cut
    hi_m = lp > np.median(lp)
    res = {}
    for lab, g in (("frozen TCGA cutoff", hi_f), ("within-cohort median", hi_m)):
        if g.sum() < 5 or (~g).sum() < 5:
            print(f"  high vs low, {lab}: groups too small ({int(g.sum())} high / {int((~g).sum())} low)")
            continue
        _, hr, hlo, hhi, p = cox_row(T, E, g.astype(float))
        lr = logrank_test(T[g], T[~g], E[g], E[~g]).p_value
        print(f"  high vs low, {lab}: {int(g.sum())} high / {int((~g).sum())} low; HR {hr:.2f} [{hlo:.2f}, {hhi:.2f}] Cox p={p:.3g}, log-rank p={lr:.3g}")
        res[lab] = (hr, hlo, hhi, p)
        rows.append(dict(cohort=name, score=f"high vs low ({lab})", hr_per_sd=hr, hr_lo=hlo, hr_hi=hhi, p=p))
    d2 = pd.DataFrame({"T": T, "E": E, "clin": lp_c, "rna": lp_r})
    f0, f1 = CoxPHFitter().fit(d2[["T", "E", "clin"]], "T", "E"), CoxPHFitter().fit(d2, "T", "E")
    print(f"  added value of RNA component over clinical component: coef {f1.params_['rna']:+.2f}, LR p={chi2.sf(2 * (f1.log_likelihood_ - f0.log_likelihood_), 1):.3g}")
    if extra is not None:
        _, hr, hlo, hhi, p = cox_row(T, E, lp / lp.std(), extra)
        print(f"  HR per SD adjusted for {adj_label}: {hr:.2f} [{hlo:.2f}, {hhi:.2f}] p={p:.3g}")
    crit = (rows[0]["harrell"] > 0.60 and rows[0]["harrell_lo"] > 0.50,
            "frozen TCGA cutoff" in res and res["frozen TCGA cutoff"][0] > 1 and res["frozen TCGA cutoff"][3] < 0.05,
            abs(slope - 1) <= 0.3)
    print(f"  criteria: C>0.60 & lower CI>0.50: {'MET' if crit[0] else 'NOT met'}; HR>1 & p<0.05 (frozen cutoff): {'MET' if crit[1] else 'NOT met'}; "
          f"calibration slope within 1±0.3: {'yes' if crit[2] else 'no'}\n")
    return rows, hi_f


all_rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for k, (name, cf, ef) in enumerate((("GSE31210", "gse31210_cohort.parquet", "gse31210_gene_log2.parquet"),
                                     ("GSE68465", "gse68465_cohort.parquet", "gse68465_gene_log2.parquet"))):
    co = pd.read_parquet(processed / cf)
    E = pd.read_parquet(interim / ef)[co.index]
    s = score(sig, co[sig["clinical"]["features"]].to_numpy(dtype=float), E, "cohort")   # gene z-scores within the full cohort
    co = co.assign(lp=s["total"], lp_c=s["clinical"], lp_r=s["rna"])
    sets = [(f"{name} primary", co)]
    if name == "GSE31210":
        sets.append((f"{name} sensitivity (204, flagged excluded)", co[~co["flagged_exclude"]]))
    else:
        sets[0] = (f"{name} primary (adenocarcinoma with follow-up)", co[co["os_time_days"].notna()])
    for nm, d in sets:
        extra = adj = None
        if name == "GSE68465":
            extra, adj = d[["adjuvant_chemo"]].astype(float), "adjuvant chemotherapy"
        rows, hi_f = evaluate(nm, d["os_time_days"].to_numpy(float), d["os_event"].to_numpy(int), d["lp"].to_numpy(),
                              d["lp_c"].to_numpy(), d["lp_r"].to_numpy(), extra, adj)
        all_rows += rows
        if nm.endswith("primary") or "primary (" in nm:
            ax = axes[k]
            for lab, g in (("low (below frozen cutoff)", ~hi_f), ("high (above frozen cutoff)", hi_f)):
                if g.sum():
                    KaplanMeierFitter().fit(d["os_time_days"].to_numpy()[g] / 365.25, d["os_event"].to_numpy()[g], label=f"{lab} n={int(g.sum())}").plot_survival_function(ax=ax)
            ax.set_title(nm); ax.set_xlabel("years"); ax.set_xlim(0, 10)
axes[0].set_ylabel("overall survival")
plt.tight_layout(); plt.savefig(figs / "external_km.png", dpi=150)
pd.DataFrame(all_rows).to_csv(out_tab, sep="\t", index=False)
print("saved results/tables/external_validation.tsv and results/figures/external_km.png")
