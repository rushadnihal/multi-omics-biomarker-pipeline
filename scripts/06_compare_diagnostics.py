"""Paired comparison vs clinical baseline + rule-9 diagnostics (is the risk score just stage / site?).

Uses only saved out-of-fold outputs; refits nothing. Works for every layer whose results exist.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold

from mosurv.evaluate import nb_ci
from mosurv.preprocess import r2_by_group

cfg = yaml.safe_load(open("config/config.yaml"))
tables, models = Path(cfg["paths"]["tables"]), Path(cfg["paths"]["models"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
qc = pd.read_csv(tables / "sample_qc.tsv", sep="\t", index_col=0, dtype={"tss": str, "tss_group": str}).reindex(core.index)
n = len(core)
cvc = cfg["evaluation"]["outer_cv"]
cv = RepeatedStratifiedKFold(n_splits=cvc["n_splits"], n_repeats=cvc["n_repeats"], random_state=cfg["seed"])
splits = list(cv.split(np.zeros(n), core["os_event"].astype(int).to_numpy()))
n_tr, n_te = len(splits[0][0]), len(splits[0][1])
fold_of = np.zeros((cvc["n_repeats"], n), dtype=int)
for i, (_, te) in enumerate(splits):
    fold_of[i // cvc["n_splits"], te] = i

clin = pd.read_csv(tables / "clinical_baseline_cv_folds.tsv", sep="\t")
base = {m: clin[clin.model == "age+sex+stage"].set_index("fold")[m] for m in ("harrell", "uno")}
layers = [l for l in ("rna", "mirna", "meth") if (tables / f"single_omics_{l}_folds.tsv").exists()]

print("== Paired difference vs clinical baseline (age+sex+stage), identical folds, Nadeau-Bengio 95% CI")
for l in layers:
    f = pd.read_csv(tables / f"single_omics_{l}_folds.tsv", sep="\t").set_index("fold")
    for m in ("harrell", "uno"):
        d = (f[m] - base[m].reindex(f.index)).to_numpy()
        mean, lo, hi, _, p = nb_ci(d, n_tr, n_te)
        print(f"  {l:6s} {m:8s} omics {f[m].mean():.3f} vs clinical {base[m].mean():.3f}: diff {mean:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.3f}")

stage_num = core["stage"].map({"I": 1, "II": 2, "III": 3, "IV": 4})
k = qc["tss_group"].nunique()
print(f"\n== Rule-9 diagnostics on out-of-fold risk (z-scored within each test fold; median over {cvc['n_repeats']} repeats)")
print(f"   chance level for R2 by TSS group ~ {(k - 1) / (n - 1):.3f}")
for l in layers:
    oof = pd.read_parquet(models / f"oof_risk_{l}.parquet").reindex(core.index)
    rows = []
    for r in range(cvc["n_repeats"]):
        risk = oof[f"repeat{r}"].to_numpy(dtype=float)
        z = np.full(n, np.nan)
        for fid in np.unique(fold_of[r]):
            m = fold_of[r] == fid
            z[m] = (risk[m] - np.nanmean(risk[m])) / (np.nanstd(risk[m]) + 1e-12)
        d = pd.DataFrame({"risk": z, "os_time": core["os_time"].to_numpy(float), "os_event": core["os_event"].to_numpy(int),
                          "stage_II": (core["stage"] == "II").to_numpy(float), "stage_III": (core["stage"] == "III").to_numpy(float),
                          "stage_IV": (core["stage"] == "IV").to_numpy(float), "tss": qc["tss_group"].to_numpy()})
        d = d[core["stage"].notna().to_numpy() & d["risk"].notna()]
        out = dict(rho_stage=spearmanr(z, stage_num, nan_policy="omit")[0], r2_tss=r2_by_group(z, qc["tss_group"]))
        for tag, cols, strata in [("raw", ["risk"], None), ("adj_stage", ["risk", "stage_II", "stage_III", "stage_IV"], None),
                                  ("adj_stage_strat_tss", ["risk", "stage_II", "stage_III", "stage_IV"], ["tss"])]:
            dd = d[cols + ["os_time", "os_event"] + (strata or [])]
            s = CoxPHFitter().fit(dd, "os_time", "os_event", strata=strata).summary.loc["risk"]
            out[f"hr_{tag}"], out[f"p_{tag}"] = s["exp(coef)"], s["p"]
        rows.append(out)
    R = pd.DataFrame(rows).median()
    print(f"  {l:6s} rho(risk, stage)={R.rho_stage:+.2f}  R2(risk by TSS)={R.r2_tss:.3f}")
    print(f"         HR per SD of risk: raw {R.hr_raw:.2f} (p={R.p_raw:.2g}) | adj stage {R.hr_adj_stage:.2f} (p={R.p_adj_stage:.2g})"
          f" | adj stage + strat TSS {R.hr_adj_stage_strat_tss:.2f} (p={R.p_adj_stage_strat_tss:.2g})")
