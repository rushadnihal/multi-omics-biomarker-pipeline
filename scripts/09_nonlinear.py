"""Milestone 6d: random survival forest / gradient-boosted survival comparators on the same folds."""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.util import Surv

from mosurv.evaluate import nb_ci
from mosurv.features import ClinPrep, MethPrep, OmicsPrep, StackPrep
from mosurv.models import nonlinear_fold

cfg = yaml.safe_load(open("config/config.yaml"))
so = cfg["single_omics"]
interim, tables, models_dir = Path(cfg["paths"]["interim"]), Path(cfg["paths"]["tables"]), Path(cfg["paths"]["models"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)
stage_ok = core["stage"].notna()
C = np.column_stack([core["age"].astype(float).to_numpy(), (core["gender"] == "MALE").astype(float).to_numpy(),
                     *[np.where(stage_ok, (core["stage"] == s).astype(float), np.nan) for s in ("II", "III", "IV")]])
Xs, names, factories = {"clin": C}, {"clin": np.array(cfg["integration"]["clinical_features"])}, {"clin": ClinPrep}
for l in ("rna", "mirna", "meth"):
    lc = so["layers"][l]
    F = pd.read_parquet(interim / lc["file"])[core.index]
    Xs[l] = np.ascontiguousarray(F.to_numpy().T, dtype=np.float32)
    names[l] = np.asarray(F.index)
    del F
    factories[l] = (partial(MethPrep, lc["max_missing"], lc["k_screen"]) if l == "meth"
                    else partial(OmicsPrep, lc["prevalence_min_value"], lc["prevalence_min_frac"], lc["k_screen"]))
order = ["clin", "rna", "mirna", "meth"]
offsets, o = {}, 0
for k in order:
    offsets[k] = o
    o += Xs[k].shape[1]
names_all = np.concatenate([names[k] for k in order])
prep_factory = partial(StackPrep, [(k, factories[k]) for k in order], offsets)
print("non-linear comparators; hyperparameters fixed in config:", {k: v for k, v in cfg["nonlinear"].items()})

cvc = cfg["evaluation"]["outer_cv"]
cv = RepeatedStratifiedKFold(n_splits=cvc["n_splits"], n_repeats=cvc["n_repeats"], random_state=cfg["seed"])
splits = list(cv.split(np.zeros(len(core)), event))
res = Parallel(n_jobs=so["n_jobs"], verbose=5)(
    delayed(nonlinear_fold)(Xs, y, tr, te, prep_factory, cfg["nonlinear"], cfg["seed"] + i,
                            cfg["evaluation"]["tau_days"], Xs["clin"].shape[1])
    for i, (tr, te) in enumerate(splits))
print(f"\nfolds ok: {sum(not r.get('failed', True) for r in res)}/{len(res)}")

n_tr, n_te = len(splits[0][0]), len(splits[0][1])
clin = pd.read_csv(tables / "clinical_baseline_cv_folds.tsv", sep="\t")
refs = {"clinical Cox": {m: clin[clin.model == "age+sex+stage"].set_index("fold")[m] for m in ("harrell", "uno")}}
f = pd.read_csv(tables / "combo_rna_folds.tsv", sep="\t").set_index("fold")
refs["early clinical+rna"] = {m: f[m] for m in ("harrell", "uno")}
models = ["RSF clinical", "RSF clinical+omics", "GBM clinical+omics"]
rows = [{"fold": i, "model": m, "harrell": r[m]["harrell"], "uno": r[m]["uno"]} for i, r in enumerate(res) for m in models]
df = pd.DataFrame(rows)
df.to_csv(tables / "nonlinear_folds.tsv", sep="\t", index=False)
print(f"\n== non-linear comparators, repeated {cvc['n_splits']}x{cvc['n_repeats']} CV, identical folds, no tuning")
for m in models:
    d0 = df[df.model == m].set_index("fold")
    for metric in ("harrell", "uno"):
        mean, lo, hi, _, _ = nb_ci(d0[metric].to_numpy(), n_tr, n_te)
        line = f"  {m:20s} {metric:8s} {mean:.3f} [{lo:.3f}, {hi:.3f}]"
        for rname, ref in refs.items():
            dm, dlo, dhi, _, p = nb_ci((d0[metric] - ref[metric].reindex(d0.index)).to_numpy(), n_tr, n_te)
            line += f" | vs {rname}: {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        print(line)
imp = pd.Series(0.0, index=names_all)
for r in res:
    imp.loc[names_all[np.asarray(r["cols"])]] += np.asarray(r["GBM clinical+omics"]["importance"]) / len(res)
print("\n  GBM mean feature importance, top 12:", ", ".join(f"{k}({v:.3f})" for k, v in imp.nlargest(12).items()))
