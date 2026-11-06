"""Milestone 5: omics-only elastic-net Cox in nested repeated CV. Usage: 05_single_omics.py rna|mirna"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from functools import partial

from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.util import Surv

from mosurv.evaluate import nb_ci
from mosurv.features import MethPrep, OmicsPrep
from mosurv.models import nested_fold

layer = sys.argv[1]
cfg = yaml.safe_load(open("config/config.yaml"))
so, lc = cfg["single_omics"], cfg["single_omics"]["layers"][layer]
interim, tables, models_dir = Path(cfg["paths"]["interim"]), Path(cfg["paths"]["tables"]), Path(cfg["paths"]["models"])
tables.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
F = pd.read_parquet(interim / lc["file"])[core.index]        # features x patients, cohort order
X = np.ascontiguousarray(F.to_numpy().T, dtype=np.float32)   # patients x features, C-order
names = np.asarray(F.index)
del F
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)
if layer == "meth":
    prep_factory = partial(MethPrep, lc["max_missing"], lc["k_screen"])
    filt = f"drop probes with >{lc['max_missing']:.0%} missing in train, median-impute"
else:
    prep_factory = partial(OmicsPrep, lc["prevalence_min_value"], lc["prevalence_min_frac"], lc["k_screen"])
    filt = f"prevalence: >{lc['prevalence_min_value']} in >={lc['prevalence_min_frac']:.0%} of train"
print(f"{layer}: X={X.shape} events={int(event.sum())} k_screen={lc['k_screen']}; {filt}")

cvc = cfg["evaluation"]["outer_cv"]
cv = RepeatedStratifiedKFold(n_splits=cvc["n_splits"], n_repeats=cvc["n_repeats"], random_state=cfg["seed"])
splits = list(cv.split(X, event))
res = Parallel(n_jobs=so["n_jobs"], verbose=5)(
    delayed(nested_fold)(X, y, tr, te, prep_factory, so["l1_grid"], so["inner_folds"],
                         so["n_alphas"], so["alpha_min_ratio"], cfg["seed"] + i, cfg["evaluation"]["tau_days"])
    for i, (tr, te) in enumerate(splits))

ok = [(i, r) for i, r in enumerate(res) if not r["failed"]]
print(f"\nfolds ok: {len(ok)}/{len(res)}; fits that errored inside folds: {sum(r['n_fail'] for _, r in ok)}")
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sel_cols", "risk", "failed")} | {"fold": i} for i, r in ok])
df.to_csv(tables / f"single_omics_{layer}_folds.tsv", sep="\t", index=False)

n_tr, n_te = len(splits[0][0]), len(splits[0][1])
print(f"\n== {layer}: omics-only elastic-net Cox, repeated {cvc['n_splits']}x{cvc['n_repeats']} CV, nested tuning")
for m in ("harrell", "uno"):
    mean, lo, hi, _, _ = nb_ci(df[m].to_numpy(), n_tr, n_te)
    print(f"  {m:8s} {mean:.3f} [{lo:.3f}, {hi:.3f}] (SD {df[m].std():.3f})")
print(f"  inner-CV C (selected) mean {df.inner_c.mean():.3f}; l1 chosen: {dict(df.l1.value_counts())}")
print(f"  non-zero features per fold: median {df.n_nonzero.median():.0f} (range {df.n_nonzero.min()}-{df.n_nonzero.max()}); "
      f"features passing filter: median {df.n_prev.median():.0f}")

sel = pd.Series([names[c] for _, r in ok for c in r["sel_cols"]]).value_counts()
freq = (sel / len(ok)).rename("selection_freq").to_frame()
freq.index.name = "feature"
freq.to_csv(tables / f"single_omics_{layer}_selection_freq.tsv", sep="\t")
print(f"  features selected in >=50% of outer folds: {int((freq.selection_freq >= 0.5).sum())}; "
      f">=20%: {int((freq.selection_freq >= 0.2).sum())}; ever: {len(freq)}")
print("  top 10:", ", ".join(f"{k}({v:.2f})" for k, v in freq.selection_freq.head(10).items()))
# out-of-fold risk scores (one row per repeat x patient) for later comparison / stacking
oof = pd.DataFrame(index=core.index)
for rep in range(cvc["n_repeats"]):
    col = np.full(len(core), np.nan)
    for i, r in ok:
        if i // cvc["n_splits"] == rep:
            col[splits[i][1]] = r["risk"]
    oof[f"repeat{rep}"] = col
oof.to_parquet(models_dir / f"oof_risk_{layer}.parquet")
print(f"saved results/tables/single_omics_{layer}_*.tsv and results/models/oof_risk_{layer}.parquet")
