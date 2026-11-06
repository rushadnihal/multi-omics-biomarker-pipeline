"""Milestone 6b: late integration (stacking). Per-layer tuned elastic-net scores + clinical -> small Cox."""
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.util import Surv

from mosurv.evaluate import nb_ci
from mosurv.features import MethPrep, OmicsPrep
from mosurv.models import late_fold

cfg = yaml.safe_load(open("config/config.yaml"))
so = cfg["single_omics"]
interim, tables, models_dir = Path(cfg["paths"]["interim"]), Path(cfg["paths"]["tables"]), Path(cfg["paths"]["models"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)
stage_ok = core["stage"].notna()
C = np.column_stack([core["age"].astype(float).to_numpy(), (core["gender"] == "MALE").astype(float).to_numpy(),
                     *[np.where(stage_ok, (core["stage"] == s).astype(float), np.nan) for s in ("II", "III", "IV")]])
Xs, factories = {"clin": C}, {}
for l in ("rna", "mirna", "meth"):
    lc = so["layers"][l]
    F = pd.read_parquet(interim / lc["file"])[core.index]
    Xs[l] = np.ascontiguousarray(F.to_numpy().T, dtype=np.float32)
    del F
    factories[l] = (partial(MethPrep, lc["max_missing"], lc["k_screen"]) if l == "meth"
                    else partial(OmicsPrep, lc["prevalence_min_value"], lc["prevalence_min_frac"], lc["k_screen"]))
print(f"late integration: layers {list(factories)} + clinical (age, male, stage II/III/IV) stacked in a small Cox; events={int(event.sum())}")

cvc = cfg["evaluation"]["outer_cv"]
cv = RepeatedStratifiedKFold(n_splits=cvc["n_splits"], n_repeats=cvc["n_repeats"], random_state=cfg["seed"])
splits = list(cv.split(np.zeros(len(core)), event))
res = Parallel(n_jobs=so["n_jobs"], verbose=5)(
    delayed(late_fold)(Xs, y, tr, te, factories, so["l1_grid"], so["inner_folds"], so["n_alphas"],
                       so["alpha_min_ratio"], cfg["seed"] + i, cfg["evaluation"]["tau_days"])
    for i, (tr, te) in enumerate(splits))
ok = [(i, r) for i, r in enumerate(res) if not r["failed"]]
print(f"\nfolds ok: {len(ok)}/{len(res)}; fits that errored inside folds: {sum(r['n_fail'] for _, r in ok)}")
df = pd.DataFrame([{"fold": i, "harrell": r["harrell"], "uno": r["uno"]} for i, r in ok])
df.to_csv(tables / "late_integration_folds.tsv", sep="\t", index=False)

n_tr, n_te = len(splits[0][0]), len(splits[0][1])
clin = pd.read_csv(tables / "clinical_baseline_cv_folds.tsv", sep="\t")
refs = {"clinical-only": {m: clin[clin.model == "age+sex+stage"].set_index("fold")[m] for m in ("harrell", "uno")}}
for name in ("rna", "all"):
    f = pd.read_csv(tables / f"combo_{name}_folds.tsv", sep="\t").set_index("fold")
    refs[f"early clinical+{name}"] = {m: f[m] for m in ("harrell", "uno")}
print(f"\n== late integration (stacking), repeated {cvc['n_splits']}x{cvc['n_repeats']} CV, identical folds")
for m in ("harrell", "uno"):
    mean, lo, hi, _, _ = nb_ci(df[m].to_numpy(), n_tr, n_te)
    print(f"  {m:8s} {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    for rname, ref in refs.items():
        d = (df.set_index("fold")[m] - ref[m].reindex(df.fold)).to_numpy()
        dm, dlo, dhi, _, p = nb_ci(d, n_tr, n_te)
        print(f"           vs {rname:22s} ({ref[m].mean():.3f}): diff {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}] p={p:.3f}")
    if m == "uno":
        d = (df.set_index("fold")[m] - refs["clinical-only"][m].reindex(df.fold)).to_numpy()
        print(f"  >> internal criterion (Uno diff vs clinical-only CI excludes 0 and > 0): "
              f"{'MET' if nb_ci(d, n_tr, n_te)[1] > 0 else 'NOT met'}")
cols = ["age", "male", "stage_II", "stage_III", "stage_IV"] + list(factories)
coefs = pd.DataFrame([r["coef"] for _, r in ok], columns=cols)
print("\n  stacker coefficients on standardised inputs (mean over folds; share of folds with positive sign):")
for c in cols:
    print(f"    {c:9s} {coefs[c].mean():+.3f}   positive in {(coefs[c] > 0).mean():.0%} of folds")
lh = pd.DataFrame([r["layer_harrell"] for _, r in ok]).mean()
print("  per-layer omics-only Harrell C inside this run:", {k: round(float(v), 3) for k, v in lh.items()})
oof = pd.DataFrame(index=core.index)
for rep in range(cvc["n_repeats"]):
    col = np.full(len(core), np.nan)
    for i, r in ok:
        if i // cvc["n_splits"] == rep:
            col[splits[i][1]] = r["risk"]
    oof[f"repeat{rep}"] = col
oof.to_parquet(models_dir / "oof_risk_late.parquet")
print("saved results/tables/late_integration_folds.tsv and results/models/oof_risk_late.parquet")
