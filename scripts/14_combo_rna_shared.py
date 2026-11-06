"""Milestone 8b: clinical (unpenalised) + penalised RNA restricted to the gene pool shared by TCGA and both arrays.
Identical to 07 'rna' except the RNA feature pool; reports the cost of the restriction (paired vs full-pool clinical+RNA)."""
import sys
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
from mosurv.models import nested_fold

combo = "rnashared"
layers = ["rna"]
cfg = yaml.safe_load(open("config/config.yaml"))
so = cfg["single_omics"]
interim, tables, models_dir = Path(cfg["paths"]["interim"]), Path(cfg["paths"]["tables"]), Path(cfg["paths"]["models"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)

stage_ok = core["stage"].notna()
C = np.column_stack([
    core["age"].astype(float).to_numpy(),
    (core["gender"] == "MALE").astype(float).to_numpy(),
    *[np.where(stage_ok, (core["stage"] == s).astype(float), np.nan) for s in ("II", "III", "IV")]])
Xs, names = {"clin": C}, {"clin": np.array(cfg["integration"]["clinical_features"])}
factories = {"clin": ClinPrep}
gmap = pd.read_csv(tables / "shared_gene_map.tsv", sep="\t")
for l in layers:
    lc = so["layers"][l]
    F = pd.read_parquet(interim / lc["file"])[core.index]
    F = F.loc[gmap["ensembl"].to_numpy()]
    F.index = gmap["symbol"].to_numpy()
    Xs[l] = np.ascontiguousarray(F.to_numpy().T, dtype=np.float32)
    names[l] = np.asarray(F.index)
    del F
    factories[l] = (partial(MethPrep, lc["max_missing"], lc["k_screen"]) if l == "meth"
                    else partial(OmicsPrep, lc["prevalence_min_value"], lc["prevalence_min_frac"], lc["k_screen"]))
order = ["clin"] + layers
offsets, o = {}, 0
for k in order:
    offsets[k] = o
    o += Xs[k].shape[1]
names_all = np.concatenate([names[k] for k in order])
prep_factory = partial(StackPrep, [(k, factories[k]) for k in order], offsets)
print(f"clinical + {combo}: blocks {[(k, Xs[k].shape[1]) for k in order]}; events={int(event.sum())}; "
      f"clinical block unpenalised, omics penalised (l1 grid {so['l1_grid']})")

cvc = cfg["evaluation"]["outer_cv"]
cv = RepeatedStratifiedKFold(n_splits=cvc["n_splits"], n_repeats=cvc["n_repeats"], random_state=cfg["seed"])
splits = list(cv.split(np.zeros(len(core)), event))
res = Parallel(n_jobs=so["n_jobs"], verbose=5)(
    delayed(nested_fold)(Xs, y, tr, te, prep_factory, so["l1_grid"], so["inner_folds"], so["n_alphas"],
                         so["alpha_min_ratio"], cfg["seed"] + i, cfg["evaluation"]["tau_days"])
    for i, (tr, te) in enumerate(splits))
ok = [(i, r) for i, r in enumerate(res) if not r["failed"]]
print(f"\nfolds ok: {len(ok)}/{len(res)}; fits that errored inside folds: {sum(r['n_fail'] for _, r in ok)}")
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("sel_cols", "risk", "failed")} | {"fold": i} for i, r in ok])
df.to_csv(tables / f"combo_{combo}_folds.tsv", sep="\t", index=False)

n_tr, n_te = len(splits[0][0]), len(splits[0][1])
clin = pd.read_csv(tables / "clinical_baseline_cv_folds.tsv", sep="\t")
base = {m: clin[clin.model == "age+sex+stage"].set_index("fold")[m] for m in ("harrell", "uno")}
print(f"\n== clinical + {combo}: repeated {cvc['n_splits']}x{cvc['n_repeats']} CV, nested tuning, identical folds")
for m in ("harrell", "uno"):
    mean, lo, hi, _, _ = nb_ci(df[m].to_numpy(), n_tr, n_te)
    d = (df.set_index("fold")[m] - base[m].reindex(df.fold)).to_numpy()
    dm, dlo, dhi, _, p = nb_ci(d, n_tr, n_te)
    print(f"  {m:8s} {mean:.3f} [{lo:.3f}, {hi:.3f}] | clinical-only {base[m].mean():.3f} | "
          f"diff {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}] p={p:.3f}")
    if m == "uno":
        print(f"  >> internal criterion (Uno diff CI excludes 0 and > 0): {'MET' if dlo > 0 else 'NOT met'}")
print(f"  inner-CV C (selected) mean {df.inner_c.mean():.3f}; l1 chosen: {dict(df.l1.value_counts())}")
print(f"  penalised non-zero omics features per fold: median {df.n_nonzero.median():.0f} "
      f"(range {df.n_nonzero.min()}-{df.n_nonzero.max()})")
sel = pd.Series([names_all[c] for _, r in ok for c in r["sel_cols"]]).value_counts()
freq = (sel / len(ok)).rename("selection_freq").to_frame()
freq.index.name = "feature"
freq.to_csv(tables / f"combo_{combo}_selection_freq.tsv", sep="\t")
print(f"  omics features selected in >=50% of folds: {int((freq.selection_freq >= 0.5).sum())}; "
      f">=20%: {int((freq.selection_freq >= 0.2).sum())}; ever: {len(freq)}")
print("  top 10:", ", ".join(f"{k}({v:.2f})" for k, v in freq.selection_freq.head(10).items()))
oof = pd.DataFrame(index=core.index)
for rep in range(cvc["n_repeats"]):
    col = np.full(len(core), np.nan)
    for i, r in ok:
        if i // cvc["n_splits"] == rep:
            col[splits[i][1]] = r["risk"]
    oof[f"repeat{rep}"] = col
oof.to_parquet(models_dir / f"oof_risk_combo_{combo}.parquet")
print(f"saved results/tables/combo_{combo}_*.tsv and results/models/oof_risk_combo_{combo}.parquet")

full = pd.read_csv(tables / "combo_rna_folds.tsv", sep="\t").set_index("fold")
print("\n== cost of restricting RNA to the shared pool (paired, same folds): shared minus full-pool clinical+RNA")
for m in ("harrell", "uno"):
    d = (df.set_index("fold")[m] - full[m].reindex(df.fold)).to_numpy()
    dm, dlo, dhi, _, p = nb_ci(d, n_tr, n_te)
    print(f"  {m:8s} full {full[m].mean():.3f} -> shared {df[m].mean():.3f}; diff {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")
