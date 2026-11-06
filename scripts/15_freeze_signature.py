"""M8c: refit clinical + shared-pool RNA elastic-net on ALL TCGA patients with the same tuning procedure that was
cross-validated, and freeze it to results/models/signature_v1.json BEFORE any external cohort is scored."""
import datetime
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sksurv.util import Surv

from mosurv.evaluate import harrell_c
from mosurv.features import ClinPrep, OmicsPrep, StackPrep
from mosurv.signature import fit_final, score

cfg = yaml.safe_load(open("config/config.yaml"))
so = cfg["single_omics"]
interim, tables, models_dir = Path(cfg["paths"]["interim"]), Path(cfg["paths"]["tables"]), Path(cfg["paths"]["models"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")
event, time = core["os_event"].astype(bool).to_numpy(), core["os_time"].astype(float).to_numpy()
y = Surv.from_arrays(event=event, time=time)
n = len(core)
stage_ok = core["stage"].notna()
feats = cfg["integration"]["clinical_features"]
C = np.column_stack([core["age"].astype(float).to_numpy(), (core["gender"] == "MALE").astype(float).to_numpy(),
                     *[np.where(stage_ok, (core["stage"] == s).astype(float), np.nan) for s in ("II", "III", "IV")]])
gmap = pd.read_csv(tables / "shared_gene_map.tsv", sep="\t")
lc = so["layers"]["rna"]
F = pd.read_parquet(interim / lc["file"])[core.index].loc[gmap["ensembl"].to_numpy()]
F.index = gmap["symbol"].to_numpy()
Xs = {"clin": C, "rna": np.ascontiguousarray(F.to_numpy().T, dtype=np.float32)}
sym = np.asarray(F.index)
offsets = {"clin": 0, "rna": C.shape[1]}
prep_factory = partial(StackPrep, [("clin", ClinPrep), ("rna", partial(OmicsPrep, lc["prevalence_min_value"], lc["prevalence_min_frac"], lc["k_screen"]))], offsets)

fit = fit_final(Xs, y, np.arange(n), prep_factory, so["l1_grid"], so["inner_folds"], so["n_alphas"], so["alpha_min_ratio"], cfg["seed"])
prep, coef = fit["prep"], fit["coef"]
pp = dict(prep.preps_)
ncl = len(pp["clin"].cols_)
rc = coef[ncl:]
nz = np.flatnonzero(rc)
genes = sym[pp["rna"].cols_[nz]]
lp = fit["lp"]
cut = float(np.median(lp))
sig = {
    "name": "signature_v1", "created": datetime.date.today().isoformat(), "seed": cfg["seed"],
    "model": "clinical (unpenalised) + elastic-net Cox on RNA genes shared by TCGA, GPL570 and GPL96",
    "tuning": {"l1_ratio": fit["l1"], "alpha": fit["alpha"], "inner_cv_harrell": fit["inner_c"],
               "k_screen": lc["k_screen"], "n_events": int(event.sum()), "n_patients": int(n), "pool_size": int(len(gmap))},
    "clinical": {"features": feats, "median": pp["clin"].med_.tolist(), "mean": pp["clin"].mu_.tolist(),
                 "sd": pp["clin"].sd_.tolist(), "coef": coef[:ncl].tolist()},
    "rna": {"genes": [str(g) for g in genes], "mean": pp["rna"].mu_[nz].tolist(), "sd": pp["rna"].sd_[nz].tolist(),
            "coef": rc[nz].tolist(),
            "scale_note": "TCGA log2(CPM+1); arrays: log2(x+1), highest-mean probe per gene, per-gene z-score within cohort"},
    "cutoff": {"tcga_median_lp": cut, "tcga_lp_sd": float(lp.std())},
    "external_plan": {
        "primary": "GSE31210, all 226 tumours (OS); sensitivity: 204 without flagged incomplete resection/adjuvant therapy",
        "secondary": "GSE68465, 442 adenocarcinomas (OS; stage derived from pT/pN, no stage IV)",
        "scoring": "frozen coefficients; no refitting, tuning, or threshold search on external data",
        "success": "Harrell C > 0.60 with lower 95% CI > 0.50; high-vs-low HR > 1 with p < 0.05; calibration slope near 1",
        "groups": "primary split at the frozen TCGA median lp; within-cohort median split reported as sensitivity",
        "also_report": "RNA component alone and clinical component alone, HR per SD, Uno C at 5y",
    },
}
# self-check: the saved file reproduces the fit on TCGA
chk = score(sig, C, F.loc[sig["rna"]["genes"]], gene_scaling="frozen")["total"]
assert np.allclose(chk, lp, atol=2e-3), np.abs(chk - lp).max()
out_name = sys.argv[1] if len(sys.argv) > 1 else "signature_v1.json"   # reproduction passes a different name
out = models_dir / out_name
out.write_text(json.dumps(sig, indent=2))
sha = hashlib.sha256(out.read_bytes()).hexdigest()
if out_name == "signature_v1.json":
    pd.DataFrame({"gene": sig["rna"]["genes"], "coef_per_SD": sig["rna"]["coef"]}).sort_values("coef_per_SD", key=abs, ascending=False) \
        .to_csv(tables / "signature_v1_genes.tsv", sep="\t", index=False)
print(f"final fit on {n} patients / {int(event.sum())} events: l1={fit['l1']}, alpha={fit['alpha']:.4f}, inner-CV Harrell {fit['inner_c']:.3f}")
print(f"RNA genes kept after screening/prevalence: {len(pp['rna'].cols_)}; non-zero in final model: {len(nz)}")
print("clinical coefficients (per SD):", {f: round(c, 3) for f, c in zip(feats, sig['clinical']['coef'])})
print("RNA genes (coef per SD of z-scored expression):")
for g, c in sorted(zip(sig["rna"]["genes"], sig["rna"]["coef"]), key=lambda t: -abs(t[1])):
    print(f"   {g:12s} {c:+.3f}")
print(f"apparent (training) Harrell C {harrell_c(event, time, lp):.3f}  [optimistic; honest estimate is the CV 0.690 / Uno 0.650]")
print(f"frozen cutoff = TCGA median lp {cut:.3f}; lp SD {lp.std():.3f}")
print(f"signature saved: {out}\nsha256 {sha}")
