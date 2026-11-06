"""Reproduction helpers.
  verify  : compare the regenerated signature (signature_v1.repro.json) with the frozen signature_v1.json
  summary : build the headline tables (internal CV comparison + external validation) from the result files"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXPECTED_SHA = "704878c704c0b13e3c1f4e3fb3e7671bf411eae7e13611f19342a47c8cd48041"
cfg = yaml.safe_load(open("config/config.yaml"))
models_dir, tables = Path(cfg["paths"]["models"]), Path(cfg["paths"]["tables"])


def compare_signatures(a, b, tol=1e-6):
    """Return (ok, message). a = frozen, b = regenerated."""
    if a["rna"]["genes"] != b["rna"]["genes"] and set(a["rna"]["genes"]) != set(b["rna"]["genes"]):
        return False, f"gene sets differ: frozen {sorted(a['rna']['genes'])} vs regenerated {sorted(b['rna']['genes'])}"
    ca = dict(zip(a["rna"]["genes"], a["rna"]["coef"]))
    cb = dict(zip(b["rna"]["genes"], b["rna"]["coef"]))
    d_rna = max(abs(ca[g] - cb[g]) for g in ca)
    d_clin = float(np.max(np.abs(np.array(a["clinical"]["coef"]) - np.array(b["clinical"]["coef"]))))
    d_cut = abs(a["cutoff"]["tcga_median_lp"] - b["cutoff"]["tcga_median_lp"])
    worst = max(d_rna, d_clin, d_cut)
    msg = f"same genes; max |coef diff| RNA {d_rna:.2e}, clinical {d_clin:.2e}; cutoff diff {d_cut:.2e}"
    return worst <= tol, msg


def verify():
    frozen = models_dir / "signature_v1.json"
    repro = models_dir / "signature_v1.repro.json"
    sha_f = hashlib.sha256(frozen.read_bytes()).hexdigest()
    if sha_f != EXPECTED_SHA:
        sys.exit("FROZEN signature_v1.json does not match the recorded sha256 - it has been modified")
    if not repro.exists():
        sys.exit("regenerated signature not found (run scripts/15_freeze_signature.py signature_v1.repro.json)")
    sha_r = hashlib.sha256(repro.read_bytes()).hexdigest()
    ok, msg = compare_signatures(json.loads(frozen.read_text()), json.loads(repro.read_text()))
    print("signature reproduction:", "bit-identical file" if sha_r == sha_f else msg)
    if not ok:
        sys.exit("REPRODUCTION MISMATCH: regenerated signature differs from the frozen one beyond 1e-6")
    print("OK: the frozen signature is reproduced from raw data.")


def summary():
    from mosurv.evaluate import nb_ci

    n = len(pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core"))
    n_te = int(round(n / cfg["evaluation"]["outer_cv"]["n_splits"]))
    n_tr = n - n_te
    sources = []   # (label, dataframe indexed by fold with harrell, uno)

    def add(label, df):
        df = df.copy()
        df.index = df["fold"] if "fold" in df.columns else range(len(df))
        sources.append((label, df[["harrell", "uno"]]))

    clin = pd.read_csv(tables / "clinical_baseline_cv_folds.tsv", sep="\t")
    for m, g in clin.groupby("model"):
        add(f"clinical: {m}", g)
    for l in ("rna", "mirna", "meth"):
        add(f"{l} only", pd.read_csv(tables / f"single_omics_{l}_folds.tsv", sep="\t"))
    for c in ("rna", "mirna", "meth", "all", "rnashared"):
        p = tables / f"combo_{c}_folds.tsv"
        if p.exists():
            add(f"early: clinical + {c}", pd.read_csv(p, sep="\t"))
    add("late stacking", pd.read_csv(tables / "late_integration_folds.tsv", sep="\t"))
    for fn, lab in (("nonlinear_folds.tsv", ""), ("mofa_folds.tsv", "")):
        d = pd.read_csv(tables / fn, sep="\t")
        for m, g in d.groupby("model"):
            add(m, g)
    ref = dict(sources)["clinical: age+sex+stage"]
    rows = []
    for label, d in sources:
        r = {"model": label}
        for metric in ("uno", "harrell"):
            mean, lo, hi, _, _ = nb_ci(d[metric].to_numpy(), n_tr, n_te)
            diff = (d[metric] - ref[metric].reindex(d.index)).to_numpy()
            dm, dlo, dhi, _, p = nb_ci(diff, n_tr, n_te)
            r.update({f"{metric}": round(mean, 3), f"{metric}_ci": f"[{lo:.3f}, {hi:.3f}]",
                      f"{metric}_diff_vs_clinical": f"{dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"})
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(tables / "headline_internal_cv.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40)
    print("== internal repeated 5x5 CV (identical folds; Nadeau-Bengio CIs, nominal fold sizes)")
    print(out[["model", "uno", "uno_ci", "uno_diff_vs_clinical", "harrell", "harrell_diff_vs_clinical"]].to_string(index=False))
    ext = pd.read_csv(tables / "external_validation.tsv", sep="\t")
    print("\n== external validation of the frozen signature (results/tables/external_validation.tsv)")
    keep = ext[ext["score"].isin(["full signature", "RNA component only", "clinical component only"]) | ext["score"].str.startswith("high vs low (frozen")]
    print(keep[["cohort", "score", "harrell", "harrell_lo", "harrell_hi", "hr_per_sd", "hr_lo", "hr_hi", "p"]].round(3).to_string(index=False))
    print("\nsaved results/tables/headline_internal_cv.tsv")


if __name__ == "__main__":
    {"verify": verify, "summary": summary}[sys.argv[1]]()
