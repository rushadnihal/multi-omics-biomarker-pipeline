"""Aligned, unit-transformed (UNFILTERED) matrices for the core cohort + outcome-blind QC PCA.

Writes data/interim/{rna_log2cpm,mirna_log2rpm,meth_mvalue}.parquet (features x patients),
results/tables/sample_qc.tsv, results/figures/qc_pca.png. Display subsets used for PCA are
chosen without outcome and are NOT used for modelling. Re-runnable.
"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from mosurv.preprocess import (  # noqa: E402
    beta_to_m, log2cpm_from_log2count1, pca_scores, r2_by_group, robust_outliers)

cfg = yaml.safe_load(open("config/config.yaml"))
raw, interim = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"])
figs, tables = Path(cfg["paths"]["figures"]), Path(cfg["paths"]["tables"])
for p in (interim, figs, tables):
    p.mkdir(parents=True, exist_ok=True)
files = cfg["cohort"]["raw_files"]
coh = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet")
core = coh[coh.in_core].copy()
n = len(core)


def reader(fname, ids, chunksize=None):
    path = raw / fname
    first = pd.read_csv(path, sep="\t", nrows=0).columns[0]
    cols = list(ids)
    return pd.read_csv(path, sep="\t", index_col=0, usecols=[first] + cols,
                       dtype={c: np.float32 for c in cols}, chunksize=chunksize), cols


print("RNA ...")
rd, cols = reader(files["rna"], core["rna_id"])
rna = rd[cols]
rna.columns = core.index
rna, lib = log2cpm_from_log2count1(rna)
rna.to_parquet(interim / "rna_log2cpm.parquet")
print(f"  saved {rna.shape}")

print("miRNA ...")
rd, cols = reader(files["mirna"], core["mirna_id"])
mir = rd[cols]
mir.columns = core.index
mir.to_parquet(interim / "mirna_log2rpm.parquet")
print(f"  saved {mir.shape}")

print("Methylation (streaming, a few minutes) ...")
rd, cols = reader(files["meth"], core["meth_id"], chunksize=50000)
parts, nan_per_sample, total = [], np.zeros(n), 0
for ch in rd:
    ch = ch[cols]
    nan_per_sample += ch.isna().sum(axis=0).to_numpy()
    total += len(ch)
    parts.append(pd.DataFrame(beta_to_m(ch.to_numpy()), index=ch.index, columns=core.index))
meth = pd.concat(parts)
del parts
meth.to_parquet(interim / "meth_mvalue.parquet")
print(f"  saved {meth.shape}")

core["tss"] = core.index.str[5:7]
core["tss_group"] = core["tss"].where(core["tss"].map(core["tss"].value_counts()) >= 10, "other")
core["rna_libsize_M"] = (lib.reindex(core.index) / 1e6).values
core["meth_nan_frac"] = nan_per_sample / total
core[["tss", "tss_group", "stage", "gender", "rna_libsize_M", "meth_nan_frac"]].to_csv(tables / "sample_qc.tsv", sep="\t")
print(f"  methylation NaN fraction per sample: median={np.median(core.meth_nan_frac):.2%} "
      f"max={core.meth_nan_frac.max():.2%}; samples >10%: {int((core.meth_nan_frac > 0.10).sum())}")

# ---- outcome-blind display subsets
sub = {}
keep = rna.mean(axis=1) > 1
sub["RNA (top 5000 var, log2CPM)"] = rna[keep].loc[rna[keep].var(axis=1).nlargest(5000).index]
nz = mir[(mir > 0).mean(axis=1) >= 0.2]
sub["miRNA (top 500 var)"] = nz.loc[nz.var(axis=1).nlargest(500).index]
full = meth[meth.notna().all(axis=1)]
print(f"  methylation probes complete in all {n} samples: {len(full)} of {len(meth)}")
sub["Methylation (top 20000 var, complete probes, M)"] = full.loc[full.var(axis=1).nlargest(20000).index]

stage = core["stage"].where(core["stage"].notna(), None)
gender = core["gender"]
k = core["tss_group"].nunique()
print(f"\nVariance of leading PCs explained by covariates (R^2). Chance level for TSS ~ {(k - 1) / (n - 1):.3f}")
print(f"{'layer':50s} {'PC':>3s} {'var%':>6s} {'R2_tss':>7s} {'R2_sex':>7s} {'R2_stage':>8s}  extra")
scores = {}
for name, F in sub.items():
    S, var = pca_scores(F, 5)
    scores[name] = (S, var)
    for i in range(5):
        extra = ""
        if name.startswith("RNA"):
            extra = f"rho(libsize)={spearmanr(S[:, i], core['rna_libsize_M'])[0]:.2f}"
        if name.startswith("Meth"):
            extra = f"rho(NaN frac)={spearmanr(S[:, i], core['meth_nan_frac'])[0]:.2f}"
        print(f"{name:50s} {i + 1:>3d} {var[i] * 100:6.1f} {r2_by_group(S[:, i], core['tss_group']):7.3f} "
              f"{r2_by_group(S[:, i], gender):7.3f} {r2_by_group(S[:, i], stage):8.3f}  {extra}")
    print(f"{'':50s} PC1/PC2 robust outliers (>4 MAD): {robust_outliers(S[:, 0])}/{robust_outliers(S[:, 1])}")

# ---- figure
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
tss_levels = sorted(core["tss_group"].unique())
cmap = plt.get_cmap("tab20")
for j, (name, (S, var)) in enumerate(scores.items()):
    for i, (label, vals, levels) in enumerate([
            ("TSS group", core["tss_group"], tss_levels),
            ("stage", core["stage"].fillna("NA"), ["I", "II", "III", "IV", "NA"])]):
        ax = axes[i, j]
        for li, lv in enumerate(levels):
            m = (vals == lv).to_numpy()
            ax.scatter(S[m, 0], S[m, 1], s=14, color=cmap(li % 20), label=lv, alpha=0.8)
        ax.set_xlabel(f"PC1 ({var[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1] * 100:.1f}%)")
        ax.set_title(f"{name.split(' (')[0]} - by {label}", fontsize=10)
        if j == 2:
            ax.legend(fontsize=7, ncol=2, loc="best")
fig.tight_layout()
fig.savefig(figs / "qc_pca.png", dpi=110)
print("\nsaved results/figures/qc_pca.png, results/tables/sample_qc.tsv and data/interim/*.parquet")
