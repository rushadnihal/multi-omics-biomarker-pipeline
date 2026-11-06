"""Fixed annotation-based methylation probe filter (autosomes only) + outcome-blind re-check. Re-runnable."""
from pathlib import Path

import pandas as pd
import yaml

from mosurv.preprocess import autosomal_probes, pca_scores, r2_by_group

cfg = yaml.safe_load(open("config/config.yaml"))
raw, interim, tables = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]), Path(cfg["paths"]["tables"])

pm = pd.read_csv(raw / "illuminaMethyl450_probemap_hg19.tsv", sep="\t")
meth = pd.read_parquet(interim / "meth_mvalue.parquet")
keep = autosomal_probes(pm)

in_map = meth.index.isin(pm["#id"])
chrom = pm.drop_duplicates("#id").set_index("#id")["chrom"].reindex(meth.index)
kept = meth.index.isin(keep)
print(f"probes in matrix: {len(meth)}")
print(f"  not in probe map (unannotated): {int((~in_map).sum())}")
print(f"  in map, chrX: {int((chrom == 'chrX').sum())}, chrY: {int((chrom == 'chrY').sum())}, no chrom: {int((in_map & chrom.isna()).sum())}")
print(f"  kept (autosomal, annotated): {int(kept.sum())}")
meth_a = meth[kept]
meth_a.to_parquet(interim / "meth_mvalue_autosomal.parquet")

qc = pd.read_csv(tables / "sample_qc.tsv", sep="\t", index_col=0, dtype={"tss": str, "tss_group": str})
qc = qc.reindex(meth_a.columns)
full = meth_a[meth_a.notna().all(axis=1)]
print(f"\nautosomal probes complete in all samples: {len(full)} of {len(meth_a)}")
top = full.loc[full.var(axis=1).nlargest(20000).index]
S, var = pca_scores(top, 5)
print("PCA of top-20000-variance complete autosomal probes (outcome-blind):")
print(f"{'PC':>3s} {'var%':>6s} {'R2_tss':>7s} {'R2_sex':>7s} {'R2_stage':>8s}")
for i in range(5):
    print(f"{i + 1:>3d} {var[i] * 100:6.1f} {r2_by_group(S[:, i], qc['tss_group']):7.3f} "
          f"{r2_by_group(S[:, i], qc['gender']):7.3f} {r2_by_group(S[:, i], qc['stage']):8.3f}")
