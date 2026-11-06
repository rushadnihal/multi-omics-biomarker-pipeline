"""RNA scale diagnostic + tissue-source-site batch vs outcome check. Re-runnable, outcome used only for the TSS test."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from lifelines.statistics import multivariate_logrank_test
cfg = yaml.safe_load(open("config/config.yaml"))
raw = Path(cfg["paths"]["raw"])
files = cfg["cohort"]["raw_files"]
coh = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet")
core = coh[coh.in_core]
path = raw / files["rna"]
first = pd.read_csv(path, sep="\t", nrows=0).columns[0]
keep = set(core["rna_id"]) | {first}
X = pd.read_csv(path, sep="\t", index_col=0, usecols=lambda c: c in keep)
print("== RNA scale")
u = np.unique(X.values[:, :60])
print("  10 smallest positive values:", np.round(u[u > 0][:10], 4))
print("  log2(k+1) for k=1..8      :", [round(float(np.log2(k + 1)), 4) for k in range(1, 9)])
lib = (2.0 ** X - 1).sum(axis=0) / 1e6
print(f"  sum of 2^x-1 per sample (M): min={lib.min():.1f} median={lib.median():.1f} max={lib.max():.1f}")
print("\n== Tissue source site (TSS) vs outcome, core cohort")
tss = pd.Series(core.index.str[5:7], index=core.index, name="tss")
t = pd.DataFrame({"n": tss.value_counts(), "events": core.groupby(tss)["os_event"].sum()}).sort_values("n", ascending=False)
print(f"  {len(t)} TSS; n>=10: {int((t.n >= 10).sum())}; n<5: {int((t.n < 5).sum())}")
print(t.head(12).to_string())
grp = tss.where(tss.map(tss.value_counts()) >= 10, "other")
res = multivariate_logrank_test(core["os_time"], grp, core["os_event"])
print(f"  multivariate log-rank across TSS groups (n>=10, rest pooled): chi2={res.test_statistic:.1f} "
      f"df={len(set(grp)) - 1} p={res.p_value:.3g}")
stage = core["stage"].fillna("NA")
ct = pd.crosstab(grp, stage)
print("  stage mix by TSS group (rows) :")
print(ct.to_string())
