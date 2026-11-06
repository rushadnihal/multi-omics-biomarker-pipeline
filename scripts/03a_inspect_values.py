"""Check value scales and missingness per layer, restricted to the core cohort. Re-runnable."""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config/config.yaml"))
raw = Path(cfg["paths"]["raw"])
files = cfg["cohort"]["raw_files"]
coh = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet")
core = coh[coh.in_core]


def load(fname, ids, nrows=None):
    path = raw / fname
    first = pd.read_csv(path, sep="\t", nrows=0).columns[0]
    keep = set(ids) | {first}
    return pd.read_csv(path, sep="\t", index_col=0, usecols=lambda c: c in keep, nrows=nrows)


print("== RNA (STAR counts)")
X = load(files["rna"], core["rna_id"])
print(f"  shape genes x samples: {X.shape}; NaN: {int(X.isna().sum().sum())}")
print(f"  min={X.values.min():.3f} max={X.values.max():.3f}")
sub = X.values[:2000, :50].ravel()
raw_counts = 2.0 ** sub - 1
print(f"  looks like log2(count+1)? integer-valued after 2^x-1: {np.allclose(raw_counts, np.round(raw_counts), atol=1e-3)}")
if np.allclose(raw_counts, np.round(raw_counts), atol=1e-3):
    lib = (2.0 ** X - 1).sum(axis=0)
    print(f"  library size (M counts): min={lib.min()/1e6:.1f} median={lib.median()/1e6:.1f} max={lib.max()/1e6:.1f}")
    cnt = 2.0 ** X - 1
    print(f"  all-zero genes: {int((cnt.sum(axis=1) == 0).sum())}; "
          f">=10 counts in >=10% samples: {int(((cnt >= 10).mean(axis=1) >= 0.10).sum())}")
base = X.index.str.replace(r"\.\d+$", "", regex=True)
print(f"  unique Ensembl base IDs: {base.nunique()} of {len(base)}; PAR_Y-style duplicates: {len(base) - base.nunique()}")

print("\n== miRNA")
M = load(files["mirna"], core["mirna_id"])
print(f"  shape: {M.shape}; NaN: {int(M.isna().sum().sum())}; min={M.values.min():.3f} max={M.values.max():.3f}")
print(f"  all-zero features: {int((M.sum(axis=1) == 0).sum())}; "
      f"features with >0 in >=20% samples: {int(((M > 0).mean(axis=1) >= 0.2).sum())}")

print("\n== Methylation (first 20000 probes only)")
B = load(files["meth"], core["meth_id"], nrows=20000)
print(f"  shape: {B.shape}; min={np.nanmin(B.values):.3f} max={np.nanmax(B.values):.3f}")
na_row = B.isna().mean(axis=1)
na_col = B.isna().mean(axis=0)
print(f"  NaN overall: {B.isna().mean().mean():.3%}; probes with >10% NaN: {int((na_row > 0.10).sum())}; "
      f"probes with any NaN: {int((na_row > 0).sum())}; worst sample NaN: {na_col.max():.2%}")
print("  counting all methylation probes (streams the 1.7 GB file, ~1 min)...")
n = int(subprocess.run(f"gzip -dc '{raw / files['meth']}' | wc -l", shell=True, capture_output=True,
                       text=True).stdout.strip())
print(f"  total methylation rows (incl. header): {n}")
