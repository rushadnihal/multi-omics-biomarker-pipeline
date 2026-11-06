"""Print a compact structural summary of data/raw (no big matrices loaded). Re-runnable."""
import gzip
import re
from collections import OrderedDict
from pathlib import Path
import pandas as pd
raw = Path("data/raw")
def xena_summary(name, count_rows=True):
    p = raw / name
    with gzip.open(p, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        first_ids = [fh.readline().split("\t", 1)[0] for _ in range(3)]
        n = 3 + sum(1 for _ in fh) if count_rows else "not counted"
    print(f"\n== {name}\n  columns: {len(header)} (first 3: {header[:3]}; last: {header[-1]})")
    print(f"  data rows: {n}; first row ids: {first_ids}")
for f in ["TCGA-LUAD.star_counts.tsv.gz", "TCGA-LUAD.mirna.tsv.gz",
          "TCGA-LUAD.somaticmutation_wxs.tsv.gz"]:
    xena_summary(f)
xena_summary("TCGA-LUAD.methylation450.tsv.gz", count_rows=False)
print("\n== clinical")
clin = pd.read_csv(raw / "TCGA-LUAD.clinical.tsv.gz", sep="\t", low_memory=False)
print("  shape:", clin.shape)
keys = [c for c in clin.columns if re.search(r"stage|smok|age|gender|sample_type|vital|days_to", c, re.I)]
print("  relevant columns:", keys[:40])
print("\n== pan-cancer survival (Liu 2018)")
sv = pd.read_csv(raw / "Survival_SupplementalTable_S1_20171025_xena_sp", sep="\t")
print("  columns:", list(sv.columns))
luad = sv[sv.iloc[:, sv.columns.get_indexer(["cancer type abbreviation"])[0]] == "LUAD"] \
    if "cancer type abbreviation" in sv.columns else sv.head(0)
print("  LUAD rows:", len(luad))
if len(luad):
    print("  sample id examples:", luad.iloc[:3, 0].tolist())
    for e, t in [("OS", "OS.time"), ("PFI", "PFI.time")]:
        if e in luad and t in luad:
            print(f"  {e}: events={int(luad[e].sum())} / {luad[e].notna().sum()}, "
                  f"{t} median={luad[t].median():.0f} d, missing time={luad[t].isna().sum()}")
def geo_summary(name):
    print(f"\n== {name}")
    chars, n_rows, in_table, plat = OrderedDict(), 0, False, None
    with gzip.open(raw / name, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Series_platform_id"):
                plat = line.split("\t")[1].strip().strip('"')
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [v.strip().strip('"') for v in line.rstrip("\n").split("\t")[1:]]
                chars[len(chars)] = vals
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                n_rows += 1
    print(f"  platform: {plat}; matrix rows (incl. header): {n_rows}; characteristic fields: {len(chars)}")
    for i, vals in chars.items():
        distinct = sorted(set(vals))
        shown = distinct if len(distinct) <= 6 else f"{len(distinct)} distinct, e.g. {distinct[:3]}"
        print(f"  [{i}] n={len(vals)} {shown}")

for f in ["GSE31210_series_matrix.txt.gz", "GSE68465_series_matrix.txt.gz"]:
    geo_summary(f)
