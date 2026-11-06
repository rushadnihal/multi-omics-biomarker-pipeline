"""Sample-ID structure, cross-layer overlap, and GSE31210 field parsing. Re-runnable."""
import gzip
from collections import Counter
from pathlib import Path
import pandas as pd
raw = Path("data/raw")
def header_ids(name):
    with gzip.open(raw / name, "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")[1:]
layers = {
    "rna": header_ids("TCGA-LUAD.star_counts.tsv.gz"),
    "meth": header_ids("TCGA-LUAD.methylation450.tsv.gz"),
    "mirna": header_ids("TCGA-LUAD.mirna.tsv.gz"),
}
tumour_patients = {}
print("== TCGA matrices: sample type codes (01=primary tumour, 11=normal, 02=recurrent)")
for k, ids in layers.items():
    tum = [s for s in ids if s[13:15] == "01"]
    pats = [s[:12] for s in tum]
    dup = len(pats) - len(set(pats))
    tumour_patients[k] = set(pats)
    print(f"  {k}: n={len(ids)} codes={dict(Counter(s[13:15] for s in ids))} "
          f"tumour={len(tum)} unique_patients={len(set(pats))} duplicate_tumour_rows={dup}")
allp = set.intersection(*tumour_patients.values())
print(f"  patients with tumour in RNA & meth & miRNA: {len(allp)}; "
      f"RNA&meth: {len(tumour_patients['rna'] & tumour_patients['meth'])}")

print("\n== Liu survival table (LUAD rows)")
sv = pd.read_csv(raw / "Survival_SupplementalTable_S1_20171025_xena_sp", sep="\t")
sv = sv[sv["cancer type abbreviation"] == "LUAD"]
print(f"  rows={len(sv)} unique _PATIENT={sv['_PATIENT'].nunique()}")
print(f"  sample type codes: {dict(Counter(s[13:15] for s in sv['sample']))}")
print(f"  histological_type: {dict(sv['histological_type'].value_counts(dropna=False))}")
t = sv[sv["sample"].str[13:15] == "01"]
print(f"  code-01 rows={len(t)} patients={t['_PATIENT'].nunique()} "
      f"OS events={int(t['OS'].sum())} missing OS.time={int(t['OS.time'].isna().sum())}")
print(f"  of those, in RNA&meth&miRNA overlap: {t['_PATIENT'].isin(allp).sum()}")
print("\n== GDC clinical table")
clin = pd.read_csv(raw / "TCGA-LUAD.clinical.tsv.gz", sep="\t", low_memory=False)
print("  id-like columns:", [c for c in clin.columns if "sample" in c.lower() or "submitter" in c.lower()][:6])
print("  sample_type.samples:", dict(clin["sample_type.samples"].value_counts(dropna=False)))
print("\n== GSE31210 parsed per sample")
rows = None
with gzip.open(raw / "GSE31210_series_matrix.txt.gz", "rt", errors="replace") as fh:
    for line in fh:
        if line.startswith("!Sample_geo_accession"):
            accs = [v.strip('"') for v in line.rstrip("\n").split("\t")[1:]]
            rows = [dict() for _ in accs]
        elif line.startswith("!Sample_characteristics_ch1"):
            for d, v in zip(rows, line.rstrip("\n").split("\t")[1:]):
                v = v.strip('"')
                if ": " in v:
                    k, val = v.split(": ", 1)
                    d[k] = val
g = pd.DataFrame(rows, index=accs)
print("  fields:", list(g.columns))
for c in g.columns:
    nn = g[c].notna().sum()
    if g[c].nunique() <= 6:
        print(f"  {c!r}: n={nn} {dict(g[c].value_counts())}")
    else:
        print(f"  {c!r}: n={nn} ({g[c].nunique()} distinct)")
tum = g[g["tissue"] == "primary lung tumor"]
print(f"  tumours={len(tum)}")
ex = [c for c in g.columns if c.startswith("exclude")]
if ex:
    print("  exclusion field among tumours:", dict(tum[ex[0]].value_counts(dropna=False)))
