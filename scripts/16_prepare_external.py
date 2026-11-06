"""M8d: build the external cohorts (phenotypes + log2 gene-level expression). Does NOT score the signature.
Prints counts to check against the pre-specified numbers (GSE31210: 226 tumours, 35 deaths, 22 flagged; GSE68465: 442 adenocarcinomas)."""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config/config.yaml"))
raw, interim, processed = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]), Path(cfg["paths"]["processed"])


def sample_chars(path):
    """Per-SAMPLE parse by cell label (GEO characteristic rows are ragged: optional fields shift cells between rows)."""
    ids, rows = None, []
    with gzip.open(path, "rt", errors="replace") as fh:
        for line in fh:
            if line.startswith("!Sample_geo_accession"):
                ids = [c.strip('"') for c in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                rows.append([c.strip('"') for c in line.rstrip("\n").split("\t")[1:]])
            elif line.startswith("!series_matrix_table_begin"):
                break
    recs = []
    for j in range(len(ids)):
        d = {}
        for r in rows:
            cell = r[j]
            if ":" in cell:
                k, v = cell.split(":", 1)
                d.setdefault(k.strip(), v.strip())      # first occurrence wins
        recs.append(d)
    return pd.DataFrame(recs, index=ids)


def finish(df, expr_file, out_name, keep):
    df = df[keep].copy()
    df["stage_II"], df["stage_III"], df["stage_IV"] = [(df["stage"] == s).astype(float) for s in ("II", "III", "IV")]
    df.loc[df["stage"].isna(), ["stage_II", "stage_III", "stage_IV"]] = np.nan
    G = pd.read_parquet(interim / expr_file)[df.index]
    L = np.log2(G.astype(float) + 1.0)
    L.to_parquet(interim / f"{out_name}_gene_log2.parquet")
    df.to_parquet(processed / f"{out_name}_cohort.parquet")
    print(f"  expression after log2(x+1): genes {L.shape[0]}, samples {L.shape[1]}, median {np.median(L.to_numpy()):.2f}, "
          f"p1 {np.percentile(L.to_numpy(), 1):.2f}, p99 {np.percentile(L.to_numpy(), 99):.2f}")
    return df


# ---------------- GSE31210 ----------------
print("== GSE31210")
p = sample_chars(raw / "GSE31210_series_matrix.txt.gz")
print("  tissue:", p["tissue"].value_counts().to_dict())
print("  pathological stage values:", p["pathological stage"].value_counts().to_dict())
ex_col = [c for c in p.columns if c.startswith("exclude for prognosis")][0]
print("  exclusion field values:", p[ex_col].fillna("(absent)").value_counts().to_dict())
df = pd.DataFrame(index=p.index)
df["tumour"] = p["tissue"].str.contains("tumor", case=False)
df["age"] = pd.to_numeric(p["age (years)"].where(p["age (years)"].notna(), p["age"]), errors="coerce")
df["male"] = (p["gender"].str.lower() == "male").astype(float)
st = p["pathological stage"].str.upper().str.strip()
df["stage"] = np.where(st.str.startswith("II"), "II", np.where(st.str.startswith("I"), "I", None))
df["os_event"] = (p["death"].str.lower() == "dead").astype(int)
df["os_time_days"] = pd.to_numeric(p["days before death/censor"], errors="coerce")
df["flagged_exclude"] = ~p[ex_col].fillna("none").str.strip().str.lower().isin(["none", ""])
df = df[df["tumour"]]
d = finish(df, "gse31210_gene.parquet", "gse31210", ["age", "male", "stage", "os_event", "os_time_days", "flagged_exclude"])
print(f"  tumours {len(d)} (pre-specified 226); deaths {int(d.os_event.sum())} (35); flagged {int(d.flagged_exclude.sum())} (22) -> sensitivity set {int((~d.flagged_exclude).sum())} (204)")
print(f"  age {d.age.min():.0f}-{d.age.max():.0f}, male {d.male.mean():.0%}, stage {d.stage.value_counts().to_dict()}, "
      f"median follow-up {d.os_time_days.median() / 365.25:.1f}y, missing age/time {int(d.age.isna().sum())}/{int(d.os_time_days.isna().sum())}")

# ---------------- GSE68465 ----------------
print("\n== GSE68465")
p = sample_chars(raw / "GSE68465_series_matrix.txt.gz")
print("  disease_state:", p["disease_state"].value_counts().to_dict())
print("  disease_stage examples:", p["disease_stage"].value_counts().head(12).to_dict())
m = p["disease_stage"].str.extract(r"pN(\d)\s*pT(\d)", flags=re.I)
tn = m.astype(float)
n_, t_ = tn[0], tn[1]
stage = pd.Series(None, index=p.index, dtype=object)
stage[(n_ == 0) & t_.isin([1, 2])] = "I"
stage[((n_ == 1) & t_.isin([1, 2])) | ((n_ == 0) & (t_ == 3))] = "II"
stage[((n_ == 1) & (t_ == 3)) | (n_ == 2) | (t_ == 4)] = "III"
print(f"  unparsed disease_stage: {int(m[0].isna().sum())}; derived stage (all samples): {stage.value_counts(dropna=False).to_dict()}")
df = pd.DataFrame(index=p.index)
df["adeno"] = p["disease_state"].str.contains("adenocarcinoma", case=False)
df["age"] = pd.to_numeric(p["age"], errors="coerce")
df["male"] = (p["Sex"].str.lower() == "male").astype(float)
df["stage"] = stage
df["os_event"] = (p["vital_status"].str.lower() == "dead").astype(int)
df["os_time_days"] = pd.to_numeric(p["months_to_last_contact_or_death"], errors="coerce") * 30.4375
df["adjuvant_chemo"] = (p["clinical_treatment_adjuvant_chemo"].str.lower() == "yes").astype(int)
df = df[df["adeno"]]
d = finish(df, "gse68465_gene.parquet", "gse68465", ["age", "male", "stage", "os_event", "os_time_days", "adjuvant_chemo"])
print(f"  adenocarcinomas {len(d)} (pre-specified 442); deaths {int(d.os_event.sum())}; adjuvant chemo {int(d.adjuvant_chemo.sum())}")
print(f"  age {d.age.min():.0f}-{d.age.max():.0f}, male {d.male.mean():.0%}, stage {d.stage.value_counts(dropna=False).to_dict()}, "
      f"median follow-up {d.os_time_days.median() / 365.25:.1f}y, missing age/time {int(d.age.isna().sum())}/{int(d.os_time_days.isna().sum())}")
print("\nsaved data/processed/gse*_cohort.parquet and data/interim/gse*_gene_log2.parquet (signature NOT scored)")
