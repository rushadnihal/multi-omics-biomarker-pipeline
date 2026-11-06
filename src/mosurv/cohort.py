"""Cohort construction: patient alignment across layers, endpoints, exclusion log."""
from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd

STAGE_RE = re.compile(r"^Stage (IV|III|II|I)[ABC]?$")


def header_ids(path: Path) -> list[str]:
    """Sample IDs from the header row of a Xena matrix (first column is the feature id)."""
    with gzip.open(path, "rt") as fh:
        return fh.readline().rstrip("\n").split("\t")[1:]


def sample_code(sample_id: str) -> str:
    return sample_id[13:15]


def vial(sample_id: str) -> str:
    return sample_id[15:16]


def pick_one_per_patient(ids, code="01", exclude_vials=()):
    """Primary tumours only; one sample per patient (lowest ID). Returns (dict, n_dropped_dups)."""
    tum = sorted(s for s in ids if sample_code(s) == code and vial(s) not in set(exclude_vials))
    chosen: dict[str, str] = {}
    for s in tum:
        chosen.setdefault(s[:12], s)
    return chosen, len(tum) - len(chosen)


def collapse_stage(x):
    if not isinstance(x, str):
        return None
    m = STAGE_RE.match(x.strip())
    return m.group(1) if m else None


def build_cohort(cfg: dict, raw: Path):
    """Returns (cohort DataFrame indexed by patient, exclusion log DataFrame)."""
    c = cfg["cohort"]
    files = c["raw_files"]
    log: list[tuple[str, str, int | None, int]] = []

    def step(name, desc, removed, remaining):
        log.append((name, desc, removed, remaining))

    sv = pd.read_csv(raw / files["survival"], sep="\t")
    sv = sv[sv["cancer type abbreviation"] == "LUAD"]
    step("S0", "Liu pan-cancer table, LUAD rows (samples)", None, len(sv))

    code = c["tumour_sample_code"]
    other = sv[sv["sample"].str[13:15] != code]
    sv = sv[sv["sample"].str[13:15] == code]
    step("S1", f"keep primary tumour (code {code}); dropped {({k: int(v) for k, v in other['sample'].str[13:15].value_counts().items()})}",
         len(other), len(sv))

    dup = int(sv["_PATIENT"].duplicated().sum())
    sv = sv.drop_duplicates("_PATIENT", keep="first")
    step("S2", "one row per patient", dup, len(sv))

    miss = sv["OS"].isna() | sv["OS.time"].isna()
    step("S3", "drop missing OS or OS.time", int(miss.sum()), int((~miss).sum()))
    sv = sv[~miss]
    bad = sv["OS.time"] <= c["drop_os_time_le"]
    step("S4", f"drop OS.time <= {c['drop_os_time_le']} d", int(bad.sum()), int((~bad).sum()))
    sv = sv[~bad]

    out = pd.DataFrame(index=pd.Index(sv["_PATIENT"].values, name="patient"))
    out["os_time"] = sv["OS.time"].values
    out["os_event"] = sv["OS"].astype(int).values
    out["pfi_time"] = sv["PFI.time"].values
    out["pfi_event"] = sv["PFI"].values
    out["age"] = sv["age_at_initial_pathologic_diagnosis"].values
    out["gender"] = sv["gender"].values
    out["stage_raw"] = sv["ajcc_pathologic_tumor_stage"].values
    out["stage"] = out["stage_raw"].map(collapse_stage)
    step("C0", f"stage unmapped to I-IV: {int(out['stage'].isna().sum())} (kept; NaN)", 0, len(out))

    # smoking: pack-years from GDC clinical table (primary tumour rows, case level)
    clin = pd.read_csv(raw / files["clinical"], sep="\t", low_memory=False)
    pk = "pack_years_smoked.exposures"
    if pk in clin.columns:
        clin["patient"] = clin["sample"].str[:12]
        clin = clin[clin["sample"].str[13:15] == code]
        out["pack_years"] = clin.groupby("patient")[pk].first().reindex(out.index)
    else:
        out["pack_years"] = pd.NA

    for layer in c["core_layers"]:
        ids = header_ids(raw / files[layer])
        chosen, ndup = pick_one_per_patient(ids, code, c["exclude_vials"])
        out[f"{layer}_id"] = pd.Series(chosen).reindex(out.index)
        out[f"has_{layer}"] = out[f"{layer}_id"].notna()
        step(f"L-{layer}", f"{layer}: tumour samples {len(chosen) + ndup}, duplicate vials dropped {ndup}, "
             f"patients in cohort with {layer}: {int(out[f'has_{layer}'].sum())}", ndup, int(out[f"has_{layer}"].sum()))

    core = out[[f"has_{layer}" for layer in c["core_layers"]]].all(axis=1)
    out["in_core"] = core
    step("C1", f"core cohort: complete case across {c['core_layers']}", int((~core).sum()), int(core.sum()))
    step("C2", f"core cohort OS events: {int(out.loc[core, 'os_event'].sum())}", None, int(core.sum()))
    return out, pd.DataFrame(log, columns=["step", "description", "n_removed", "n_remaining"])
