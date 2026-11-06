"""Build data/processed/cohort.parquet and results/tables/cohort_exclusions.tsv."""
from pathlib import Path

import pandas as pd
import yaml

from mosurv.cohort import build_cohort

cfg = yaml.safe_load(open("config/config.yaml"))
raw = Path(cfg["paths"]["raw"])
cohort, log = build_cohort(cfg, raw)

Path(cfg["paths"]["processed"]).mkdir(parents=True, exist_ok=True)
Path(cfg["paths"]["tables"]).mkdir(parents=True, exist_ok=True)
cohort.to_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet")
log.to_csv(Path(cfg["paths"]["tables"]) / "cohort_exclusions.tsv", sep="\t", index=False)

pd.set_option("display.width", 200, "display.max_colwidth", 110)
print(log.to_string(index=False))
core = cohort[cohort.in_core]
print(f"\ncohort.parquet: {len(cohort)} patients, core {len(core)}")
print("core OS events:", int(core.os_event.sum()), "| median follow-up-ish (OS.time) days:",
      int(core.os_time.median()))
print("core stage:", dict(core.stage.value_counts(dropna=False)))
print("core gender:", dict(core.gender.value_counts(dropna=False)))
print("core age missing:", int(core.age.isna().sum()), "| pack_years missing:", int(core.pack_years.isna().sum()))
