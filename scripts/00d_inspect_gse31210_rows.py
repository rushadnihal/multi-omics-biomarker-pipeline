"""One-off: GSE31210 characteristic rows are ragged (cells shifted for some samples). Show the structure."""
import gzip
from collections import Counter

rows, ids = [], None
with gzip.open("data/raw/GSE31210_series_matrix.txt.gz", "rt", errors="replace") as fh:
    for line in fh:
        if line.startswith("!Sample_geo_accession"):
            ids = [c.strip('"') for c in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            rows.append([c.strip('"') for c in line.rstrip("\n").split("\t")[1:]])
        elif line.startswith("!series_matrix_table_begin"):
            break
print(len(rows), "characteristic rows,", len(ids), "samples")
for i, r in enumerate(rows):
    c = Counter(x.split(":", 1)[0].strip()[:42] for x in r)
    print(f"row {i:2d}: " + "; ".join(f"{k!r} x{v}" for k, v in c.most_common(4)))
print("\ndistinct cell texts mentioning exclude/incomplete/adjuvant/resection:")
cells = Counter(x[:80] for r in rows for x in r if any(w in x.lower() for w in ("exclude", "incomplete", "adjuvant", "resection")))
for k, v in cells.most_common(12):
    print(f"  {v:4d}  {k!r}")
print("\nper-sample death/days cell counts:")
n_death = sum(any(x.lower().startswith("death:") for x in (r[j] for r in rows)) for j in range(len(ids)))
n_days = sum(any(x.lower().startswith("days before death/censor") for x in (r[j] for r in rows)) for j in range(len(ids)))
print(f"  samples with a 'death:' cell {n_death}; with 'days before death/censor' cell {n_days}")
