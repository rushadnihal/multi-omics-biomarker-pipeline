"""One-off: which clinical fields do the GEO series matrices carry, and can platform annotations be downloaded?"""
import glob
import gzip
import subprocess
from pathlib import Path

print("raw GEO/GPL files:", sorted(p.name for p in Path("data/raw").iterdir() if "gse" in p.name.lower() or "gpl" in p.name.lower()))
for pat in ("*GSE31210*matrix*", "*GSE68465*matrix*"):
    for f in glob.glob(f"data/raw/{pat}"):
        print(f"\n=== {f}")
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt", errors="replace") as fh:
            k = 0
            for line in fh:
                if line.startswith(("!Sample_characteristics_ch1", "!Sample_platform_id", "!Sample_geo_accession")):
                    cells = line.rstrip("\n").split("\t")
                    vals = [c.strip('"')[:38] for c in cells[1:4]]
                    print(f"  {cells[0][:30]:30s} {vals}  (n={len(cells) - 1})")
                    k += 1
                if line.startswith("!series_matrix_table_begin"):
                    break
print()
for u in ("https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GPL570&targ=self&form=text&view=brief",
          "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL570nnn/GPL570/annot/GPL570.annot.gz",
          "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL96nnn/GPL96/annot/GPL96.annot.gz"):
    r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-L", "-I", u], capture_output=True, text=True)
    print(r.stdout, u)
