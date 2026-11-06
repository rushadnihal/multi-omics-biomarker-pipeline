"""Download raw data listed in config/config.yaml. Safe to re-run (skips finished files,
resumes partial ones). Writes data/raw/MANIFEST.tsv. Stdlib + pyyaml only."""
import hashlib
import subprocess
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(open("config/config.yaml"))
raw = Path(cfg["paths"]["raw"])
raw.mkdir(parents=True, exist_ok=True)


def fetch(url: str, dest: Path) -> None:
    """Download via system curl (uses the OS trust store; resumes with -C -)."""
    part = dest.with_suffix(dest.suffix + ".part")
    cmd = ["curl", "-fL", "--retry", "3", "-C", "-", "--progress-bar", "-o", str(part), url]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"curl exit code {res.returncode}")
    part.rename(dest)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


rows, failed_required = [], []
for name, spec in cfg["downloads"].items():
    got = None
    for url in spec["urls"]:
        dest = raw / (spec.get("filename") or url.rsplit("/", 1)[1])
        if dest.exists():
            got = (url, dest)
            print(f"[skip] {name}: {dest.name} already present")
            break
        print(f"[get ] {name}: {url}")
        try:
            fetch(url, dest)
            got = (url, dest)
            break
        except Exception as e:  # keep .part so a re-run resumes
            print(f"  failed: {e}")
    if got:
        rows.append((name, got[0], got[1].name, got[1].stat().st_size, sha256(got[1])))
    elif spec["required"]:
        failed_required.append(name)
    else:
        print(f"[warn] optional '{name}' not available; continuing")

with open(raw / "MANIFEST.tsv", "w") as fh:
    fh.write("name\turl\tfile\tbytes\tsha256\n")
    for r in rows:
        fh.write("\t".join(map(str, r)) + "\n")
print("\nManifest:", raw / "MANIFEST.tsv")
for r in rows:
    print(f"  {r[0]:18s} {r[2]:55s} {r[3] / 1e6:9.1f} MB")
if failed_required:
    sys.exit(f"MISSING REQUIRED: {failed_required}")
