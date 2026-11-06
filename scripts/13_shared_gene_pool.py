"""M8a: define the gene pool measurable on TCGA RNA-seq AND both arrays (GPL570, GPL96).
Fixed, outcome-blind rules: symbols from GENCODE v36 / GPL annotation; ambiguous probes ('///') dropped;
duplicates collapsed to the highest-mean feature (unsupervised). Writes the map and gene-level array matrices."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

cfg = yaml.safe_load(open("config/config.yaml"))
raw, interim, tables = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["interim"]), Path(cfg["paths"]["tables"])
core = pd.read_parquet(Path(cfg["paths"]["processed"]) / "cohort.parquet").query("in_core")


def read_gpl(path):
    start = end = None
    with open(path, errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith("!platform_table_begin"):
                start = i
            elif line.startswith("!platform_table_end"):
                end = i
                break
    df = pd.read_csv(path, sep="\t", skiprows=start + 1, nrows=end - start - 2, dtype=str, quoting=3, low_memory=False)
    sym = df["Gene Symbol"].fillna("").str.strip().str.upper()
    ok = (sym != "") & (sym != "---") & ~sym.str.contains("///", regex=False)
    return pd.Series(sym[ok].to_numpy(), index=df["ID"][ok].to_numpy()), len(df), int(ok.sum())


def read_matrix(path):
    with gzip.open(path, "rt", errors="replace") as fh:
        for i, line in enumerate(fh):
            if line.startswith("!series_matrix_table_begin"):
                start = i
                break
    df = pd.read_csv(path, sep="\t", skiprows=start + 1, index_col=0, compression="gzip", low_memory=False)
    df = df[~df.index.astype(str).str.startswith("!")]
    return df.apply(pd.to_numeric, errors="coerce")


# TCGA side
gm = pd.read_csv(raw / "gencode_v36_gene_probemap.tsv", sep="\t").set_index("id")["gene"].str.upper()
rna = pd.read_parquet(interim / "rna_log2cpm.parquet")[core.index]
tcga = pd.DataFrame({"ensembl": rna.index, "symbol": gm.reindex(rna.index).to_numpy(), "mean": rna.mean(axis=1).to_numpy()}).dropna(subset=["symbol"])
tcga = tcga.sort_values("mean", ascending=False).drop_duplicates("symbol").set_index("symbol")
print(f"TCGA RNA features {rna.shape[0]}; with a symbol {int(gm.reindex(rna.index).notna().sum())}; unique symbols {len(tcga)}")

# array side
out = {}
for name, gpl, mat in (("gse31210", "GPL570", "GSE31210_series_matrix.txt.gz"), ("gse68465", "GPL96", "GSE68465_series_matrix.txt.gz")):
    ann, n_all, n_ok = read_gpl(raw / f"{gpl}_full.soft")
    X = read_matrix(raw / mat)
    v = X.to_numpy(dtype=float)
    print(f"\n{gpl}: {n_all} probes, {n_ok} with a single gene symbol; {name} matrix {X.shape}; "
          f"values: min {np.nanmin(v):.2f}, median {np.nanmedian(v):.2f}, p99 {np.nanpercentile(v, 99):.2f}, max {np.nanmax(v):.2f}, NaN {np.isnan(v).mean():.4f}")
    common = X.index.intersection(ann.index)
    sym = ann.reindex(common)
    m = X.loc[common].mean(axis=1)
    best = pd.DataFrame({"symbol": sym.to_numpy(), "mean": m.to_numpy()}, index=common).sort_values("mean", ascending=False).drop_duplicates("symbol")
    out[name] = (X, best)
    print(f"  probes matched to matrix rows {len(common)}; unique symbols {len(best)}")

shared = sorted(set(tcga.index) & set(out["gse31210"][1]["symbol"]) & set(out["gse68465"][1]["symbol"]))
print(f"\nshared symbols: TCGA ∩ GPL570 ∩ GPL96 = {len(shared)}")
gmap = pd.DataFrame({"symbol": shared, "ensembl": tcga.loc[shared, "ensembl"].to_numpy()})
for name, gpl in (("gse31210", "gpl570"), ("gse68465", "gpl96")):
    b = out[name][1].reset_index().rename(columns={"index": f"{gpl}_probe"}).set_index("symbol")
    gmap[f"{gpl}_probe"] = b.loc[shared, f"{gpl}_probe"].to_numpy()
    X = out[name][0]
    G = X.loc[gmap[f"{gpl}_probe"].to_numpy()]
    G.index = gmap["symbol"].to_numpy()
    G.to_parquet(interim / f"{name}_gene.parquet")
gmap.to_csv(tables / "shared_gene_map.tsv", sep="\t", index=False)
# how much of the TCGA RNA signal the shared pool keeps (outcome-blind: variance share)
var = rna.var(axis=1)
sel = var.reindex(gmap["ensembl"]).sum() / var.sum()
print(f"share of total TCGA RNA variance in shared genes: {sel:.2f}; of TCGA features: {len(gmap) / rna.shape[0]:.2f}")
print("saved results/tables/shared_gene_map.tsv, data/interim/gse31210_gene.parquet, data/interim/gse68465_gene.parquet")
