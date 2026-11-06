# STATE

## Environment
- macOS, Python 3.12, venv + pip. Project root = this folder. `make test` passes (milestone 1).
- Downloads go through system `curl` (Python's urllib failed TLS verification on this network; never disable verification).

## Decisions
- Primary endpoint: OS. Secondary: PFI (Liu et al. 2018 table).
- Discovery: TCGA-LUAD (Xena GDC hub, STAR counts, HM450, miRNA, clinical; Liu survival from PanCanAtlas hub).
- External (pre-specified, decided before any results):
  - GSE31210 (GPL570, 226 tumours, 35 deaths, stage I-II). Primary = all 226; sensitivity = 204 after the authors' exclusion flag (incomplete resection/adjuvant therapy).
  - GSE68465 (GPL96, 442 adenocarcinomas, OS) as secondary, better-powered check.
  - Caveat: only 35 OS events in GSE31210 -> wide CI; criteria not changed.
- Cohort rules (config/config.yaml `cohort:`): primary tumour (-01) only; one sample per patient per layer (lowest ID); drop missing OS/OS.time or OS.time <= 0; benchmark cohort = complete case over RNA + methylation + miRNA.
- CNV not downloaded (GDC hub file names tried returned 403); revisit at QC, optional layer.

## Data facts (from inspection)
- Liu table LUAD: 641 rows = 519 tumour + 120 normal + 2 recurrent; 519 patients; 188 OS events; 9 missing OS.time.
- Tumour samples: RNA 528 rows/516 patients, methylation 469/458, miRNA 516/513. 453 patients have all three.
- Smoking: no categorical status in the GDC table; only pack_years_smoked (expect missingness).

## Done
- M1 scaffold (make test passes, 4 tests).
- M2 download + cohort: data/processed/cohort.parquet, results/tables/cohort_exclusions.tsv.
  - 641 Liu rows -> 519 primary tumours -> 510 (9 missing OS) -> 506 (4 with OS.time <= 0).
  - Layers per patient (after one-vial rule): RNA 503, methylation 447, miRNA 500.
  - Core cohort (all three layers): 442 patients, 159 OS events. Stage I/II/III/IV/NA = 239/108/70/20/5.
  - Missingness in core: age 10, pack_years 142 (32%) -> impute inside CV folds only.

- M3 per-layer QC (done): aligned, unit-transformed, UNFILTERED matrices (features x core patients) in data/interim/:
  rna_log2cpm (60,660 genes; Xena file is log2(count+1) with 0.5-step counts -> back-transformed, log2(CPM+1)),
  mirna_log2rpm (1,881), meth_mvalue (486,427 probes) and meth_mvalue_autosomal (385,809; fixed annotation rule:
  chr1-22 per hg19 probeMap; 90,442 unannotated + 9,779 chrX + 362 chrY + 35 no-chrom removed).
  QC findings: no outlier samples; no library-size effect; methylation PC2 was sex (R2 0.69) -> 0.05 after filter;
  TSS R2 on top PCs ~0.03-0.16 (chance 0.03; miRNA PC5 0.16, RNA PC2 0.11) - monitor, not dominant.
  TSS is associated with OS (log-rank p=0.03, 13 groups) -> check signature vs TSS in M7/M9.
  Methylation missingness ~14%/sample (uniform), 336k/486k probes complete -> in-fold probe filter + imputation.

- M4 clinical baseline (done; repeated 5x5 CV, identical folds, n=442, 159 events):
  Uno C (tau 5y): age+sex 0.469 | stage 0.625 | age+sex+stage 0.636 [0.586,0.686] | +smoking 0.622.
  Harrell: 0.471 | 0.659 | 0.657 [0.607,0.707] | 0.645. Paired: stage adds +0.168 Uno [0.100,0.235];
  smoking -0.015 [-0.040,+0.011] (no gain). Full-data HR: stage II 2.07, III 3.02, IV 3.44 (ref I); age/sex ~1.
  PH test: sex p=0.021, stage III p=0.012 (mild). BASELINE TO BEAT = age+sex+stage, Uno C 0.636.

- M5 single-omics, omics-only elastic-net Cox, nested repeated 5x5 CV, same folds as clinical baseline:
  miRNA: Uno 0.603 [0.537,0.669], Harrell 0.624 [0.558,0.690]; inner-CV C 0.626; l1=0.1 chosen in 16/25 folds
         (grid edge, dense: median 101 non-zero). Top-selected miRNAs include low-abundance precursors -> check TSS/batch.
  RNA:   Uno 0.572 [0.510,0.635], Harrell 0.623 [0.569,0.677]; inner-CV C 0.622; sparse (median 13 genes);
         ENSG00000107984.10 selected in 100% of folds (map to symbols in M9).
  Inner-CV C ~ outer C in both layers (no leakage sign). Both below clinical baseline (Uno 0.636).
  No post-hoc grid changes made (l1 grid fixed at {0.1,0.5,0.9}).

  meth:  Uno 0.565 [0.510,0.621], Harrell 0.600 [0.539,0.661]; inner-CV C 0.609; dense (median 34 probes);
         ~347.6k of 385.8k probes pass the in-fold missingness filter; 8.8 min for 25 folds (4 workers).
  Paired vs clinical (age+sex+stage), omics-only: all CIs include 0 (meth Uno -0.071 [-0.148,+0.006], p=0.069).
  Rule-9 diagnostics on OOF risk: rho(risk,stage) 0.11-0.23; R2 by TSS 0.04-0.05 (chance 0.029); HR per SD of risk
  1.3-1.4 raw/adjusted for stage/adjusted+TSS-stratified (rna 1.48/1.39/1.42, mirna 1.48/1.35/1.32, meth 1.32/1.28/1.29;
  p<=0.002) -> signals not simply stage or site proxies (diagnostic only, not a formal test).

## In-fold (pipeline) steps (src/mosurv/features.py + models.py; all fitted on training rows only)
- OmicsPrep, MethPrep (block-wise missingness filter + median impute), ClinPrep, StackPrep (clinical block unpenalised
  via penalty_factor=0 + penalised omics), elastic-net Cox nested tuning (l1 {0.1,0.5,0.9}, 30-alpha path from
  outer-train; prep re-fitted in every inner fold).

- M6a early integration (clinical unpenalised + penalised omics; same folds; Uno / Harrell diff vs clinical-only 0.636 / 0.657):
  +RNA   Uno 0.648 (+0.012 [-0.026,+0.050]); Harrell 0.686 (+0.029 [+0.005,+0.053], p=0.020, not primary)
  +miRNA Uno 0.626 (-0.011 [-0.045,+0.024]); Harrell 0.660 (+0.002)
  +meth  Uno 0.632 (-0.004 [-0.038,+0.031]); Harrell 0.668 (+0.010 [-0.018,+0.039])
  +all   Uno 0.636 (-0.000 [-0.030,+0.030]); Harrell 0.676 (+0.018 [-0.008,+0.044])
  INTERNAL CRITERION NOT MET for any early combination (honest report; CI half-width ~0.035 limits detectable gain).
  Stable RNA genes: ENSG00000196781.16 (selected 100% of folds with clinical), ENSG00000107984.10 (96%).
  Harrell p=0.020 for +RNA does not survive correction for 4 combinations and is not the pre-specified metric.

- M6b late integration (stacking of inner-OOF per-layer scores + clinical in a small Cox; same folds):
  Uno 0.634 [0.581,0.688], diff vs clinical-only -0.002 [-0.040,+0.035]; Harrell 0.672 (+0.015 [-0.020,+0.049]).
  vs early clinical+RNA: Uno -0.014 [-0.052,+0.025]. INTERNAL CRITERION NOT MET.
  Stacker weights (standardised): stage II/III/IV +0.32/+0.38/+0.25 (100% of folds), miRNA +0.22 (92%), RNA +0.14 (84%),
  meth +0.04 (60%), age +0.10, sex -0.03. Per-layer Harrell C reproduced exactly (rna .623, mirna .624, meth .600)
  -> the nested_fold return_oof refactor did not change single-omics results.
- BENCHMARK SO FAR (Uno C, same folds): clinical-only 0.636 | +RNA 0.648 | +miRNA 0.626 | +meth 0.632 |
  early all 0.636 | late stacking 0.634. No integration strategy meets the criterion (CI half-width ~0.035).

## Next
- Decision pending with user: finish M6 (MOFA+ factors -> Cox, RSF / gradient-boosted survival) vs go to freeze + external
  validation. Final-candidate selection must follow a rule fixed beforehand (e.g. simplest model within 1 SE of best),
  not best-C-index hunting.
- M7 full evaluation, M8 freeze signature_v1.json + GSE31210 validation (pre-specified), M9 interpretation.

## Reproduce
- `bash setup.sh && .venv/bin/python scripts/01_download.py && .venv/bin/python scripts/02_cohort.py`
  `&& .venv/bin/python scripts/03c_build_matrices_qc.py && .venv/bin/python scripts/03d_filter_meth_autosomal.py`
  `&& .venv/bin/python scripts/04_clinical_baseline.py && for l in rna mirna meth; do .venv/bin/python scripts/05_single_omics.py $l; done`
  `&& .venv/bin/python scripts/06_compare_diagnostics.py && for c in rna mirna meth all; do .venv/bin/python scripts/07_clinical_plus_omics.py $c; done`
  `&& .venv/bin/python scripts/08_late_integration.py`

## M6c/M6d results (fixed settings, no tuning, identical 5x5 folds; Uno C, tau = 5y)
| Model | Uno C | vs clinical Cox (0.636) |
|---|---|---|
| RSF clinical | 0.607 | -0.029 [-0.071, +0.013] |
| RSF clinical+omics | 0.579 | -0.057 [-0.136, +0.021] |
| GBM clinical+omics | 0.574 | -0.062 [-0.140, +0.015] |
| MOFA factors only (8) | 0.568 | -0.068 [-0.149, +0.013] |
| clinical + MOFA factors | 0.613 | -0.023 [-0.080, +0.034] |

- Every integration strategy (early, late, intermediate/MOFA, non-linear) fails the internal criterion; none beats clinical-only.
- Best Uno: clinical + RNA elastic-net 0.648 (+0.012 [-0.026, +0.050]); within 1 SE of clinical-only.
- MOFA factor signs are arbitrary per fold, so mean factor coefficients are not interpretable across folds.
- Milestone 6 complete. Logs: results/logs_09_nonlinear.txt, results/logs_10_mofa.txt (MOFA stdout is noisy; read the tables).
- Files: scripts/09_nonlinear.py, scripts/10_mofa.py, src/mosurv/mofa.py, tests/test_mofa.py.

## M6 decision and M7 (internal evaluation of the frozen candidates)
- Frozen candidates (decided before GEO): **clinical + RNA elastic-net** (best omics model; RNA is the only layer scorable on GEO arrays) with **clinical-only** as benchmark.
- M7a (OOF risks, mean of 5 repeats, patient-bootstrap CIs; anti-conservative because it ignores retraining variance):
  Uno C clin 0.622 vs clin+RNA 0.642, diff +0.020 [+0.001, +0.038]; AUC 1y +0.044 [+0.019, +0.074]; AUC 3y +0.026 [-0.005, +0.054]; AUC 5y +0.014 [-0.018, +0.043].
  Median-split HR high vs low: clin 2.84, clin+RNA 2.76 (RNA adds nothing visible there).
  The pre-specified criterion (Nadeau-Bengio CI) remains NOT met: +0.012 [-0.026, +0.050].
- M7b: calibration slope clin 0.84, clin+RNA 0.97 [0.70, 1.24]; LR test for RNA score added to clinical OOF score median p 4.6e-4 (optimistic, collinear scores);
  IBS 0.198 vs 0.193 (null 0.213); net benefit at 40% threshold 0.316 vs 0.336 (treat-all 0.323);
  PH p 0.082 (clin) / 0.026 (clin+RNA); 5y RMST low-minus-high ~410-417 days for both.
- Reading: small, consistent edge for clin+RNA on Brier/Harrell/1y AUC/calibration; no large gain; external validation decides.
- Files: scripts/11_m7a_discrimination.py, scripts/12_m7b_calibration.py, results/models/oof_risk_clinical.parquet, results/figures/km_risk_groups_oof.png, m7b_calibration_dca.png.
- Open design issue for M8: GEO is microarray (probe IDs), TCGA RNA is Ensembl; GSE31210 is stage I-II only so stage III/IV dummies are unscorable. Signature feature pool must be restricted to genes measurable on the arrays, and the clinical block must match what GEO provides, before refitting.

## M8: signature frozen (BEFORE any external scoring)
- Shared gene pool (TCGA ∩ GPL570 ∩ GPL96) = 11,753 genes (19% of TCGA features, 56% of RNA variance). Arrays are on a linear scale -> log2(x+1), then per-gene z-score within cohort.
- Restricting RNA to the shared pool costs nothing in CV: clin+RNA(shared) Harrell 0.690, Uno 0.650 (paired vs full pool +0.004 / +0.002, CIs include 0). Internal criterion still NOT met (Uno +0.014 [-0.023, +0.051]). DKK1 and TLE1 selected in 100% of folds.
- signature_v1 (scripts/15_freeze_signature.py): fit on 442 patients / 159 events; l1_ratio 0.9, alpha 0.1049, inner-CV Harrell 0.700; apparent Harrell 0.725 (optimistic).
  Genes (coef per SD): DKK1 +0.148, TLE1 +0.116, SEC61G +0.019, LDHA +0.010, USP4 -0.009. Clinical (per SD): age 0.064, male -0.011, stage II 0.344, III 0.430, IV 0.291.
  Frozen cutoff = TCGA median lp -0.166 (lp SD 0.583).
- results/models/signature_v1.json sha256 704878c704c0b13e3c1f4e3fb3e7671bf411eae7e13611f19342a47c8cd48041. DO NOT re-run 15_freeze_signature.py after external scoring starts.
- External plan is stored inside the JSON (primary GSE31210 all 226; sensitivity 204; secondary GSE68465 442 adenocarcinomas; no refit/tuning; frozen-cutoff primary split, within-cohort median as sensitivity; report RNA-only and clinical-only components).
- Files: scripts/13_shared_gene_pool.py, 14_combo_rna_shared.py, 15_freeze_signature.py, src/mosurv/signature.py, tests/test_signature.py (18 tests pass).

## M8e: EXTERNAL VALIDATION of frozen signature_v1 (run once; results/tables/external_validation.tsv, figures/external_km.png)
Cohorts: GSE31210 226 tumours/35 deaths (sensitivity 204/30 events); GSE68465 442 adenocarcinomas/236 deaths (stage derived from pT/pN; 3 missing stage scored as stage I). Arrays log2(x+1), per-gene z within cohort, no refit.
| Cohort | Full signature Harrell C [95% CI] | RNA part only | Clinical part only | HR high vs low (frozen cutoff) | Calibration slope |
|---|---|---|---|---|---|
| GSE31210 (226) | 0.715 [0.627, 0.792] | 0.522 [0.411, 0.631] | 0.693 [0.591, 0.786] | 4.36 [2.21, 8.60], p=2e-5 | 1.52 [0.81, 2.24] |
| GSE31210 (204) | 0.715 [0.608, 0.811] | 0.556 [0.450, 0.671] | 0.681 [0.562, 0.787] | 4.40 [2.13, 9.07], p=6e-5 | 1.62 [0.85, 2.39] |
| GSE68465 (442) | 0.697 [0.661, 0.734] | 0.574 [0.534, 0.611] | 0.691 [0.657, 0.725] | 2.90 [2.23, 3.77], p=1e-15 | 1.24 [1.00, 1.47] |
- Fixed criteria: C>0.60 with lower CI>0.50 MET in all; HR>1 with p<0.05 MET in all; calibration slope within 1+/-0.3 (rule I set before running): MET in GSE68465, NOT met in GSE31210 (point estimate 1.5-1.6, CI includes 1). Slope >1 = predictions too compressed (not over-fit).
- Reading: the full signature validates, but most discrimination comes from the clinical block (stage, age): clinical part alone 0.68-0.69 vs full 0.70-0.72. The RNA part alone is not distinguishable from chance in GSE31210 (stage I-II, 35 events, low power) and is small but real in GSE68465 (C 0.574, HR/SD 1.24, adds to clinical LR p=0.002; HR/SD 1.92 after adjusting for adjuvant chemo).
- Honest claim: a clinical+RNA prognostic model replicates externally; an independent RNA contribution is supported by one large cohort and not detectable in the other. Internal CV criterion for gain over clinical-only was NOT met.
- Incident: first run crashed at the adjuvant-chemo adjustment (index misalignment bug); fixed mechanically and re-run; first-pass numbers were identical. A stale-file delivery also required a re-send. GSE31210 characteristic rows are ragged: parse by cell label, not row.
