# Revision changelog - Paper 445

Chronological within each section. "Codex" = the earlier session; "this session" = the continuation.

## Code (research software)

### Codex (retained)
* New validated package `src/cropxai/` (data audit, registry-checked models, metrics, input-times-gradient explanations, deletion controls, feature cache, immutable provenance, unified CLI).
* Legacy `src/research/xai/trust.py`: the constants 0.65/0.50 returned on ImportError / missing layer replaced by `raise` ("NOT COMPUTED"). Legacy registry no longer substitutes ViT for Prithvi.
* `tests/test_revision.py` (25 tests), `scripts/*` first versions, JSON evaluation/experiment configs.

### This session
| Change | Why |
|---|---|
| `src/cropxai/inputs.py` (new): `InputSpec` for RGB and six-band representations; unit-scale corruption semantics; GeoTIFF band order and Prithvi mapping constants | One source of truth for channels/normalization; band sets are never inferred or padded |
| `src/cropxai/data.py`: `TifFolder` (GeoTIFF view of the JPEG split, same sample IDs), `validate_bands` generalized, `band_order_evidence`, `audit_ms_dataset` (hash manifest, JPEG<->GeoTIFF pairing, band-order assertions) | Genuine multispectral input for Prithvi; the audit found B8A is stored **last**, which would have silently mis-selected a band under the nominal order |
| `src/cropxai/models.py`: `Prithvi-EO-1.0-100M` (vendored Apache-2.0 encoder, pinned revision, SHA-256 verified at load, one-frame position embedding verified against checkpoint, mean-pooled linear head); `model.input_spec` | Replace the withdrawn proxy with the real checkpoint (R3.4) |
| `src/cropxai/vendor/prithvi_mae.py` + `NOTICE.md` | Upstream model code, unmodified, licence retained |
| `explainability.py`, `feature_cache.py`, `experiment.py`: spec-aware corruption; configuration/model/tensor consistency assertions on real data; cache identity includes the input spec | Same assertions for both cohorts (R3.5) |
| `provenance.source_hash`: only `src/cropxai/**.py` (excluding `cli.py`) | Editing analysis/paper scripts must not split a run cohort |
| `src/cropxai_analysis/stats.py` (new): t-interval, paired seed differences, test-item bootstrap, Bonferroni level, rank correlations | Statistics outside the hashed run-determining code; 7 tests |
| `src/cropxai_analysis/compliance.py` (new) + `src/research/validation/benchmark_validator.py` rewritten as a wrapper + `scripts/check_compliance.py` + `cropxai compliance` | The old checker was either unconditional success or `NotImplementedError` with dead code; the new one recomputes 15 assertions and exits 1 on failure; 15 negative tests |
| `scripts/analyze_results.py` rewritten: cohorts, current/superseded source revisions, derived metrics (accuracy retention, deletion controls, AOPC), per-corruption intervals, validation history, paired comparisons, weight sensitivity (RGB-only and a flagged confounded cross-input set), failed runs separated | R2.4, R3.2, R3.3 |
| `scripts/audit_legacy.py` (new) | Reproduces the submitted Table I; detects fallback runs, the Prithvi=ViT proxy, the database divergence |
| `scripts/rerun_legacy_trainer.py` (new) | Re-executes the unchanged legacy trainer with `store_run` redirected to capture per-epoch validation values |
| `scripts/generate_figures.py` rewritten (9 figures, 5 LaTeX tables + JSON with hashes), `scripts/build_manuscript.py` rewritten (all numbers from records), `scripts/validate_manuscript.py` rewritten (gate with claim-level checks), `scripts/make_input_audit.py`, `scripts/package_revision.py`, `scripts/reviewer_reproduce.py` | Result-driven paper, reviewer package |
| `cropxai` CLI: `compliance`, `audit-legacy`; `paper package` now resolves | One entry point for every step (pyproject already targets it) |

Tests: `tests/test_revision_v2.py` (new) covers input specs, band mapping on synthetic GeoTIFFs, band-order evidence (positive and negative), Prithvi construction/identity/rejection, statistics, run-identity hashing, 15 compliance-failure cases, fallback elimination, legacy reconstruction and database divergence, CLI.

## Experiments
| Item | State |
|---|---|
| Dry-run campaign (earlier source revision) | Stopped; its records kept but superseded. One orphaned child process was found running old code after `pkill` of the wrapper and was terminated; no record was written by it |
| RGB cohort: ResNet-18, ViT-Base, ConvNeXt-T x seeds 42,100,999,1234,2026 x 5 epochs | See SUBMISSION_CHECKLIST.md for the executed count |
| Six-band cohort: Prithvi-EO-1.0-100M x 5 seeds x 5 epochs | See SUBMISSION_CHECKLIST.md |
| Legacy trainer rerun (ResNet-18, seed 42, 5 epochs) | See EXPERIMENT_DISCREPANCIES.md |

## Results, figures, tables
All regenerated from validated records: `results/processed/{revision_summary,paired_comparisons,weight_sensitivity}.json`, `results/figures/*` (+ JSON sidecars), `results/tables/*`. Submitted figures and tables are preserved in the archive and not reused.

## Manuscript claims changed
| Submitted claim | Revised |
|---|---|
| Five seeds, five epochs, Table 1 means | Table 1 pooled 1- and 5-epoch runs with three seeds; replaced by a homogeneous 5-seed, 5-epoch cohort |
| "Prithvi-100M" results | Proxy (ViT-Base on RGB) withdrawn; real Prithvi-EO-1.0-100M evaluated on six bands |
| Multispectral/13-band framing | RGB for ImageNet models, six Sentinel-2 bands for Prithvi; no 13-band experiment |
| Bayesian Trust Score | Composite Trust Score (weighted sum, not Bayesian) |
| SSIM = faithfulness | SSIM = explanation stability; faithfulness = deletion AUC with random/least-salient controls |
| Six pillars | All six computed and tabulated with seed-level intervals |
| Calibration, geographic drift, Friedman/Nemenyi, critical-difference diagram | Calibration computed (ECE/NLL/Brier, reliability diagram); synthetic corruption only (labelled); Nemenyi withdrawn, replaced by exploratory paired analysis |
| Figure 7 / Table 1 | Explained: different executions of one configuration under a reused run identifier (see EXPERIMENT_DISCREPANCIES.md) |
| Crop-health framing | Land-cover classification only; crop-health uses are future work |
| Reference list | Rebuilt from verified primary records (Crossref/IEEE/arXiv); EuroSAT now cites the journal version |

## Not changed / preserved
Legacy databases (both), original paper tree, legacy `src/research` prototypes, data, venv. `paper/references.bib` was momentarily overwritten by mistake in this session and restored byte-for-byte from the preservation archive.

## Final-completion pass (2026-10-23)
* **Clean-environment install (VERIFIED):** new venv (`python3 -m venv`, Python 3.14.8, macOS 26.3 arm64) outside the repo; `pip install -r requirements.txt`. The first install exposed a real defect: `requirements.txt` lacked `rasterio`, `einops`, `timm` (7 test errors). Added `rasterio==1.5.0`, `einops==0.8.2`, `timm==1.0.27`, `huggingface_hub==1.22.0`. After reinstall: numpy 2.5.1, scipy 1.18.0, scikit-learn 1.9.0, scikit-image 0.26.0, torch 2.12.1, torchvision 0.27.1, Pillow 12.3.0, matplotlib 3.11.0; **76 tests OK**; `cropxai compliance` PASS for both cohorts. The existing venv was not touched.
* Claim audit (`evidence/proofreading_notes.md`): Section 5 corrected - the figure registered 90 s after the run is `baseline_results.pdf`, which prints the reviewer-database values (85.19 %, T3 0.2394); the convergence figure is a different file; "deterministic" softened to "reproducible (one re-execution)"; legacy tables verified identical to the archive; "Level-1C" qualified as inferred (the EuroSAT paper does not state the processing level).
* New: `scripts/scan_legacy_placeholders.py` + `LEGACY_PLACEHOLDER_SCAN.md` + a test that the active path has no mock markers; `submitted_figure_files` in `legacy_audit.json` + test.
* Not done (inputs missing): similarity report, conference notification/author instructions (official Procedia guide still HTTP 403), Docker (daemon not running).
