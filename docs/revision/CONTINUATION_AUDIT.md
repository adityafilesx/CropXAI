# Continuation audit - Paper 445

Written 2026-10-22 at the start of the continuation session; sections marked **(updated)** were revised at the end of the session with the final state.
Labels: VERIFIED (re-checked by command or test in this session), COMPLETED-NOT-INDEPENDENTLY-VERIFIED, BLOCKED, NOT STARTED.

## 1. Repository state found

* Not a Git repository (`git status` fails). No history exists, so "Codex changes" were identified from file timestamps (21-22 Oct 2026) and from `docs/revision/INITIAL_AUDIT.md`. Git was not initialised.
* Original evidence is preserved in `archives/pre_revision_445/original_source_and_evidence.tar.gz` (SHA-256 `f6d59dcf...`, with `inventory.json` and `sha256.json`). It contains both SQLite stores, the original `paper/` tree and legacy sources.
* Python 3.14.8 venv (`venv/`), torch 2.12.1 with MPS available, no CUDA. 16 GB RAM, 10 CPU cores.

## 2. What Codex already did (retained)

| Area | Retained work | Status in this session |
|---|---|---|
| New package | `src/cropxai/{data,models,metrics,explainability,feature_cache,experiment,provenance,cli}.py`, `metric_registry.json` | Reviewed line by line; extended (see changelog), not replaced |
| Scripts | `scripts/{analyze_results,generate_figures,validate_manuscript,build_manuscript,validate_dataset,run_campaign}.py` | analyze/figures/dataset/manuscript builders rewritten for two cohorts; validators tightened |
| Tests | `tests/test_revision.py` (25 tests) | All pass; kept unchanged |
| Legacy fixes | `src/research/xai/trust.py` raises instead of returning 0.65/0.50; legacy registry no longer substitutes models | VERIFIED by new tests |
| Docs | `INITIAL_AUDIT.md`, `LITERATURE_COMPARISON.md`, `reviewer_comments.json`, `evidence/*` | Kept; statuses revised from evidence |
| Paper | `paper/source/manuscript_template.tex` (+ built PDFs) | Template substantially revised; PDFs rebuilt |
| Runs | 1 smoke run (ResNet-18, 1 epoch, 500/200/200 images) | Kept as `real_data_smoke`; never aggregated with benchmarks |

Codex's central design decisions were confirmed rather than reversed: RGB is the true representation of the JPEG pipeline; legacy proxies are withdrawn; the protocol is a frozen-backbone linear probe with cached features.

## 3. Findings in the prompt, independently checked

| # | Finding | Result |
|---|---|---|
| 1 | Mixed 1- and 5-epoch aggregation | **VERIFIED and reproduced numerically.** `scripts/audit_legacy.py` rebuilds the submitted Table I: ResNet-18 91.53% = mean over 3 one-epoch + 3 five-epoch runs (90.54 and 92.52); ViT-Base 95.41%; "Prithvi" 95.30% (3+2 runs). Five-epoch-only means are 92.52 / 96.07 / 96.14. Only three distinct seeds exist, not five. Evidence: `docs/revision/evidence/legacy_audit.json`. |
| 2 | Trust metric narrower than described; fixed fallbacks | **VERIFIED.** Original `trust.py` (extracted from the archive) had `return 0.65` on ImportError and `return 0.50` when no Grad-CAM layer was found. Five legacy runs have `trust_T3` exactly 0.5 (all three 1-epoch "Prithvi" runs, ViT-Base seeds 42 and 999 at one epoch): their trust values are fallbacks and are invalid. |
| 3 | Unsupported calibration / drift / Nemenyi claims | **VERIFIED** absent from the legacy pipeline; now replaced by computed ECE/Brier/NLL, synthetic corruption (labelled as such), and a paired analysis. Geographic drift is not evaluated. |
| 4 | RGB instead of multispectral | **VERIFIED for the legacy and RGB cohort** (3-channel JPEG, ImageNet statistics). **New:** 13-band GeoTIFFs exist locally and a genuine six-band Prithvi cohort was built (Section 5). |
| 5 | Reproducibility infrastructure | Two CLIs: pyproject now targets `src.cropxai.cli:main` (legacy Hydra CLI kept, warning-only train commands). Compliance checker: found half-fixed (`NotImplementedError` + dead code) - replaced by a real checker with 15 negative tests. Reviewer package: rebuilt by `scripts/package_revision.py`. Tests: 25 -> see Section 6. |
| 6 | Syntax check only | Agreed; no claim here rests on it. |
| Prompt 5 note | "Prithvi-100M" is a ViT proxy | **VERIFIED**: accuracy, F1, precision, recall of "Prithvi-100M" are bit-identical to ViT-Base in all five shared (epochs, seed) cells. |

## 4. Authoritative sources

* Results: `results/raw/<run_id>.json` (immutable) mirrored in `results/experiment_db.sqlite::revision_runs`. The legacy tables `experiments`/`metrics` are historical evidence only. The reviewer-package database was a stale subset (10 of 22 legacy runs, no value differing); it is preserved under `reviewer_package/legacy_stale_reviewer_db/`.
* Manuscript: `paper/source/manuscript_template.tex` + `paper/source/references.bib` -> generated `paper/source/445_revised_version.tex` (do not edit the generated file). `paper/main.tex` and `paper/references.bib` are legacy, non-authoritative.
* Template: `Paper format /` (2024 Procedia files). Current conference/Elsevier compliance is NOT confirmed.

## 5. Experiments recoverable / executed (updated)

* Legacy runs: no checkpoints, no epoch logs; **not recoverable**. Preserved as `RECOVERED_UNVERIFIED` summaries (`results/processed/legacy_*.json`).
* Executed in this session: RGB cohort 15/15 runs (ResNet-18, ViT-Base, ConvNeXt-T x 5 seeds x 5 epochs); Prithvi-EO-1.0-100M six-band cohort 5/5 runs (one interruption, resumed under the identical source hash); legacy-trainer rerun (ResNet-18, seed 42, 5 epochs) for the Figure 7 investigation. All records pass `cropxai compliance` and are recomputed by `cropxai analyze`.
* Superseded: one dry-run campaign (earlier source revision, 6 records in the old schema, reported as `superseded_unvalidated`), partial run directories without records (`05b0db0d...`, two from the stalled Prithvi attempt, one from the interrupted fourth seed).

## 6. Tests that actually execute (updated)

`python -m unittest discover -s tests -p "test_revision*.py"` -> **76 tests, OK** (25 by Codex, 51 new): input specs and band mapping on synthetic GeoTIFFs, band-order evidence (pass and fail), Prithvi construction/identity/tamper rejection, statistics, run-identity hashing, 15 compliance-failure cases, fallback elimination, legacy reconstruction and database divergence, CLI, and generated-asset consistency. `tests/dataset`, `tests/integration`, `tests/e2e` etc. remain empty directories.

## 7. Remaining at the end of the session

See `SUBMISSION_CHECKLIST.md` (similarity report, conference format, language proofreading, Docker verification, manual portal steps).
