# Process checklist against the continuation prompt (Parts A-N)

Labels: **VERIFIED** (re-checked by command/test) · **DONE-NV** (completed, not independently verifiable by software) · **BLOCKED** · **TODO-AUTHORS** (only the authors can do it) · **N/A**.
Date of this pass: 2026-10-23. Detail for every row lives in the named file.

## Part A - Context and conference requirements

| # | Requirement | Status | Evidence / note |
|---|---|---|---|
| A1 | Title kept; scope made explicit | VERIFIED | manuscript Sec. 1; gate `scope_statement` |
| A1-1 | Corrections for all reviewer comments | VERIFIED (16/16 addressed; R2.3 DONE-NV) | `REVIEWER_TRACEABILITY.md` |
| A1-2 | English-language proofreading | **TODO-AUTHORS** | software proofreading only; two passes done |
| A1-3 | Proper citations throughout | VERIFIED (mechanical) | 19 entries checked against Crossref/arXiv/model card; gate `citations_resolve`, `verified_bibliography` |
| A1-4 | Citations for reused figures | N/A | no published figure is reused; all 9 figures generated from run records |
| A1-5 | Procedia template | DONE-NV | conference-supplied `PROCS_ICMLDE2024` class files used; current Elsevier guide not fetchable (403); page limit unknown (19 pp. incl. response) |
| A1-6 | Detailed point-by-point response | VERIFIED | `paper/response_to_reviewers.md`, PDF pp. 16-19 |
| A1-7 | Manuscript + response in one file | VERIFIED | `output/pdf/445_revised_version.pdf` |
| A1-8 | Genuine similarity report | **TODO-AUTHORS** | `SIMILARITY_REPORT.md`; none fabricated |
| A1-9 | File named `445_revised_version` | VERIFIED | `output/pdf/445_revised_version.pdf` |
| A1-10 | Acknowledgement email to info@icmlde.org | **TODO-AUTHORS** | manual |
| A2 | Codex findings 1-6 treated as hypotheses | VERIFIED | `CONTINUATION_AUDIT.md` Sec. 3 (all confirmed; plus proxy identity, DB divergence, mock legacy code) |

## Part B - Non-negotiable rules

| Rule | Status | How it was honoured |
|---|---|---|
| 1-2 no fabricated results/CIs/fallback scores | VERIFIED | every number is read from run records; active path has 0 mock markers (test); no fallback constants |
| 3 no silent mixing of epochs/splits/checkpoints/bands/preprocessing | VERIFIED | `cropxai compliance` + analysis grouping key; 15 negative tests |
| 4 no "validated because tests pass" | VERIFIED | gate rechecks claims; open items listed |
| 5-6 preserve originals; never overwrite the only copy | VERIFIED with one incident | archive + both DBs kept; `paper/references.bib` briefly overwritten and restored byte-for-byte from the archive (documented) |
| 7 keep Codex changes | VERIFIED | `CONTINUATION_AUDIT.md` Sec. 2 |
| 8 no invented multispectral adapter | VERIFIED | Prithvi uses its own six bands, no added parameters; RGB models never given extra channels (test) |
| 9 stability != faithfulness | VERIFIED | separate metrics, separate sections |
| 10 weighted sum != Bayesian | VERIFIED | "Composite Trust Score"; gate `composite_not_bayesian` |
| 11 no unexecuted baseline claimed | VERIFIED | literature comparison only |
| 12-13 no destructive commands, no history rewrite | VERIFIED | none used (not a Git repo); one orphaned process terminated, no data touched |
| 14 no fake similarity report | VERIFIED | none exists |
| 15 status labels | VERIFIED | used throughout |
| 16 blockers documented, work continued | VERIFIED | D8 (stall/resume), Docker, common-input comparison |
| 17 comments resolved only with evidence | VERIFIED | statuses derive from checks; R2.3 left DONE-NV |
| 18 change claims rather than invalidate science | VERIFIED | proxies withdrawn, claims narrowed |

## Part C - Phase 0 (resume)

| Item | Status |
|---|---|
| pwd / git commands | VERIFIED: not a Git repository (no branch, diff or log exist); timestamps used instead |
| Read Codex attachment and existing audit files | VERIFIED |
| `docs/revision/CONTINUATION_AUDIT.md` | VERIFIED written and updated |
| Began changes after the audit | VERIFIED |

## Part D - One authoritative data source

| Item | Status | Evidence |
|---|---|---|
| D1 inspect all stores; why DBs differ; preserve both | VERIFIED | same `run_id` carries different metrics (INSERT OR REPLACE); `EXPERIMENT_DISCREPANCIES.md` D1/D5 |
| D2 machine-readable run identity | VERIFIED except two fields | record schema v2: config hash, checkpoint SHA-256, split id, seed, epochs, learning rate, batch size, freeze policy, input spec, environment packages, source hash, artifact hashes. The optimizer (Adam) is implicit in the hashed code, not a config field; Git revision N/A (no Git) |
| D3 repair mixed-epoch aggregation; preserve original; corrected cohort; comparison table | VERIFIED | `audit_legacy.py`; manuscript Table 6; 5 seeds x 5 epochs, 3 RGB + 1 six-band model |
| D4 data-integrity checks fail on violations | VERIFIED | `compliance.py`, tests |

## Part E - Input modality

| Item | Status | Evidence |
|---|---|---|
| E1 verify pipeline (RGB JPEG, 3 channels, ImageNet norm.) | VERIFIED | `MODEL_INPUT_AUDIT.md` |
| E2 Direction A (RGB) for ImageNet models **and** Direction B for Prithvi | VERIFIED | real checkpoint, GeoTIFF bands, tests; domain gaps disclosed |
| Sentinel-2 vs HLS band equivalence not assumed | VERIFIED | mapping table; B8A stored last (empirical audit, negative test) |
| E3 model-input audit | VERIFIED | generated from run records |

## Part F - Trust metrics

| Item | Status |
|---|---|
| F1 stability defined and tested | VERIFIED |
| F2 deletion faithfulness (AUC, AOPC, random/least controls, known-model tests) | VERIFIED |
| F3 fallback search and elimination (legacy + active) | VERIFIED: 5 contaminated legacy runs; legacy function raises; mock-code scan `LEGACY_PLACEHOLDER_SCAN.md` |
| F4 six pillars computed from real outputs; each documented | VERIFIED: `metric_registry.json`, manuscript Sec. 3.3-3.5 |
| F5 composite terminology, weights versioned, no composite if a component is missing | VERIFIED |
| F6 weight sensitivity | VERIFIED; plus an explicitly post-hoc check that reverses the ranking |

## Part G - Statistics and robustness

| Item | Status |
|---|---|
| G1 n, mean, SD, 95 % t-CI, missing runs per model and pillar | VERIFIED (Tables 1 and 5) |
| G2 calibration: ECE/NLL/Brier, reliability diagram from same predictions | VERIFIED |
| G3 OOD: synthetic corruptions only, labelled; geographic drift = future work | VERIFIED / NOT STARTED (real shift) |
| G4 Friedman/Nemenyi replaced by paired seed-level and test-item analyses; no significance from plots | VERIFIED |

## Part H - Infrastructure

| Item | Status |
|---|---|
| H1 one CLI (`./cropxai`, pyproject targets it); legacy Hydra CLI documented | VERIFIED |
| H2 compliance checker real, exit 1 on failure, valid/invalid tests | VERIFIED |
| H3 reviewer package complete, correct DB, lightweight path | VERIFIED from an isolated copy (MANIFEST 458/458 hashes); **Docker build BLOCKED** (daemon not running on 2026-10-23) |
| H4 real tests | VERIFIED: 76 tests OK (also in a clean venv) |

## Part I - Reviewer comments (R1.1-R3.6)

All 16 have status, code change, evidence, location and page in `REVIEWER_TRACEABILITY.md`: 15 VERIFIED, R2.3 DONE-NV (Figure 7 source inferred, not provable). R3.5 sub-part "controlled common-input comparison" is BLOCKED and disclosed.

## Part J - Manuscript

| Item | Status |
|---|---|
| J1 abstract reports only verified metrics, real input, no unsupported claims | VERIFIED (gate `abstract_numbers_traceable`) |
| J2 methodology matches implementation; symbols defined | DONE-NV (read against code; `L`, `K` defined in this pass) |
| J3 tables generated from machine-readable data | VERIFIED |
| J4 discussion/conclusion within evidence | VERIFIED |
| J5 references/figures/captions | VERIFIED |
| J6 Procedia template check | DONE-NV / TODO-AUTHORS (see A1-5) |

## Part K - Paper-result validator (17 checks)

`./cropxai validate-manuscript`: 138 checks, **136 pass**; the 2 failures are the external items (similarity report, conference format). Mapping: 1 models registered (`registered_*`) · 2 metrics in store (`table_*`) · 3 seed count (`seed_count_matches`, `complete_group_*`) · 4 epochs consistent (`protocol_homogeneous`) · 5 input matches (`input_representation`, `band_mismatch_disclosed`) · 6 six pillars (`six_pillars_*`) · 7 weights (`weights_documented`) · 8 no fallback masquerade (active-path test) · 9 CIs from real runs (`ci_present_*`) · 10-11 tables/figures match data (`table_*`, `figure_*`) · 12 Fig. 7/Table I explained (`discrepancy_explained`) · 13 calibration/significance (`nemenyi_withdrawn`, tables) · 14-15 citations, numbering (`citations_resolve`, `references_resolve`, `labels_unique`) · 16 no crop-health claim (`no_crop_health_claim`) · 17 incomplete experiments disclosed (`failed_runs_disclosed`, limitations).

## Part L - Deliverables

| Item | Status |
|---|---|
| L1 code, commands, tests, checker, reviewer package | VERIFIED (Docker excepted) |
| L2 audit documents: CONTINUATION_AUDIT, MODEL_INPUT_AUDIT, EXPERIMENT_DISCREPANCIES, REVIEWER_TRACEABILITY, REVISION_CHANGELOG, SUBMISSION_CHECKLIST (+ LITERATURE_COMPARISON, LEGACY_PLACEHOLDER_SCAN, SIMILARITY_REPORT, this file) | VERIFIED |
| L3 manuscript PDF + response in one file; separate response sheet | VERIFIED |
| L4 similarity report | TODO-AUTHORS |
| L5 submission checklist | VERIFIED |

## Part M - Stages 1-9

1 audit ✔ · 2 investigate (DBs, epochs, modality, fallbacks, Fig. 7) ✔ · 3 consistent cohort ✔ · 4 metrics/statistics ✔ · 5 tests/checker/CLI/package ✔ · 6 experiments run ✔ (20 runs + legacy rerun) · 7 figures/tables ✔ · 8 manuscript/response ✔ · 9 checks and blockers ✔ (this file).

## Part N - Final report

Delivered in the chat summary; blockers repeated in `SUBMISSION_CHECKLIST.md`.

## Remaining for the authors

Similarity report · confirm Procedia template/page limit/filename · professional English proofreading and read-through · Docker check and clean install · email and portal upload.
