# Submission checklist - Paper 445 (ICMLDE 5.0)

Status labels: **VERIFIED** (checked by command/test in this session) · **COMPLETED, NOT INDEPENDENTLY VERIFIED** · **BLOCKED** · **NOT STARTED**.
**The revision is NOT submission-ready**: two items below can only be completed by the authors (similarity report, conference format) and several judgement items need their review.

## Deliverables and filenames

| Item | File | Status |
|---|---|---|
| Revised manuscript + point-by-point response in one PDF (19 pp.; response from p. 16) | `output/pdf/445_revised_version.pdf` | VERIFIED built (gate: PDF newer than source; all references resolve) |
| Separate response sheet (5 pp.), same content | `output/pdf/reviewer_response_sheet.pdf` | VERIFIED built |
| Editable source | `paper/source/manuscript_template.tex` (text) -> generated `paper/source/445_revised_version.tex`; `paper/source/references.bib` | VERIFIED (compiles) |
| Word version | - | NOT STARTED (no DOCX build path; LaTeX source is the editable source) |
| Response in Markdown | `paper/response_to_reviewers.md`; `docs/revision/REVIEWER_TRACEABILITY.md` | VERIFIED generated from the same records |
| Similarity report | `output/pdf/similarity_report.pdf` | **TODO-AUTHORS** - file not provided; see `SIMILARITY_REPORT.md` |
| Email acknowledgement to info@icmlde.org | `docs/revision/evidence/ack_email_draft.txt` | DRAFT only, NOT SENT (needs notification date; awaiting your explicit confirmation) |
| Portal upload | - | NOT STARTED (manual) |

## Experiments and evidence

| Item | Status | Evidence |
|---|---|---|
| RGB cohort: ResNet-18, ViT-Base/16, ConvNeXt-T x 5 seeds (42,100,999,1234,2026) x 5 epochs, full 18,900/4,050/4,050 split | VERIFIED: 15/15 runs completed, 0 failed | `results/raw`, compliance report PASS |
| Six-band cohort: Prithvi-EO-1.0-100M x 5 seeds x 5 epochs | VERIFIED: 5/5 completed (3 + 2 after a process interruption, same source hash) | compliance PASS; `EXPERIMENT_DISCREPANCIES.md` D8 |
| Homogeneous protocol, one source revision, same test items, no mixed epochs | VERIFIED by `cropxai compliance` (15 checks per cohort, 15 negative tests) | `docs/revision/evidence/compliance_report.json` |
| Every run record recomputed from its saved predictions/explanations/corruptions | VERIFIED | `scripts/analyze_results.py::validate_record` (24 records valid, 0 rejected) |
| Mixed-epoch legacy aggregation explained | VERIFIED | `legacy_audit.json`, tests |
| Figure 7 vs Table 1 | COMPLETED, NOT INDEPENDENTLY VERIFIED: mechanism verified (overwritten metrics under a reused run id; legacy rerun reproduces the main-DB values exactly); that the figure came from the earlier execution is inferred | `EXPERIMENT_DISCREPANCIES.md` D1 |
| Fallback trust values removed / legacy contamination identified | VERIFIED (5 runs); legacy trust function raises | tests `FallbackEliminationTests` |
| Input modality | VERIFIED: RGB for three models; six Sentinel-2 bands for Prithvi; **no 13-band experiment** | `MODEL_INPUT_AUDIT.md` |
| Prithvi band mapping (incl. B8A stored last in the GeoTIFFs) | VERIFIED empirically + unit tests; domain gaps (L1C vs HLS, 64->224, T=1) uncorrected and disclosed | `MODEL_INPUT_AUDIT.md` |
| Geographic / real distribution shift, field-level crop tasks | NOT STARTED (future work, stated) | manuscript Section 6 |
| Common-input controlled comparison RGB vs six-band | BLOCKED (no validated adapter) - disclosed | R3.5 response |
| Post-hoc control-normalized explanation analysis | COMPLETED; explicitly labelled post-hoc; changes the composite ranking | manuscript Section 4.5 |

## Metrics and statistics

| Item | Status |
|---|---|
| Six pillars computed for every model/seed, composite rejects missing/invalid components | VERIFIED (compliance `metrics_valid`) |
| Composite is a weighted sum (not Bayesian); weights versioned in `configs/evaluation/revision_v1.json`, fixed before the final campaign | VERIFIED |
| Stability (SSIM) vs faithfulness (deletion AUC/AOPC + random/least-salient controls) separated | VERIFIED (tests, results) |
| Seed-level SD and 95 % t-intervals, n = 5, missing counts | VERIFIED (`table1.json`, `table_ci.tex`) |
| Test-item bootstrap, paired comparisons (Bonferroni), exploratory only | VERIFIED |
| Calibration (ECE, NLL, Brier, reliability diagram from the same predictions/bins) | VERIFIED |
| Friedman/Nemenyi significance, critical-difference diagram | WITHDRAWN (design unsupported); stated in text |
| Weight sensitivity (equal, two profiles, 50 perturbations) | VERIFIED; ResNet-18 ranks first in all 54 weight configurations, the order of the other models differs from the default in 2 (the explanation-priority profile and one random draw). This stability is only for the pre-declared explanation component; the post-hoc variant reverses the leader |

## Tests and infrastructure

| Item | Status |
|---|---|
| `python -m unittest discover -s tests -p "test_revision*.py"` | VERIFIED: 76 tests, OK (25 from Codex + 51 new); also OK in a clean venv. One real-data integration test needs `data/processed/eurosat` |
| Compliance checker fails on protocol violations | VERIFIED (15 negative tests; unconditional-success path removed) |
| CLI | VERIFIED: one entry point `./cropxai` (pyproject targets it); legacy Hydra CLI retained but not advertised |
| Manuscript gate `./cropxai validate-manuscript` | 136 of 138 checks pass; the two failures are the external items above and are intentionally not suppressed |
| Reviewer package (`reviewer_package/`, 757 MB, `MANIFEST.json`) | VERIFIED executable from an isolated copy with `python scripts/reviewer_reproduce.py` (3 PASS lines). **Docker build/run NOT verified** - the Docker daemon was not running when re-checked on 2026-10-23 (BLOCKED until you start it) |
| Database mismatch | VERIFIED explained (D1/D5); authoritative source defined; stale snapshot preserved in `reviewer_package/legacy_stale_reviewer_db/` |
| Clean-environment install (`requirements.txt`) | VERIFIED 2026-10-23: fresh venv, 76 tests OK, compliance PASS; defect found and fixed (missing rasterio/einops/timm pins) |

## Manuscript quality

| Item | Status |
|---|---|
| Citations resolve, every entry cited, every entry verified against a primary record (Crossref/arXiv/model card) | VERIFIED (19 entries) |
| Figure sources / captions; figures regenerated from validated data with JSON sidecars | VERIFIED; no figure from published work is reused |
| Equations match implementation | COMPLETED, NOT INDEPENDENTLY VERIFIED (read against code; no automated equation test) |
| English proofreading | **NOT DONE by a professional/native editor**; text was proofread by the revising software only - the authors must do the language pass the conference requires |
| Procedia template | COMPLETED, NOT INDEPENDENTLY VERIFIED: the conference-supplied 2024 `PROCS_ICMLDE2024` class files were used and the PDF compiles; the current Elsevier guide could not be fetched (HTTP 403) and no page limit was found in the supplied materials; 19 pages including the response may exceed the limit |
| Overfull/underfull boxes | A few minor ones remain in `docs/revision/evidence/445_revised_version_build.log` |
| Acknowledgement/funding, licence, release URL | NOT VERIFIED - no public repository or licence decision is asserted |

## Reviewer comments (details: `REVIEWER_TRACEABILITY.md`)

| ID | Status | Note |
|---|---|---|
| R1.1 citations | VERIFIED (mechanical checks) | Authors should still read every sentence for support |
| R1.2 dataset citation/description | VERIFIED | |
| R1.3 metric explanation | VERIFIED | |
| R1.4 limitations | VERIFIED | |
| R1.5 graphical findings | VERIFIED | 9 figures |
| R1.6 comparison with published work | VERIFIED | literature comparison only; no baseline from the literature was re-run |
| R2.1 implementation description | VERIFIED | |
| R2.2 comparison with published approaches | VERIFIED | |
| R2.3 Figure 7 / Table 1 | COMPLETED, NOT INDEPENDENTLY VERIFIED | see above |
| R2.4 SD and 95 % CI | VERIFIED | n = 5 per model |
| R3.1 stability vs faithfulness | VERIFIED | |
| R3.2 composite / Bayesian | VERIFIED | |
| R3.3 six pillars | VERIFIED | |
| R3.4 Prithvi band adaptation | VERIFIED | real checkpoint executed |
| R3.5 same bands? | VERIFIED (audit + disclosure) | controlled common-input comparison BLOCKED |
| R3.6 crop-health framing | VERIFIED | |

The reviewer wording follows the author-supplied brief; the original notification email was not available to the software, so the authors should compare the comment text with the real notification before submitting.

## Remaining work before the authors can truthfully submit

1. Obtain and check the genuine similarity report (<= 10 %).
2. Confirm the conference's current Procedia template, page limit and filename rule; adjust length if needed (e.g. move Tables 2-5 and some figures to an appendix).
3. Professional English proofreading and an author read-through of every claim, especially Section 5 (audit) and Section 6.
4. Decide whether to keep the post-hoc explanation analysis in the main text (recommended, since it changes the ranking).
5. Verify the Docker path on a machine with Docker running; do a clean-environment install.
6. Send the acknowledgement email to info@icmlde.org and upload through the portal (manual).
