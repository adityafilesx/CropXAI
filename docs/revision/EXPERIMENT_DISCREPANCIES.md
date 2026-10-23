# Experiment discrepancies - Paper 445

All numbers below are reproduced by `scripts/audit_legacy.py` (output: `evidence/legacy_audit.json`) and `scripts/rerun_legacy_trainer.py` (output: `evidence/legacy_rerun_resnet18_seed42_ep5.{log,json}`), and are guarded by tests in `tests/test_revision_v2.py`. Labels: VERIFIED = recomputed from stored data in this session; INFERRED = consistent with the evidence but not provable from what was preserved.

## D1. Table 1 (91.53%) versus Figure 7 (validation ~84-86%) - R2.3

| Item | Value | Source |
|---|---|---|
| Submitted Table I, ResNet-18 accuracy | 91.53% | `evidence/original_manuscript.txt` |
| Submitted convergence figure (Fig. 4 in the file, "Figure 7" in the reviewer's wording) | validation ~84-86%, training loss ~0.80 -> ~0.50 | same text |
| Main DB (`results/experiment_db.sqlite`), run `96f1e21d95740cc2` (ResNet-18, seed 42, 5 epochs) | accuracy 92.37%, final loss 0.2492, trust_T3 0.3267 | VERIFIED |
| Reviewer-package DB, **same run_id, same row/timestamp** | accuracy **85.19%**, final loss **0.499**, trust_T3 0.2394 | VERIFIED |
| Re-execution of the unchanged legacy trainer (this session) | per-epoch val 89.7, 90.8, 91.9, 91.6, 92.1%; final loss 0.2492; test 92.37%, F1 0.9231, T3 0.3267 | VERIFIED (matches main DB to every printed digit) |

**Why Table 1 is not 91.53% of anything homogeneous (VERIFIED).** 91.53% is the unweighted mean of *all* six ResNet-18 runs in the main DB: three one-epoch runs (mean 90.54%) and three five-epoch runs (92.52%). ViT-Base (95.41%) and the "Prithvi-100M" row (95.30%) were pooled in the same way. Only seeds 42, 100, 999 exist (plus 1234 for ResNet-50 at one epoch); the five-seed description was incorrect.

**Why the two databases disagree (VERIFIED mechanism).** The legacy run identifier is `sha256(model|dataset|seed|epochs|mode|git_sha)[:16]` and `git_sha` was always `unknown`, so re-running a configuration produced the *same* id. `store_run` inserted the experiment row with `INSERT OR IGNORE` (keeping the first timestamp) but wrote metrics with `INSERT OR REPLACE`. Re-executing a configuration therefore silently replaced its metrics under the original row. Seven values differ between the databases: six for ResNet-18/5 epochs/seed 42 (accuracy, F1, precision, recall, final loss, trust) and the trust value of ViT-Base/1 epoch/seed 100 (0.5 = fallback in the reviewer copy, 0.3875 in the main DB).

**Which execution does Figure 7 match? (INFERRED, strongly supported.)** The reviewer-package values (final loss 0.499, test accuracy 85.19%) agree with the convergence figure (`paper/figures/training_curve.pdf`) on both its end-of-training loss (~0.50) and its accuracy band (84-86%). A second submitted figure file, `paper/figures/baseline_results.pdf`, literally prints "Final Test Accuracy: 85.19%" and "Final Trust Score (T3): 0.2394" - the reviewer-package values of this run (VERIFIED by `legacy_audit.json::submitted_figure_files` and a test) - and it was registered in the database 89.6 s after that run's timestamp. In the submitted PDF the training-dynamics figure is Fig. 8 and Fig. 7 is a cross-dataset drift plot; the reviewer's wording "Figure 7" must be checked against the real notification. The values later used in Table 1 (92.37%, 0.249) match neither. The figure's plotted data and per-epoch logs were not preserved, so the link cannot be proven.

**Can the Figure-7-like execution be reproduced? (VERIFIED: no.)** The current legacy code is deterministic and returns exactly the main-DB values. The first execution (85.19%) must therefore have used a different code or data state (git revision unrecorded). It cannot be regenerated.

**Final action.** Neither number is "corrected". Both sources are preserved; the pooled Table 1 and the unsupported convergence figure are withdrawn; the manuscript's Section 5 (audit) states the above; and all reported numbers now come from a homogeneous 5-seed, 5-epoch cohort whose validation history is plotted next to the test accuracy with explicit labels (Figure "training history"). Status: **explained, residual ambiguity about the figure's exact source run remains and is disclosed**.

## D2. Mixed epochs and seed count
See D1. Legacy cohort composition (VERIFIED): ResNet-18 {1 epoch: 42,100,999; 5 epochs: 42,100,999}; ViT-Base the same; "Prithvi-100M" {1 epoch: 42,100,999; 5 epochs: 42,100}; ConvNeXt-T {1 epoch: 100; 5 epochs: 42,100,999}; ResNet-50 {1 epoch: 1234}.

## D3. Fallback trust values
Original `trust.py` returned `0.65` on ImportError and `0.50` when no Grad-CAM layer was found (extracted from the preservation archive). Five stored runs have `trust_T3` exactly 0.5: Prithvi-100M 1 epoch (seeds 42, 100, 999) and ViT-Base 1 epoch (seeds 42, 999). Their trust values are fallbacks, not measurements, and are invalid. No legacy trust value is used anywhere in the revision. The function now raises; tests enforce it.

## D4. "Prithvi-100M" was a proxy
For all five (epochs, seed) cells present for both models, "Prithvi-100M" accuracy, F1, precision and recall are bit-identical to ViT-Base (VERIFIED). The legacy registry mapped Prithvi/SatMAE/DOFA/ScaleMAE to ViT and SegFormer to Swin. All such results are withdrawn. Prithvi-EO-1.0-100M was evaluated for the first time in this revision.

## D5. Two databases
`reviewer_package/experiment_db.sqlite` is a stale, earlier snapshot (10 of the 22 legacy experiments, 60 of 132 metric values; seven values differ, see D1). The main database is authoritative for the *historical* record and is additionally preserved unchanged; new results live in `revision_runs` and `results/raw/`. The stale snapshot is moved (not deleted) to `reviewer_package/legacy_stale_reviewer_db/`.

## D6. Newly discovered input issue
The EuroSAT GeoTIFFs store B8A last, not in the nominal position (see `MODEL_INPUT_AUDIT.md`). Using the nominal index would have fed the water-vapour band B09 to Prithvi in place of B8A. The audit and its negative unit test prevent this.

## D7. Process incident (this session)
A `pkill` of the campaign wrapper left its child process running the superseded dry run with old code for ~30 minutes. It was terminated; it wrote no result record. One partially written run directory (`artifacts/revision/05b0db0d68bc6a56-204f6ab2`) has no record and is ignored by the analysis.

## D8. Prithvi campaign: stall, restart and resume (process record, not a result issue)
* First Prithvi launch (`artifacts/logs/campaign_ms6_attempt1_killed_stalled.log`, empty): the process waited on the GPU for ~7 h of wall-clock with 48 s of CPU time while the laptop was swapping/sleeping; it was killed and a throughput benchmark (MPS ~50 img/s) showed the model itself is not the cause. Two partial run directories from that attempt have no result record and are ignored.
* Second launch (`campaign_ms6_final_part1.log`) completed seeds 42, 999 and 100, then the controlling session ended and the process died while training a fourth seed (no record written).
* Seeds 1234 and 2026 were run afterwards with the same configuration and the identical source hash (`eb8e733d278f`, checked before resuming); logs: `campaign_ms6_resume_seed{1234,2026}.log`. The compliance checker and analysis require one completed record per declared (model, seed) with a single source revision, so a resumed cohort is accepted only if it is homogeneous.

## D9. Mock statistical/benchmark code in the legacy tree
`docs/revision/LEGACY_PLACEHOLDER_SCAN.md` (generated by `scripts/scan_legacy_placeholders.py`) lists legacy modules that label themselves mocks: `experiments/statistics/engine.py` ("Mocked Nemenyi post-hoc"), `meta/aggregator.py` ("mock Bayesian ranking"), `science/power_analysis.py`, `science/stability.py` ("Mocked ANOVA"), `xai/benchmarking/manager.py` (deletion/insertion AUC from `np.random.uniform`), publication figure/table/readiness generators and the reviewer-mode reproduction. Whether any produced submitted numbers cannot be determined from the preserved evidence. The active pipeline has zero hits (test-enforced); the corresponding claims are withdrawn.
