# Response to reviewers - Paper 445

Comment wording follows the author-provided revision brief; the original notification is unavailable.


## R1.1 - Cite contents properly throughout the manuscript.

**Status: VERIFIED**

We rebuilt the bibliography from inspected primary records (journal/DOI records checked against Crossref, arXiv and the Hugging Face model card), cite the journal version of EuroSAT, add sources for the new methods (MAE pooling, HLS, AOPC), distinguish preprints from journal versions, and removed unsupported novelty and release claims. Every in-text key resolves and every bibliography entry is cited (checked by the manuscript gate). The legacy bibliography is preserved unchanged.

Change: paper/source/references.bib; docs/revision/evidence/verified_sources.json; scripts/validate_manuscript.py. Analysis: Primary-source bibliography and claim-scope audit.

Location: Section 2, manuscript page 2. Verification: citation resolution and verified-source registry. Evidence: verified_sources.json (19 entries); manuscript_validation.json (citations_resolve, no_uncited_bibliography, verified_bibliography). Language proofreading by the authors is still required.


## R1.2 - Cite dataset sources.

**Status: VERIFIED**

We cite the EuroSAT journal paper and document the two local representations: 27,000 RGB JPEGs and the matching 13-band Sentinel-2 GeoTIFFs, their counts (18,900/4,050/4,050), classes, preprocessing, the random class-wise split, and the checks performed (file/pixel hashes, JPEG-GeoTIFF pairing). The upstream release version and geographic independence are unverified and stated as limitations. The empirical GeoTIFF band order (B8A stored last) is documented.

Change: src/cropxai/data.py; results/processed/dataset_manifest.json. Analysis: Audit all 27,000 images, channels, class counts and duplicate hashes.

Location: Section 3.1, manuscript page 2. Verification: DataTests; RealDataIntegration. Evidence: results/processed/dataset_manifest.json; dataset_manifest_ms.json; docs/revision/MODEL_INPUT_AUDIT.md.


## R1.3 - Explain every performance metric thoroughly.

**Status: VERIFIED**

We supplied formulas, inputs, ranges, directions, normalizations, edge cases and limitations. Macro-F1 is distinct from historical weighted F1. Calibration, uncertainty, stability, deletion faithfulness and representation are not treated as interchangeable.

Change: src/cropxai/metrics.py; metric_registry.json; explainability.py. Analysis: Deterministic metric and edge-case tests; real-data evaluation.

Location: Section 3.3, manuscript page 4. Verification: MetricsTests; ExplanationTests. Evidence: src/cropxai/metric_registry.json; tests/test_revision.py MetricsTests/ExplanationTests; tests/test_revision_v2.py.


## R1.4 - Improve limitations.

**Status: VERIFIED**

We rewrote the limitations: land-cover labels versus crop health, patch-level decisions, random non-geographic split, no real distribution shift, four architectures, five seeds that randomize only a linear head, frozen-backbone linear probing, different inputs/pretraining between the RGB and six-band cohorts, the apparent top-of-atmosphere versus surface-reflectance domain gap, masking artefacts in deletion, correlated components and subjective weights. Crop-health, disease and yield uses are future work.

Change: paper/source/manuscript_template.tex; docs/methodology.md. Analysis: Match limitations to verified input and training audit.

Location: Section 6, manuscript page 13. Verification: scope and discrepancy checks in manuscript validator. Evidence: docs/model_configurations.md; docs/methodology.md.


## R1.5 - Add graphical findings.

**Status: VERIFIED**

All figures are regenerated from validated run records: accuracy with seed and bootstrap intervals, six components with intervals, reliability diagrams, deletion curves with random/least-salient controls, corruption curves, validation-versus-test training history, weight sensitivity, pairwise differences and confusion matrices. Each figure has a JSON sidecar with its plotted values and source hash; the legacy figures are not reused.

Change: scripts/generate_figures.py; results/figures. Analysis: Generate figures from validated prediction and explanation artifacts.

Location: Section 4, manuscript page 5. Verification: figure hashes, source hashes and table consistency. Evidence: results/figures/manifest.json and sidecars; GeneratedAssetConsistencyTests; manuscript gate figure_* checks.


## R1.6 - Compare with existing published approaches.

**Status: VERIFIED**

We added a literature comparison covering EuroSAT, foundation models, XAI evaluation, calibration, corruption robustness and TorchGeo. Published scores are not pooled with local experiments. We explain protocol differences and limit the contribution to auditable integration.

Change: docs/revision/LITERATURE_COMPARISON.md; paper/source/references.bib. Analysis: Structured targeted primary-source comparison.

Location: Section 2, manuscript page 2. Verification: verified source registry and citation checks. Evidence: docs/revision/LITERATURE_COMPARISON.md; verified_sources.json. Literature scores are not pooled with local runs; no external baseline was re-executed.


## R2.1 - Explain the proposed implementation.

**Status: VERIFIED**

We documented and diagrammed the actual data, model, training, explanation, score and reporting path. The unified CLI uses tested components. Legacy prototypes are not represented as executed research; JSON and additive SQLite records preserve configurations, hashes, artifacts and failures.

Change: src/cropxai; scripts; docs/methodology.md. Analysis: End-to-end real-data smoke run with saved artifacts.

Location: Section 3, manuscript page 2. Verification: RealDataIntegration; ProvenanceTests; validate_record. Evidence: docs/methodology.md; docs/revision/MODEL_INPUT_AUDIT.md; Section 3; tests; compliance report.


## R2.2 - Compare the proposed work with published work.

**Status: VERIFIED**

We added the comparison and fairness paragraph, distinguish methodological literature from controlled same-protocol runs, and make no unexecuted baseline or superiority claim.

Change: docs/revision/LITERATURE_COMPARISON.md. Analysis: Same targeted review as R1.6, with explicit fairness statement.

Location: Section 2, manuscript page 2. Verification: primary source checks. Evidence: docs/revision/LITERATURE_COMPARISON.md.


## R2.3 - Resolve Figure 7/Table 1 inconsistency: ResNet-18 test accuracy 91.53% versus validation accuracy around 84-86%.

**Status: COMPLETED BUT NOT INDEPENDENTLY VERIFIED**

We traced both numbers. Table 1 (91.53%) is the unweighted mean of all six stored ResNet-18 runs, three of one epoch and three of five epochs. Figure 7 (validation ~84-86%) matches a different execution of the same configuration: the reviewer-package copy of the database holds, under the identical run identifier, test accuracy 85.19% and final loss 0.499, whereas the main database holds 92.37% and 0.249 for that identifier, because the legacy store replaced metrics when a configuration was re-run. Re-executing the legacy trainer reproduced the main-database values exactly and not the figure, so the figure-like run cannot be regenerated. Both sources are preserved; the pooled table and the figure are withdrawn, and all results are replaced by a homogeneous five-seed, five-epoch cohort with validation and test accuracy plotted and labelled separately. The figure's own plotted data were not preserved, so its link to the earlier execution is strongly supported but not proven.

Change: scripts/audit_legacy.py; scripts/rerun_legacy_trainer.py; docs/revision/EXPERIMENT_DISCREPANCIES.md. Analysis: Reconstruct Table I; compare both databases; re-execute the legacy trainer.

Location: Section 5, manuscript page 12. Verification: LegacyReconstructionTests (pooled mean, divergence, proxy identity). Evidence: docs/revision/evidence/legacy_audit.json; legacy_rerun_resnet18_seed42_ep5.log/json.


## R2.4 - Add standard deviations and 95% confidence intervals.

**Status: VERIFIED**

Every reported component, per model, has n=5 independent seeds, mean, sample SD, 95% Student-t interval and missing-run count (Tables 1 and 5). Test-patch bootstrap intervals and paired comparisons are reported separately; the assumptions and the narrowness of seed intervals for a frozen backbone are stated. Intervals reflect no geographic variance.

Change: src/cropxai/metrics.py; scripts/analyze_results.py. Analysis: Homogeneous independent-seed groups; Student-t mean intervals.

Location: Section 4.8, manuscript page 12. Verification: MetricsTests.test_ci; duplicate-seed rejection. Evidence: results/tables/table1.json, table_ci.tex; results/processed/revision_summary.json; paired_comparisons.json.


## R3.1 - Distinguish explanation stability from faithfulness.

**Status: VERIFIED**

We renamed SSIM as explanation stability and implemented a separate deletion metric with a fixed target, explicit mean-RGB baseline, patch ordering and trapezoidal integration. Random and least-salient controls are retained. The new common attribution rule differs from legacy Grad-CAM and is explicitly documented.

Change: src/cropxai/explainability.py; metrics.py. Analysis: Fixed-target noise stability and salient/random/least deletion curves.

Location: Section 3.4, manuscript page 4. Verification: ExplanationTests.test_deletion_known_signal; MetricsTests.test_stability. Evidence: explainability.py; tests ExplanationTests; results: stability SSIM and deletion AUC/AOPC with controls (Table 2, Figure 5).


## R3.2 - Clarify the composite score and Bayesian terminology.

**Status: VERIFIED**

The score is a weighted sum T = sum w_j s_j with fixed, versioned weights (default 1/6 each) and fixed normalizations; it is renamed Composite Trust Score and explicitly not Bayesian or a probability. Sensitivity covers equal, reliability-priority and explanation-priority weights and 50 seeded perturbations, with scores, ranks, preferred models and Spearman correlations (Section 4.4).

Change: src/cropxai/metrics.py; configs/evaluation/revision_v1.json. Analysis: Equal/default, alternative profiles and 50 deterministic perturbations.

Location: Section 3.6, manuscript page 5. Verification: MetricsTests.test_score; MetricsTests.test_sensitivity. Evidence: configs/evaluation/revision_v1.json; results/processed/weight_sensitivity.json; MetricsTests.


## R3.3 - Include all six pillars.

**Status: VERIFIED**

All six components (macro-F1, 1-deletion AUC, 1-ECE, corrupted accuracy, 1-Brier/2, normalized silhouette) are computed for every model and seed and reported with intervals in Table 1 and Table 5; the score rejects any missing or invalid component. Real geographic/seasonal OOD is not computed; the robustness component is a synthetic-corruption proxy and is labelled as such.

Change: src/cropxai/metric_registry.json; experiment.py. Analysis: Compute all six from saved predictions, deletion curves, corruptions and embeddings.

Location: Section 3.5, manuscript page 5. Verification: MetricsTests.test_pillar_normalization; validate_record. Evidence: results/tables/table1.json; manuscript gate six_pillars_* checks; compliance metrics_valid.


## R3.4 - Explain Prithvi-100M's spectral-band adaptation.

**Status: VERIFIED**

The submitted "Prithvi-100M" was an RGB ViT-Base proxy (bit-identical metrics) and is withdrawn. The real Prithvi-EO-1.0-100M checkpoint (pinned revision, SHA-256 verified) is now evaluated. Adaptation: its six pretrained channels (HLS blue, green, red, narrow NIR, SWIR1, SWIR2) receive Sentinel-2 B02, B03, B04, B8A, B11, B12 from the GeoTIFFs (storage indices 1,2,3,12,10,11, verified empirically because the nominal order is wrong); no input parameter is added or initialized; one time step is used with a one-frame position embedding verified equal to the checkpoint's first frame; Prithvi's own normalization is used. Uncorrected domain differences (apparent top-of-atmosphere vs surface reflectance, 64->224 upsampling) are disclosed.

Change: src/cropxai/models.py; inputs.py; data.py; vendor/prithvi_mae.py. Analysis: Five-seed Prithvi cohort; band-order audit.

Location: Section 3.2, manuscript page 3. Verification: PrithviModelTests; SyntheticGeoTiffTests; BandOrderEvidenceTests. Evidence: docs/revision/MODEL_INPUT_AUDIT.md; results/raw (Prithvi runs).


## R3.5 - Verify whether all models use the same spectral bands.

**Status: VERIFIED**

We generated a model-input audit from the run records. The models do not use the same bands: ResNet-18, ViT-Base and ConvNeXt-T receive RGB; Prithvi receives six Sentinel-2 bands. All share the split and test patches. This is disclosed, cross-cohort comparisons are labelled confounded, and the weight analysis that includes all four models is marked descriptive. A controlled common-input comparison is not possible without an unvalidated input adapter (BLOCKED, stated as a limitation). Configuration, model metadata, tensor shape and registered representation are asserted on every run.

Change: src/cropxai/data.py; models.py; docs/model_configurations.md. Analysis: Input metadata and shape assertions for supported RGB architectures.

Location: Section 3.2, manuscript page 3. Verification: DataTests.test_bands; ModelTests; RealDataIntegration. Evidence: docs/revision/MODEL_INPUT_AUDIT.md; compliance input_representation checks.


## R3.6 - Correct the crop-health framing.

**Status: VERIFIED**

Title kept. The abstract, introduction, methods, discussion, limitations and conclusion now describe EuroSAT land-cover classification, state that nothing here validates crop-health, disease or yield claims, and describe those as future work needing dedicated labels. The suggested "multispectral" wording is used only for the six-band Prithvi cohort; no 13-band experiment is claimed.

Change: paper/source/manuscript_template.tex. Analysis: Claim audit against dataset labels and actual channels.

Location: Section 1, manuscript page 1. Verification: scope check in scripts/validate_manuscript.py. Evidence: paper/source/445_revised_version.tex; docs/methodology.md.
