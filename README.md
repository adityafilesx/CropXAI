# CropXAI - revision workflow (Paper ID 445)

Evaluation of Earth-observation image classifiers on **EuroSAT land-cover classification**, reporting predictive performance,
calibration, explanation stability, deletion-based faithfulness, synthetic-corruption robustness, probabilistic forecast quality and
feature separation, summarised by a transparent **Composite Trust Score** (a weighted sum - not a Bayesian posterior).

Scope: land-cover classification only. Nothing here validates crop-disease detection, crop-health monitoring or yield estimation.
No public release URL has been verified; no open-source release claim is made.

## What is evaluated

| Cohort | Models | Input | Config |
|---|---|---|---|
| `rgb_imagenet` | ResNet-18, ViT-Base/16, ConvNeXt-T (torchvision ImageNet-1K V1) | EuroSAT RGB JPEG, 3 channels, ImageNet statistics | `configs/experiments/revision_v1.json` |
| `ms6_prithvi` | Prithvi-EO-1.0-100M (revision `f3a9ea7a`, SHA-256 `7fac0c8a...`) | EuroSAT Sentinel-2 GeoTIFF, bands B02,B03,B04,B8A,B11,B12 (= Prithvi HLS B02..B07), single time step | `configs/experiments/revision_v1_prithvi_ms6.json` |

Both cohorts: frozen backbone, seeded linear head, Adam lr 1e-3, batch 32, **5 epochs**, seeds 42/100/999/1234/2026, the same 18,900/4,050/4,050 split and the same
test items. The cohorts differ in input information and pretraining, so cross-cohort comparisons are confounded and labelled as such.
Details: `docs/revision/MODEL_INPUT_AUDIT.md`, `docs/methodology.md`, `docs/model_configurations.md`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# torchvision weights must already be in $(python -c "import torch;print(torch.hub.get_dir())")/checkpoints:
#   resnet18-f37072fd.pth  vit_b_16-c867db91.pth  convnext_tiny-983f1562.pth      (no automatic download)
# Prithvi checkpoint (pinned revision; verified by SHA-256 at load time):
python -c "from huggingface_hub import hf_hub_download as d; d('ibm-nasa-geospatial/Prithvi-EO-1.0-100M','Prithvi_EO_V1_100M.pt',revision='f3a9ea7a1723621b0aeceb4de993093704d712c5',local_dir='artifacts/pretrained/prithvi_eo_1.0_100m')"
```

Data (not bundled): EuroSAT RGB under `data/processed/eurosat/{train,val,test}/<class>/*.jpg` and the 13-band GeoTIFFs under
`data/raw/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/<class>/*.tif`.

## Commands (all through one entry point, `./cropxai`)

```bash
./cropxai environment                 # versions and device
./cropxai validate-dataset            # hash both manifests, leakage and JPEG<->GeoTIFF pairing, band-order evidence
./cropxai smoke                       # small real-data run (not a result)
./cropxai campaign run --config configs/experiments/revision_v1.json              # 3 models x 5 seeds
./cropxai campaign run --config configs/experiments/revision_v1_prithvi_ms6.json  # Prithvi x 5 seeds
./cropxai analyze                     # revalidate every record, aggregate, paired comparisons, weight sensitivity
./cropxai compliance                  # protocol/matrix checker; exit 1 on any violation
./cropxai figures                     # figures + LaTeX tables from validated records
./cropxai audit-legacy                # reconstruct the submitted Table 1 from the legacy database
./cropxai paper build                 # manuscript + reviewer response PDF
./cropxai validate-manuscript         # paper/result consistency gate; exit 1 on any critical failure
```

Tests: `python -m unittest discover -s tests -p "test_revision*.py"` (CPU; the real-data integration test needs the processed EuroSAT folder).

## Provenance

Every run is an immutable JSON record (`results/raw/<run_id>.json`) mirrored in the append-only table `revision_runs` of `results/experiment_db.sqlite`;
the legacy `experiments`/`metrics` tables from the submission are untouched. A record stores the configuration hash, hash of the result-determining code
(`src/cropxai/**`), dataset manifest hash, model/checkpoint hashes, environment packages, and the SHA-256 of every artifact under `artifacts/revision/<run_id>/`.
Aggregation recomputes every metric from the saved predictions before using it and groups runs only by identical protocol, split and source revision.

## Lightweight verification (no GPU, data or checkpoints)

`reviewer_package/` (built by `./cropxai paper package`) re-derives every reported number from the bundled records:

```bash
cd reviewer_package && docker build -t cropxai-reviewer . && docker run --rm cropxai-reviewer
# or without Docker:  pip install -r requirements.txt && python scripts/reviewer_reproduce.py
```

## Documentation of the revision

`docs/revision/`: `CONTINUATION_AUDIT.md`, `MODEL_INPUT_AUDIT.md`, `EXPERIMENT_DISCREPANCIES.md`, `REVIEWER_TRACEABILITY.md`, `REVISION_CHANGELOG.md`, `SUBMISSION_CHECKLIST.md`, `LITERATURE_COMPARISON.md`.
Original submission artifacts: `archives/pre_revision_445/` (hash-listed).
