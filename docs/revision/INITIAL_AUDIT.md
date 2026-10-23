# Initial repository audit - Paper 445

Audit performed 2026-10-21, before scientific implementation edits.

## Preservation and Git
The four commands `git status --short`, `git branch --show-current`, `git log -5 --oneline`, and `git ls-files` all report that this is not a Git repository. No branch was created and Git was not initialized. Original editable sources, figures, PDFs, configurations and databases are preserved in `archives/pre_revision_445/original_source_and_evidence.tar.gz`, with member SHA256 hashes and complete file inventory. Datasets and environment remain in place; no cleanup deletion is authorized by this audit.

## Inventory and environment
Repository 6.3 GB; data 5.1 GB; environment 1.2 GB. Python 3.14.8. Installed package versions captured separately. `pdflatex` and Poppler are available. Main roots: src, configs, data, results, artifacts, paper, Paper format, reviewer_package, tests, cluster_scripts. Artifacts/logs/experiments/output directories have no recoverable training checkpoints or epoch logs at initial inspection. Cached torchvision pretrained weights exist outside the repository, but are not fine-tuned experiment checkpoints.

## Actual implementation
`cropxai` is the argparse executable, dispatching to src/cli. The installed pyproject entry point instead invokes src/research/cli/main.py, whose training/evaluation commands only warn. `src/research/execution/trainer.py` trains RGB ImageFolder models with Adam, fixed ImageNet normalization and a default frozen backbone. It writes final aggregate metrics to results/experiment_db.sqlite, without saving trained weights or epoch histories. `src/models/registry.py` silently substitutes ViT for Prithvi/SatMAE/DOFA/ScaleMAE and Swin for SegFormer. This invalidates architecture-specific claims. Research adapters in another registry are not used by this path.

`src/data/preprocessing.py` copies JPEGs into class-wise random 70/15/15 splits, seed 42, but unsorted filesystem iteration prevents portable regeneration. `src/data/dataset.py` loads RGB at 224x224, horizontal flip only for training. No evidence establishes geographic separation or 13-band experiments.

`src/research/xai/trust.py` compares clean/noisy Grad-CAM SSIM on the first 100 ordered test samples; this is stability, not deletion faithfulness. Missing library/layer returns made-up constants. Grad-CAM hooks are not removed. Composite/uncertainty/representation modules elsewhere are not wired into the active experiment.

## Results and paper workflow
SQLite aggregation mixes epoch counts and counts runs as independent seeds. All legacy Git revisions are unknown. Run fingerprints omit split and checkpoint hashes; is_validated is set on insertion without scientific validation. Compliance checker always succeeds. Two databases differ and must both be retained.

`src/cli/paper.py` generates two figures, overwrites paper/main.tex and embeds unsupported additional figures and claims; no PDF compilation. main.tex claims five seeds/five epochs, six trust pillars, geographic shifts and statistical significance. Existing data do not establish these. Reviewer package omits code/executable and places DB at the wrong runtime path. Local Procedia template is from 2024; current conference compliance requires external verification.

## Tests and missing files
233 source files passed AST parsing. tests contains conftest.py only; no scientific tests existed. README.md and LICENSE referenced by packaging were absent. No release URL is verified. Many research modules are prototypes or stubs, including calibration metric base interfaces, campaign metrics, publication generators and compliance validation. Existing abstractions are not evidence of executed analyses.

## Architecture decision and cleanup candidates
Retain src paths to avoid breaking the large legacy tree. Add a clearly documented tested revision workflow, then route active entry points through it. Keep legacy files for audit; do not claim them as validated features. Candidates: bytecode, .DS_Store, root TeX build files, unused generated assets and duplicated registries. They remain preserved, with obsolete paths documented rather than removed. Preserve both databases, original paper/figures, local template, raw/processed images, all source/configuration and cached pretrained weights.

## Highest-risk issues
1. Proxy architecture labels and unknown checkpoint provenance.
2. Mixed epochs/seeds and unavailable epoch logs behind Figure 7.
3. Missing six-pillar measurements; fallback scores may contaminate legacy results.
4. RGB versus multispectral claims and non-spatial splits.
5. Unsupported charts/statistics and incomplete reviewer bundle.
6. No genuine similarity-service report or confirmed public release.
