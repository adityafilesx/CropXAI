"""Build the self-contained reviewer package (code + immutable run records + light artifacts; no checkpoints/data)."""
import sys, shutil, json, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]; PKG=ROOT/'reviewer_package'
CODE_FILES=['cropxai','scripts/analyze_results.py','scripts/generate_figures.py','scripts/check_compliance.py','scripts/audit_legacy.py','scripts/reviewer_reproduce.py']
SKIP_PARTS={'__pycache__'}

README='''# CropXAI reviewer package (Paper 445)

Re-derives every reported number from the bundled immutable run records. No GPU, no EuroSAT data and no model checkpoint are needed.

```bash
docker build -t cropxai-reviewer . && docker run --rm cropxai-reviewer
# without Docker (Python >= 3.11):
pip install -r requirements.txt && python scripts/reviewer_reproduce.py
```

Expected output: three `[PASS]` lines - (1) every run record is revalidated from its saved predictions, explanations and corruption outputs; (2) the protocol-compliance checker accepts both cohorts; (3) the regenerated Table 1 equals `reference/table1.json`. Exit status 1 on any discrepancy.

Contents: `src/cropxai` (the exact code whose hash is stored in every record), `src/cropxai_analysis`, `scripts`, `configs`, `results/raw` (immutable run records) and `results/experiment_db.sqlite` (authoritative store; legacy tables untouched), `artifacts/revision/<run_id>` (predictions, explanations, corruption outputs, histories - without checkpoints), `legacy_stale_reviewer_db/` (the earlier, divergent reviewer snapshot, preserved), `MANIFEST.json` (SHA-256 of every file).

Not bundled (hashes are recorded in the run records): model checkpoints (`checkpoint.pt`, ~4 GB), the EuroSAT data, torchvision/Prithvi weights. Full re-training needs the repository root README: pinned Prithvi revision f3a9ea7a (SHA-256 7fac0c8a8693198e32a055e0c5a967f8b005f382182b63df1b29fdcd5c880731), torchvision IMAGENET1K_V1 weights, and EuroSAT RGB + 13-band GeoTIFF files.
'''

def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)

def main():
    legacy=PKG/'legacy_stale_reviewer_db'; legacy.mkdir(exist_ok=True)
    old=PKG/'experiment_db.sqlite'
    if old.exists() and not (legacy/'experiment_db.sqlite').exists(): shutil.move(str(old),str(legacy/'experiment_db.sqlite'))   # preserved, not deleted
    for d in ('src','scripts','configs','results','artifacts','reference','docs'):
        if (PKG/d).exists(): shutil.rmtree(PKG/d)
    for f in CODE_FILES: copy(ROOT/f,PKG/f)
    for pattern in ('src/cropxai','src/cropxai_analysis'):
        for p in (ROOT/pattern).rglob('*'):
            if p.is_file() and not SKIP_PARTS&set(p.parts): copy(p,PKG/p.relative_to(ROOT))
    for p in list((ROOT/'src/research').rglob('*')) if False else []: pass
    for pat in ('configs/experiments/*.json','configs/evaluation/*.json'):
        for p in ROOT.glob(pat): copy(p,PKG/p.relative_to(ROOT))
    copy(ROOT/'results/experiment_db.sqlite',PKG/'results/experiment_db.sqlite')   # authoritative store (legacy + revision tables)
    for p in (ROOT/'results/raw').glob('*.json'): copy(p,PKG/p.relative_to(ROOT))
    for p in (ROOT/'results/processed').glob('*.json'): copy(p,PKG/p.relative_to(ROOT))
    copy(ROOT/'results/tables/table1.json',PKG/'reference/table1.json')
    for p in (ROOT/'artifacts/revision').rglob('*'):
        if p.is_file() and p.name not in ('checkpoint.pt','running.json') and (ROOT/'results/raw'/(p.parent.name+'.json')).exists(): copy(p,PKG/p.relative_to(ROOT))
    copy(ROOT/'docs/revision/evidence/legacy_audit.json',PKG/'docs/revision/evidence/legacy_audit.json')
    (PKG/'requirements.txt').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\ntorch\ntorchvision\nnumpy\nscipy\nscikit-learn\nscikit-image\nmatplotlib\npillow\n')
    (PKG/'Dockerfile').write_text('FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD ["python", "scripts/reviewer_reproduce.py"]\n')
    (PKG/'README.md').write_text(README)
    manifest={str(p.relative_to(PKG)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(PKG.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and not SKIP_PARTS&set(p.parts)}
    (PKG/'MANIFEST.json').write_text(json.dumps(manifest,indent=1,sort_keys=True)+'\n')
    print('package files:',len(manifest))
if __name__=='__main__': main()
