import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cropxai.data import audit_dataset, audit_ms_dataset
TIF_ROOT='data/raw/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
if __name__=='__main__':
    report=audit_dataset(output='results/processed/dataset_manifest.json')
    print({k:v for k,v in report.items() if k not in ('samples','file_duplicates','pixel_duplicates')})
    ms=audit_ms_dataset('results/processed/dataset_manifest.json',TIF_ROOT,output='results/processed/dataset_manifest_ms.json')
    print({k:v for k,v in ms.items() if k not in ('samples',)})
    raise SystemExit(bool(report['errors'] or ms['errors']))
