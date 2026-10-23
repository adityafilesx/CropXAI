"""Benchmark compliance gate. Exit status 1 if any critical check fails."""
import sys, json, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cropxai_analysis.compliance import check_all
from src.cropxai.provenance import write_json

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--config',action='append',help='experiment configuration (repeatable)')
    p.add_argument('--no-hash-verify',action='store_true',help='skip re-hashing artifacts (faster; weaker)')
    p.add_argument('--light',action='store_true',help='reviewer package: checkpoints not bundled')
    p.add_argument('--output',default='docs/revision/evidence/compliance_report.json')
    a=p.parse_args(argv)
    cfgs=a.config or ['configs/experiments/revision_v1.json','configs/experiments/revision_v1_prithvi_ms6.json']
    report=check_all(cfgs,verify_hashes=not a.no_hash_verify,require_checkpoints=not a.light)
    write_json(a.output,report)
    for r in report['reports']:
        print(f"[{'PASS' if r['ok'] else 'FAIL'}] cohort={r['cohort']}")
        for c in r['checks']:
            if not c['passed']: print(f"   {'CRITICAL' if c['critical'] else 'note'} {c['check']}: {str(c['detail'])[:300]}")
    return 0 if report['ok'] else 1
if __name__=='__main__': raise SystemExit(main())
