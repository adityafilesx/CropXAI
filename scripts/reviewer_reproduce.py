"""Reviewer entry point: re-derive every reported number from the bundled immutable run records.

Needs no GPU, no dataset and no model checkpoint. It (1) recomputes each run's metrics from its saved predictions,
explanations and corruption outputs, (2) checks the experimental matrix, (3) regenerates figures/tables, and
(4) compares the regenerated Table 1 with the one shipped in `reference/`. Exit status 1 on any discrepancy.
"""
import os, sys, json, shutil
from pathlib import Path
os.environ['CROPXAI_LIGHT']='1'
ROOT=Path(__file__).resolve().parents[1]; os.chdir(ROOT); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts'))

def main():
    import analyze_results, generate_figures, check_compliance
    status=[]
    out=analyze_results.analyze(); status.append(('records revalidated from artifacts',len(out['records']) and not out['rejected'],f"{len(out['records'])} valid, {len(out['rejected'])} rejected"))
    rc=check_compliance.main(['--light','--output','compliance_report.json']); status.append(('protocol compliance (all cohorts)',rc==0,'see compliance_report.json'))
    generate_figures.generate()
    new=json.loads(Path('results/tables/table1.json').read_text()); ref=json.loads(Path('reference/table1.json').read_text())
    same=len(new)==len(ref) and all(a['model']==b['model'] and a['metric']==b['metric'] and abs(a['mean']-b['mean'])<1e-9 for a,b in zip(new,ref))
    status.append(('regenerated Table 1 equals shipped Table 1',same,f'{len(new)} rows'))
    for name,ok,detail in status: print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return 0 if all(ok for _,ok,_ in status) else 1
if __name__=='__main__': raise SystemExit(main())
