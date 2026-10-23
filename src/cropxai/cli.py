"""Unified CLI for the validated revision workflow."""
import argparse, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(prog='cropxai'); sub=p.add_subparsers(dest='command',required=True)
    for name in ('environment','validate-dataset','analyze','figures','validate-manuscript','smoke','compliance','audit-legacy'): sub.add_parser(name)
    camp=sub.add_parser('campaign'); camp.add_argument('action',choices=['run']); camp.add_argument('--config',default='configs/experiments/revision_v1.json'); camp.add_argument('--model'); camp.add_argument('--seed',type=int)
    paper=sub.add_parser('paper'); paper.add_argument('action',choices=['build','package'])
    args=p.parse_args(); mapping={'environment':'validate_environment.py','validate-dataset':'validate_dataset.py','analyze':'analyze_results.py','figures':'generate_figures.py','validate-manuscript':'validate_manuscript.py','compliance':'check_compliance.py','audit-legacy':'audit_legacy.py'}
    extra=[]
    if args.command in mapping: script=mapping[args.command]
    elif args.command=='smoke': script='run_campaign.py'; extra=['--config','configs/experiments/smoke_v1.json']
    elif args.command=='campaign':
        script='run_campaign.py'; extra=['--config',args.config]
        if args.model: extra+=['--model',args.model]
        if args.seed is not None: extra+=['--seed',str(args.seed)]
    else: script='build_manuscript.py' if args.action=='build' else 'package_revision.py'
    raise SystemExit(subprocess.call([sys.executable,str(ROOT/'scripts'/script),*extra],cwd=ROOT))
if __name__=='__main__': main()
