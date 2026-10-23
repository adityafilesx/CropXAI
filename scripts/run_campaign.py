import sys, argparse, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cropxai.experiment import run, validate_config

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/experiments/revision_v1.json'); p.add_argument('--model'); p.add_argument('--seed',type=int)
    args=p.parse_args(); cfg=json.loads(Path(args.config).read_text()); validate_config(cfg)
    failed=[]
    for model in ([args.model] if args.model else cfg['models']):
        for seed in ([args.seed] if args.seed is not None else cfg['seeds']):
            try: run(cfg,model,seed)
            except Exception as exc: failed.append((model,seed,str(exc))); print('FAILED',failed[-1],flush=True)
    if failed: raise SystemExit(1)
if __name__=='__main__': main()
