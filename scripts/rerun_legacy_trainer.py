"""Re-execute the legacy trainer once to see what curve its own code produces (R2.3 investigation).

The legacy `store_run` is replaced so the preserved legacy SQLite store is not modified; per-epoch validation
accuracy and training loss (which the legacy code only logged) are captured to docs/revision/evidence.
"""
import sys, json, logging, re, argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import src.research.execution.trainer as legacy

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',default='ResNet-18'); p.add_argument('--seed',type=int,default=42); p.add_argument('--epochs',type=int,default=5)
    a=p.parse_args(); out=Path('docs/revision/evidence'); out.mkdir(parents=True,exist_ok=True)
    tag=f"legacy_rerun_{a.model.replace('-','').lower()}_seed{a.seed}_ep{a.epochs}"; log=out/f'{tag}.log'
    logging.basicConfig(level=logging.INFO,format='%(message)s',handlers=[logging.FileHandler(log,mode='w'),logging.StreamHandler()])
    captured={}
    def fake_store(**kw): captured.update(kw); return 'not-stored'
    legacy.store_run=fake_store
    legacy.run_real_training('eurosat',a.model,epochs=a.epochs,seed=a.seed,finetune_mode='frozen')
    text=log.read_text()
    epochs=[{'epoch':int(m[0]),'training_loss':float(m[1]),'validation_accuracy_percent':float(m[2])} for m in re.findall(r'Epoch (\d+)/\d+ — Loss: ([\d.]+) — Val Acc: ([\d.]+)%',text)]
    (out/f'{tag}.json').write_text(json.dumps({'model':a.model,'seed':a.seed,'epochs':a.epochs,'per_epoch':epochs,'final_metrics_passed_to_store_run':captured.get('metrics'),
        'note':'Legacy code unchanged except store_run redirected; BatchNorm running statistics are updated in train mode (legacy behaviour).'},indent=2,sort_keys=True))
if __name__=='__main__': main()
