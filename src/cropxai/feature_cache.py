"""Exact eval-mode frozen features; content hashes reject stale cache entries."""
from pathlib import Path
import json
import numpy as np
import torch
from .models import classifier
from .explainability import corrupt
from .provenance import digest, file_hash, write_json

def cached_features(model, loader, device, identity, variant='clean', corruption=None):
    spec={'version':1,'identity':identity,'variant':variant,'corruption':corruption,
          'torch':torch.__version__,'extractor_sha256':file_hash(__file__),
          'corruption_sha256':file_hash(Path(__file__).with_name('explainability.py'))}
    key=digest(spec); root=Path('artifacts/feature_cache'); root.mkdir(parents=True,exist_ok=True)
    path=root/(key+'.npz'); meta=root/(key+'.json')
    if path.exists() and meta.exists():
        info=json.loads(meta.read_text())
        if info['spec']!=json.loads(json.dumps(spec)) or info['sha256']!=file_hash(path): raise ValueError('Feature cache corrupted or stale')
        with np.load(path,allow_pickle=False) as z: return z['features'],z['labels'],str(meta)
    model.eval(); features=[]; labels=[]
    hook=classifier(model).register_forward_pre_hook(lambda module,args:features.append(args[0].detach().flatten(1).cpu().numpy()))
    gen=torch.Generator().manual_seed(445)
    try:
        with torch.no_grad():
            for x,y in loader:
                x=x.to(device)
                if variant=='horizontal_flip': x=x.flip(-1)
                if corruption: x=corrupt(x,*corruption,generator=gen,spec=getattr(model,'input_spec',None))
                model(x); labels.append(y.numpy())
    finally: hook.remove()
    x,y=np.concatenate(features),np.concatenate(labels)
    np.savez_compressed(path,features=x,labels=y)
    write_json(meta,{'spec':spec,'sha256':file_hash(path),'shape':list(x.shape)})
    print('Feature cache complete:',variant,len(y),'samples',flush=True)
    return x,y,str(meta)

def head_predict(model,features,device,batch_size=1024):
    probs=[]; model.eval()
    with torch.no_grad():
        for start in range(0,len(features),batch_size):
            x=torch.from_numpy(features[start:start+batch_size]).to(device)
            probs.append(classifier(model)(x).softmax(1).cpu().numpy())
    return np.concatenate(probs)

def train_cached_head(model,train,flipped,val,device,config,seed,history_path):
    from .metrics import prediction_metrics
    x,y=train; xf,yf=flipped; vx,vy=val
    if not np.array_equal(y,yf): raise ValueError('Flipped cache sample ordering differs')
    head=classifier(model); optimizer=torch.optim.Adam(head.parameters(),lr=config['learning_rate'])
    gen=torch.Generator().manual_seed(seed); history=[]
    for epoch in range(config['epochs']):
        model.eval(); head.train(); order=torch.randperm(len(y),generator=gen).numpy(); total=0.
        flips=torch.rand(len(y),generator=gen).numpy()<.5
        for start in range(0,len(y),config['batch_size']):
            ids=order[start:start+config['batch_size']]
            batch=np.where(flips[ids,None],xf[ids],x[ids])
            bx=torch.from_numpy(batch).to(device); by=torch.from_numpy(y[ids]).to(device)
            optimizer.zero_grad(set_to_none=True); loss=torch.nn.functional.cross_entropy(head(bx),by)
            loss.backward(); optimizer.step(); total+=float(loss.detach())*len(ids)
        pm=prediction_metrics(head_predict(model,vx,device),vy,config['evaluation']['ece_bins'])
        history.append({'epoch':epoch+1,'training_loss':total/len(y),'validation':pm})
        write_json(history_path,history)
        print(f'cached epoch {epoch+1}/{config["epochs"]} loss={total/len(y):.5f} val_accuracy={pm["accuracy"]:.5f}',flush=True)
    return history
