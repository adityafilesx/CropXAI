"""Training and evaluation with immutable evidence, deterministic subsets and explicit failures."""
from __future__ import annotations
import json, random, time, uuid, traceback
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from .data import loaders, validate_bands, stratified_indices
from .models import create_model, classifier, freeze_backbone, ARCHITECTURES
from .inputs import spec_for_bands
from .metrics import prediction_metrics, representation_quality, pillar_scores, composite
from .explainability import explain, corrupt
from .provenance import digest, source_hash, file_hash, environment, write_json, append_record

def validate_config(cfg):
    spec=validate_bands(cfg['bands'],len(cfg['bands']))
    if cfg.get('input_representation')!=spec.name: raise ValueError(f"input_representation must be '{spec.name}' for bands {cfg['bands']}")
    if spec.name!='eurosat_rgb_jpeg' and not cfg.get('tif_root'): raise ValueError('tif_root required for GeoTIFF inputs')
    unknown=[m for m in cfg['models'] if m not in ARCHITECTURES]
    if unknown: raise ValueError(f'Unregistered architectures: {unknown}')
    if cfg['size']!=224 or cfg['epochs']<1 or cfg['batch_size']<1 or cfg['learning_rate']<=0:
        raise ValueError('Invalid experiment configuration')
    if cfg['finetune_mode']!='frozen_backbone_eval': raise ValueError('Only explicit frozen-backbone protocol implemented')
    if not cfg['seeds'] or len(set(cfg['seeds']))!=len(cfg['seeds']): raise ValueError('Independent unique seeds required')
    if cfg['purpose']=='controlled_benchmark' and cfg.get('per_class'): raise ValueError('Benchmark cannot silently subsample')
    ev=json.loads(Path(cfg['evaluation_config']).read_text())
    for weights in ev['weights'].values():
        from .metrics import weights_valid
        weights_valid(weights)
    return ev

def predict(model,loader,device,corruption=None):
    probs=[]; labels=[]; features=[]
    def capture(module,args): features.append(args[0].detach().flatten(1).cpu().numpy())
    hook=classifier(model).register_forward_pre_hook(capture)
    model.eval(); gen=torch.Generator().manual_seed(445)
    try:
        with torch.no_grad():
            for x,y in loader:
                x=x.to(device)
                if corruption: x=corrupt(x,*corruption,generator=gen,spec=getattr(model,'input_spec',None))
                probs.append(model(x).softmax(1).cpu().numpy()); labels.append(y.numpy())
    finally: hook.remove()
    return np.concatenate(probs),np.concatenate(labels),np.concatenate(features)

def evaluate(model,loaders_,device,ev,run_dir,selections,cache_identity=None):
    from .feature_cache import cached_features, head_predict
    cache_metadata=[]
    if cache_identity:
        features,y,meta=cached_features(model,loaders_['test'],device,{**cache_identity,'samples':selections['test']})
        p=head_predict(model,features,device); cache_metadata.append(meta)
    else:
        p,y,features=predict(model,loaders_['test'],device)
    metrics=prediction_metrics(p,y,ev['ece_bins'])
    prediction_path=run_dir/'predictions.npz'
    np.savez_compressed(prediction_path,probabilities=p,labels=y,features=features,sample_ids=np.array(selections['test']))
    idx=stratified_indices(y,ev['representation_per_class'],ev['selection_seed'])
    rep=representation_quality(features[idx],y[idx]); metrics.update(rep)
    xai_idx=stratified_indices(y,ev['explanation_per_class'],ev['selection_seed'])
    ds=loaders_['test'].dataset
    x=torch.stack([ds[int(i)][0] for i in xai_idx]).to(device)
    # Bound activation memory for transformer gradient explanations.
    chunks=[]; map_chunks=[]
    for start in range(0,len(x),8):
        part,part_maps=explain(model,x[start:start+8],ev['perturbation_seed']+start,ev['explanation_noise_std_rgb'],ev['deletion_steps'],ev['deletion_patch_size'])
        chunks.append(part); map_chunks.append(part_maps)
    explanation=chunks[0]; maps=np.concatenate(map_chunks)
    explanation['targets']=sum([p['targets'] for p in chunks],[])
    explanation['stability_per_sample']=sum([p['stability_per_sample'] for p in chunks],[])
    explanation['stability_ssim']=float(np.mean(explanation['stability_per_sample']))
    for order in ('salient','random','least'):
        d=explanation['deletion'][order]
        d['curves']=sum([p['deletion'][order]['curves'] for p in chunks],[])
        d['auc_per_sample']=sum([p['deletion'][order]['auc_per_sample'] for p in chunks],[])
        d['auc']=float(np.mean(d['auc_per_sample']))
        curves=np.asarray(d['curves']); d['aopc']=float((curves[:,0,None]-curves[:,1:]).mean())
    explanation['deletion_auc']=explanation['deletion']['salient']['auc']
    explanation['batch_protocol']='chunks of 8; perturbation and random-deletion seed = base + chunk start'
    explanation['sample_ids']=[selections['test'][i] for i in xai_idx]
    write_json(run_dir/'explanations.json',explanation)
    np.savez_compressed(run_dir/'attributions.npz',maps=maps,indices=xai_idx)
    metrics.update(stability_ssim=explanation['stability_ssim'],deletion_auc=explanation['deletion_auc'])
    corruption_results=[]
    for kind,severities in ev['corruptions'].items():
        for severity in severities:
            if cache_identity:
                cx,cy,meta=cached_features(model,loaders_['test'],device,{**cache_identity,'samples':selections['test']},f'{kind}_{severity:g}',(kind,severity))
                cp=head_predict(model,cx,device); cache_metadata.append(meta)
            else:
                cp,cy,_=predict(model,loaders_['test'],device,(kind,severity))
            cm=prediction_metrics(cp,cy,ev['ece_bins'])
            name=f'corruption_{kind}_{severity:g}.npz'
            np.savez_compressed(run_dir/name,probabilities=cp,labels=cy,sample_ids=np.array(selections['test']))
            corruption_results.append({'kind':kind,'severity':severity,'accuracy':cm['accuracy'],'macro_f1':cm['macro_f1'],'predictions':str(run_dir/name)})
    metrics['corruptions']=corruption_results
    metrics['real_geographic_ood']=None
    metrics['pillars']=pillar_scores(metrics,[r['accuracy'] for r in corruption_results],rep['representation'])
    metrics['composite_trust']=composite(metrics['pillars'],ev['weights']['default'])
    write_json(run_dir/'evaluation_cache.json',cache_metadata)
    return metrics

def run(cfg,model_name,seed):
    ev=validate_config(cfg)
    manifest=json.loads(Path(cfg['manifest']).read_text())
    if manifest['errors']: raise ValueError('Dataset audit failed; do not train on overlapping or invalid splits')
    # Recheck exact file contents against the immutable audited split before every run.
    for row in manifest['samples']:
        path=Path(cfg.get('tif_root') or cfg['dataset_root'])/row['path']
        if file_hash(path)!=row['sha256']: raise ValueError(f'Dataset changed after audit: {path}')
    if seed not in cfg['seeds'] or model_name not in cfg['models']: raise ValueError('Run outside declared campaign')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(cfg.get('threads',4))
    torch.use_deterministic_algorithms(True)
    device=cfg['device']
    if device=='auto': device='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    revision=source_hash(); config={**cfg,'evaluation':ev,'selected_model':model_name,'selected_seed':seed}
    run_id=digest({'config':config,'source':revision,'split':manifest['split_id']})[:16]+'-'+uuid.uuid4().hex[:8]
    run_dir=Path('artifacts/revision')/run_id; run_dir.mkdir(parents=True,exist_ok=False)
    record={'schema_version':2,'run_id':run_id,'timestamp':datetime.now(timezone.utc).isoformat(),'status':'running',
            'config':config,'config_hash':digest(config),'source_revision':revision,'seed':seed,
            'model':{'architecture':model_name},'dataset':{'id':manifest['dataset'],'version':manifest['release'],'split_id':manifest['split_id'],'manifest':cfg['manifest'],'manifest_sha256':file_hash(cfg['manifest'])},
            'environment':environment(),'device':device,'artifacts':{},'metrics':{},'errors':[]}
    write_json(run_dir/'running.json',record); start=time.perf_counter()
    try:
        model,metadata=create_model(model_name,len(manifest['classes']),cfg['pretrained'],cfg['bands'])
        spec=spec_for_bands(cfg['bands'])
        # Configured bands, model metadata and registered representation must agree before any data is read.
        if metadata['bands']!=list(cfg['bands']) or metadata['channels']!=spec.channels or metadata['input_spec']!=spec.name:
            raise ValueError('Model input metadata disagrees with the configured bands')
        record['model']=metadata; record['input']=spec.describe(); model=model.to(device); freeze_backbone(model)
        use_cache=cfg.get('feature_cache',False)
        data,mapping,selections=loaders(cfg['dataset_root'],cfg['size'],cfg['batch_size'],seed,cfg.get('per_class'),training=not use_cache,bands=cfg['bands'],tif_root=cfg.get('tif_root'))
        probe=next(iter(data['test']))[0]
        if tuple(probe.shape[1:])!=(spec.channels,cfg['size'],cfg['size']): raise ValueError(f'Batch tensor {tuple(probe.shape)} disagrees with {spec.name}')
        with torch.no_grad(): model(probe[:2].to(device))  # forward-shape assertion on real data
        record['class_to_idx']=mapping; record['dataset']['selection_hash']=digest(selections)
        write_json(run_dir/'samples.json',selections)
        cache_identity=None
        if use_cache:
            from .feature_cache import cached_features, train_cached_head
            cache_identity={'model':metadata,'split_id':manifest['split_id'],'size':cfg['size'],
                            'batch_size':cfg['batch_size'],'normalization':spec.name,'input':spec.describe(),'mode':'frozen_backbone_eval'}
            cached={}; cache_paths=[]
            for split in ('train','val'):
                cx,cy,meta=cached_features(model,data[split],device,{**cache_identity,'samples':selections[split]})
                cached[split]=(cx,cy); cache_paths.append(meta)
            fx,fy,meta=cached_features(model,data['train'],device,{**cache_identity,'samples':selections['train']},'horizontal_flip')
            cache_paths.append(meta); write_json(run_dir/'training_cache.json',cache_paths)
            train_cached_head(model,cached['train'],(fx,fy),cached['val'],device,config,seed,run_dir/'history.json')
        optimizer=torch.optim.Adam(classifier(model).parameters(),lr=cfg['learning_rate'])
        history=[]
        for epoch in range(0 if use_cache else cfg['epochs']):
            model.eval(); classifier(model).train(); total=0.; n=0
            for x,y in data['train']:
                x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True)
                logits=model(x); loss=torch.nn.functional.cross_entropy(logits,y); loss.backward(); optimizer.step()
                total+=float(loss.detach())*len(y); n+=len(y)
            vp,vy,_=predict(model,data['val'],device)
            history.append({'epoch':epoch+1,'training_loss':total/n,'validation':prediction_metrics(vp,vy,ev['ece_bins'])})
            write_json(run_dir/'history.json',history)
            print(f'{run_id} epoch {epoch+1}/{cfg["epochs"]} loss={total/n:.5f} val_accuracy={history[-1]["validation"]["accuracy"]:.5f}',flush=True)
        # Fixed final epoch selection; validation is diagnostic, never test-selected.
        torch.save({'state_dict':model.state_dict(),'metadata':metadata,'config':config,'class_to_idx':mapping},run_dir/'checkpoint.pt')
        record['metrics']=evaluate(model,data,device,ev,run_dir,selections,cache_identity)
        record['status']='completed'
    except Exception as exc:
        record['status']='failed'; record['errors']=[str(exc),traceback.format_exc()]
    record['elapsed_seconds']=time.perf_counter()-start
    record['artifacts']={str(p):file_hash(p) for p in sorted(run_dir.iterdir()) if p.name!='running.json'}
    append_record(record)
    write_json(Path('results/raw')/(run_id+'.json'),record)
    print(f'{run_id}: {record["status"]}, {record["elapsed_seconds"]:.1f}s',flush=True)
    if record['status']=='failed': raise RuntimeError(record['errors'][0])
    return record
