"""Precompute deterministic frozen features; does not create experiment results."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.cropxai.models import create_model,freeze_backbone
from src.cropxai.data import loaders
from src.cropxai.feature_cache import cached_features
from src.cropxai.provenance import file_hash
if __name__=='__main__':
    cfg=json.loads(Path('configs/experiments/revision_v1.json').read_text()); manifest=json.loads(Path(cfg['manifest']).read_text())
    if manifest['errors']: raise ValueError('Dataset audit failed')
    for row in manifest['samples']:
        if file_hash(Path(cfg['dataset_root'])/row['path'])!=row['sha256']: raise ValueError('Changed dataset')
    ev=json.loads(Path(cfg['evaluation_config']).read_text()); torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    data,mapping,selections=loaders(cfg['dataset_root'],cfg['size'],cfg['batch_size'],42,{},training=False)
    for name in cfg['models']:
        print('Preparing',name,flush=True); model,metadata=create_model(name,len(mapping)); freeze_backbone(model)
        identity={'model':metadata,'split_id':manifest['split_id'],'size':cfg['size'],'batch_size':cfg['batch_size'],'normalization':'ImageNet','mode':'frozen_backbone_eval'}
        for split in ('train','val','test'):
            cached_features(model,data[split],'cpu',{**identity,'samples':selections[split]})
        cached_features(model,data['train'],'cpu',{**identity,'samples':selections['train']},'horizontal_flip')
        for kind,severities in ev['corruptions'].items():
            for severity in severities:
                cached_features(model,data['test'],'cpu',{**identity,'samples':selections['test']},f'{kind}_{severity:g}',(kind,severity))
        print('Prepared',name,flush=True)
