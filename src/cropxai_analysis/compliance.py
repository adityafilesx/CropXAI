"""Benchmark compliance: does a set of run records actually satisfy the declared experimental matrix?

Every check can fail. Nothing here reads a "passed" flag from a record; each assertion is recomputed from the
records, the declared configuration and the artifacts on disk. `ok` is False when any critical check fails.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
from src.cropxai.metrics import PILLARS, composite
from src.cropxai.models import ARCHITECTURES, PRITHVI, PRITHVI_SHA256
from src.cropxai.inputs import SPECS
from src.cropxai.provenance import digest, file_hash

PROTOCOL_KEYS=('epochs','batch_size','learning_rate','finetune_mode','size','pretrained','feature_cache','bands','input_representation','evaluation_config')

def _check(checks, name, ok, detail, critical=True):
    checks.append({'check':name,'passed':bool(ok),'detail':detail,'critical':critical})

def _finite(x): return isinstance(x,(int,float)) and math.isfinite(x)

def check_cohort(config, records, *, current_source=None, verify_artifacts=True, verify_hashes=True, artifact_root=Path('.'), require_checkpoints=True):
    """config: declared experiment configuration. records: raw run records (any status) believed to belong to it."""
    checks=[]; declared=[(m,s) for m in config['models'] for s in config['seeds']]
    mine=[r for r in records if r['config'].get('cohort')==config.get('cohort') and r['config'].get('purpose')==config['purpose']]
    failed=[r for r in mine if r['status']!='completed']; done=[r for r in mine if r['status']=='completed']
    if current_source is not None:
        stale=[r['run_id'] for r in done if r['source_revision']!=current_source]
        _check(checks,'source_current',not stale,f'{len(stale)} completed record(s) come from superseded code: {stale}')
        done=[r for r in done if r['source_revision']==current_source]
    _check(checks,'failed_runs_disclosed',True,[{'run_id':r['run_id'],'model':r['model'].get('architecture'),'seed':r['seed']} for r in failed],critical=False)
    # 1. matrix: exactly one completed record per declared (model, seed)
    count={}
    for r in done: count.setdefault((r['model']['architecture'],r['seed']),[]).append(r['run_id'])
    missing=[c for c in declared if c not in count]; dupes={f'{k[0]}/{k[1]}':v for k,v in count.items() if len(v)>1}; extra=[k for k in count if k not in declared]
    _check(checks,'matrix_complete',not missing and not dupes and not extra,{'missing':missing,'duplicates':dupes,'undeclared':extra})
    seeds_per_model={m:sorted({r['seed'] for r in done if r['model']['architecture']==m}) for m in config['models']}
    _check(checks,'seed_count_matches',all(len(v)==len(config['seeds']) for v in seeds_per_model.values()),seeds_per_model)
    # 2. model identity
    bad=[]
    for r in done:
        md=r['model']
        if md.get('architecture') not in ARCHITECTURES: bad.append((r['run_id'],'unregistered architecture'))
        elif r['config']['pretrained'] and (not md.get('weights') or not md.get('checkpoint_sha256')): bad.append((r['run_id'],'pretrained run lacks checkpoint identity'))
        elif md.get('architecture')==PRITHVI and md.get('checkpoint_sha256')!=PRITHVI_SHA256: bad.append((r['run_id'],'Prithvi checkpoint hash is not the pinned revision'))
    _check(checks,'model_identity',not bad,bad)
    ids={}
    for r in done: ids.setdefault(r['model']['architecture'],set()).add(r['model'].get('checkpoint_sha256'))
    _check(checks,'one_checkpoint_per_architecture',all(len(v)==1 for v in ids.values()),{k:sorted(map(str,v)) for k,v in ids.items()})
    # 3. input representation
    bad=[]
    for r in done:
        md=r['model']; spec=SPECS.get(r['config'].get('input_representation'))
        if spec is None: bad.append((r['run_id'],'unknown input representation')); continue
        if md['bands']!=list(spec.bands) or md['channels']!=spec.channels or md.get('input_spec')!=spec.name or list(r['config']['bands'])!=list(spec.bands):
            bad.append((r['run_id'],'bands/channels/spec disagree'))
        elif r.get('input',{}).get('channels')!=spec.channels: bad.append((r['run_id'],'recorded input spec differs'))
    _check(checks,'input_representation',not bad,bad)
    # 4. training protocol homogeneity and agreement with declaration
    bad=[]
    for k in PROTOCOL_KEYS:
        values={json.dumps(r['config'].get(k),sort_keys=True) for r in done}
        if len(values)>1: bad.append((k,'mixed values: '+', '.join(sorted(values))))
        elif done and json.dumps(config.get(k),sort_keys=True) not in values: bad.append((k,'differs from declared configuration'))
    _check(checks,'protocol_homogeneous',not bad,bad)
    bad=[]
    for r in done:
        h=artifact_root/'artifacts'/'revision'/r['run_id']/'history.json'
        n=len(json.loads(h.read_text())) if h.exists() else None
        if n!=r['config']['epochs']: bad.append((r['run_id'],f"history has {n} epochs, declared {r['config']['epochs']}"))
    _check(checks,'epochs_completed',not bad,bad)
    # 5. dataset identity and evaluation items
    splits={r['dataset']['split_id'] for r in done}; _check(checks,'single_split_id',len(splits)<=1,sorted(splits))
    mp=Path(config['manifest']); manifest_ok=mp.exists() and not json.loads(mp.read_text())['errors']
    _check(checks,'manifest_valid',manifest_ok,'manifest present and error-free' if manifest_ok else 'manifest missing or reports audit errors')
    if verify_hashes and mp.exists():
        cur=file_hash(mp); stale=[r['run_id'] for r in done if r['dataset']['manifest_sha256']!=cur]
        _check(checks,'manifest_unchanged',not stale,f'{len(stale)} run(s) used a different manifest version: {stale}')
    items=set()
    for r in done:
        s=artifact_root/'artifacts'/'revision'/r['run_id']/'samples.json'
        items.add(digest(json.loads(s.read_text())['test']) if s.exists() else None)
    _check(checks,'identical_test_items',len(items)<=1 and None not in items,'all runs evaluate the same ordered test items' if len(items)<=1 and None not in items else 'test selections differ or are missing')
    # 6. metrics
    bad=[]
    for r in done:
        m=r['metrics']; need=['accuracy','macro_f1','ece','nll','brier','stability_ssim','deletion_auc','silhouette']
        if any(not _finite(m.get(k)) for k in need): bad.append((r['run_id'],'missing or non-finite scalar metric')); continue
        p=m.get('pillars',{})
        if set(p)!=set(PILLARS) or any(not _finite(p[k]) or not 0<=p[k]<=1 for k in PILLARS): bad.append((r['run_id'],'pillar missing or outside [0,1]')); continue
        try: c=composite(p,r['config']['evaluation']['weights']['default'])
        except ValueError as e: bad.append((r['run_id'],str(e))); continue
        if abs(c-m.get('composite_trust',math.nan))>1e-12: bad.append((r['run_id'],'composite differs from pillars and weights'))
    _check(checks,'metrics_valid',not bad,bad)
    # 7. artifacts
    bad=[]
    if verify_artifacts:
        for r in done:
            for need in ('predictions.npz','explanations.json','attributions.npz','history.json','samples.json',*(('checkpoint.pt',) if require_checkpoints else ())):
                if not (artifact_root/'artifacts'/'revision'/r['run_id']/need).exists(): bad.append((r['run_id'],'missing '+need))
            if verify_hashes:
                for path,sha in r['artifacts'].items():
                    q=artifact_root/path
                    if not require_checkpoints and path.endswith('checkpoint.pt') and not q.exists(): continue
                    if not q.exists(): bad.append((r['run_id'],'missing '+path))
                    elif file_hash(q)!=sha: bad.append((r['run_id'],'hash mismatch '+path))
    _check(checks,'artifacts_present',not bad,bad if verify_artifacts else 'NOT CHECKED')
    # 8. provenance
    bad=[]
    for r in done:
        if r['config_hash']!=digest(r['config']): bad.append((r['run_id'],'config hash mismatch'))
        if not r.get('environment',{}).get('packages') or not r.get('source_revision') or r.get('seed') not in config['seeds']: bad.append((r['run_id'],'incomplete provenance'))
    revs={r['source_revision'] for r in done}
    if len(revs)>1: bad.append(('cohort',f'{len(revs)} different source revisions are mixed'))
    _check(checks,'provenance_complete',not bad,bad)
    # 9. explanation sample completeness (no silent drops, no constant fallbacks)
    bad=[]
    for r in done:
        e=artifact_root/'artifacts'/'revision'/r['run_id']/'explanations.json'
        if not e.exists(): continue
        d=json.loads(e.read_text()); st=d['stability_per_sample']; n=len(d['sample_ids'])
        if len(st)!=n or n!=len(d['targets']) or len(d['deletion']['salient']['auc_per_sample'])!=n or len(set(map(lambda v: round(v,12),st)))<2:
            bad.append((r['run_id'],'explanation sample counts inconsistent or constant stability'))
    _check(checks,'explanation_samples_complete',not bad,bad)
    return {'cohort':config.get('cohort'),'ok':all(c['passed'] for c in checks if c['critical']),'checks':checks}

def check_all(config_paths, raw_dir=Path('results/raw'), **kw):
    from src.cropxai.provenance import source_hash
    records=[json.loads(p.read_text()) for p in sorted(Path(raw_dir).glob('*.json'))]
    cur=source_hash(); reports=[]
    for cp in config_paths:
        cfg=json.loads(Path(cp).read_text()); rep=check_cohort(cfg,records,current_source=cur,**kw); rep['config']=str(cp); reports.append(rep)
    return {'ok':all(r['ok'] for r in reports),'reports':reports}
