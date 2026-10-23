"""Export immutable legacy rows and homogeneous summaries; no inferred measurements."""
from contextlib import closing
import sys, os, json, sqlite3
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cropxai.metrics import mean_ci, PILLARS, sensitivity
from src.cropxai.provenance import write_json, file_hash, digest, source_hash
from src.cropxai_analysis.stats import t_interval, paired_seed_difference, bootstrap_mean, bootstrap_paired_difference, bonferroni_level

def legacy_analysis():
    exports={}; summaries=[]
    for db in ['results/experiment_db.sqlite','reviewer_package/legacy_stale_reviewer_db/experiment_db.sqlite','reviewer_package/experiment_db.sqlite']:
        if not Path(db).exists():
            print(f'legacy database {db} not present in this checkout; skipped'); continue
        with closing(sqlite3.connect(f'file:{db}?mode=ro',uri=True)) as conn:
            conn.row_factory=sqlite3.Row
            rows=[dict(r) for r in conn.execute('SELECT e.*,m.metric_name,m.metric_value,m.is_validated FROM experiments e JOIN metrics m USING(run_id) ORDER BY e.run_id,m.metric_name')]
        exports[db]=rows; grouped=defaultdict(list)
        for row in rows:
            grouped[(row['model'],row['dataset'],row['epochs'],row['finetune_mode'],row['metric_name'])].append(row)
        for (model,dataset,epochs,mode,metric),rs in grouped.items():
            seeds=[r['seed'] for r in rs]
            if len(seeds)!=len(set(seeds)): raise ValueError('Duplicate seeds in legacy homogeneous group')
            summaries.append({'database':db,'model_label':model,'dataset':dataset,'epochs':epochs,'finetune_mode':mode,'metric':metric,
                'run_ids':[r['run_id'] for r in rs],'seeds':seeds,'evaluation_split':'test according to legacy trainer; saved predictions unavailable',
                'status':'RECOVERED_UNVERIFIED','caution':'No split/checkpoint hashes; trust_T3 may contain fallbacks; foundation names are proxies',
                **mean_ci([r['metric_value'] for r in rs],expected_runs=5)})
    write_json('results/processed/legacy_records.json',exports)
    write_json('results/processed/legacy_summary.json',summaries)
    return exports,summaries

def validate_record(r):
    if r['status']!='completed': raise ValueError('Run failed')
    light=os.environ.get('CROPXAI_LIGHT')=='1'   # reviewer package: model checkpoints are not bundled, their hashes stay recorded
    for p,sha in r['artifacts'].items():
        if light and p.endswith('checkpoint.pt') and not Path(p).exists(): continue
        if not Path(p).exists() or file_hash(p)!=sha: raise ValueError('Artifact hash mismatch: '+p)
    if r['config_hash']!=digest(r['config']): raise ValueError('Config hash mismatch')
    from src.cropxai.metrics import composite
    actual=composite(r['metrics']['pillars'],r['config']['evaluation']['weights']['default'])
    if abs(actual-r['metrics']['composite_trust'])>1e-12: raise ValueError('Composite differs from components')
    import numpy as np
    from src.cropxai.metrics import prediction_metrics, representation_quality, pillar_scores
    from src.cropxai.data import stratified_indices, validate_bands
    from src.cropxai.models import ARCHITECTURES
    if r['model']['architecture'] not in ARCHITECTURES: raise ValueError('Unregistered actual model')
    spec=validate_bands(r['model']['bands'],r['model']['channels'])
    if r['config'].get('input_representation')!=spec.name or r['model'].get('input_spec')!=spec.name: raise ValueError('Input representation mismatch')
    if r['model']['weights'] is None and r['config']['pretrained']: raise ValueError('Pretrained run lacks weight identity')
    root=Path('artifacts/revision')/r['run_id']; ev=r['config']['evaluation']
    selections=json.loads((root/'samples.json').read_text())
    with np.load(root/'predictions.npz',allow_pickle=False) as p:
        if p['sample_ids'].tolist()!=selections['test']: raise ValueError('Prediction ordering differs from test selection')
        measured=prediction_metrics(p['probabilities'],p['labels'],ev['ece_bins'])
        for k in ['accuracy','macro_f1','weighted_f1','ece','nll','brier','normalized_entropy']:
            if not np.isclose(measured[k],r['metrics'][k],atol=1e-12): raise ValueError('Prediction/metric mismatch: '+k)
        if measured['confusion_matrix']!=r['metrics']['confusion_matrix']: raise ValueError('Confusion counts differ')
        ix=stratified_indices(p['labels'],ev['representation_per_class'],ev['selection_seed'])
        rep=representation_quality(p['features'][ix],p['labels'][ix])
    e=json.loads((root/'explanations.json').read_text())
    auc=float(np.mean(np.trapezoid(e['deletion']['salient']['curves'],x=e['deletion']['salient']['fractions'],axis=1)))
    if not np.isclose(auc,r['metrics']['deletion_auc'],atol=1e-12): raise ValueError('Deletion AUC/curve mismatch')
    if not np.isclose(np.mean(e['stability_per_sample']),r['metrics']['stability_ssim'],atol=1e-12): raise ValueError('Stability aggregation mismatch')
    corrupted=[]
    for row in r['metrics']['corruptions']:
        with np.load(row['predictions'],allow_pickle=False) as p:
            if p['sample_ids'].tolist()!=selections['test']: raise ValueError('Corruption sample mismatch')
            value=prediction_metrics(p['probabilities'],p['labels'],ev['ece_bins'])['accuracy']
        if not np.isclose(value,row['accuracy'],atol=1e-12): raise ValueError('Corruption accuracy mismatch')
        corrupted.append(value)
    measured['deletion_auc']=auc
    expected=pillar_scores(measured,corrupted,rep['representation'])
    if any(not np.isclose(expected[k],r['metrics']['pillars'][k],atol=1e-12) for k in PILLARS): raise ValueError('Pillar normalization mismatch')

BASE_METRICS=['accuracy','macro_f1','weighted_f1','ece','nll','brier','normalized_entropy','stability_ssim','deletion_auc','silhouette','composite_trust']
DERIVED_METRICS=['explanation_control_normalized','accuracy_retention','aopc_salient','deletion_auc_random','deletion_auc_least','deletion_gap_random','error_detection_auroc']

def run_values(r):
    """All scalar metrics of one validated run, including those derived from saved artifacts (never re-estimated)."""
    m=r['metrics']; e=json.loads((Path('artifacts/revision')/r['run_id']/'explanations.json').read_text())
    corr=[c['accuracy'] for c in m['corruptions']]
    vals={k:m[k] for k in BASE_METRICS}; vals.update(m['pillars'])
    vals['accuracy_retention']=sum(corr)/len(corr)/m['accuracy']
    vals['aopc_salient']=e['deletion']['salient']['aopc']; vals['deletion_auc_random']=e['deletion']['random']['auc']; vals['deletion_auc_least']=e['deletion']['least']['auc']
    vals['deletion_gap_random']=vals['deletion_auc_random']-m['deletion_auc']
    # POST-HOC (defined after inspecting results; never used in the primary composite): symmetric control-normalized deletion score.
    # 0.5 = salient deletion is no faster than random deletion; >0.5 = faster (more faithful); <0.5 = slower.
    tot=vals['deletion_auc_random']+m['deletion_auc']
    if tot<=0: raise ValueError('Degenerate deletion AUCs')
    vals['explanation_control_normalized']=(1+(vals['deletion_auc_random']-m['deletion_auc'])/tot)/2
    vals['error_detection_auroc']=m['error_detection_auroc']  # None when a run has no errors; handled as missing, never zero
    return vals

def lineage(manifest_path):
    d=json.loads(Path(manifest_path).read_text()); return d.get('rgb_split_id',d['split_id'])

def analyze():
    legacy_analysis(); records=[]; rejected=[]; failed=[]; superseded=[]; current_now=source_hash()
    for p in sorted(Path('results/raw').glob('*.json')):
        r=json.loads(p.read_text())
        if r['status']=='failed':
            # Failed runs are disclosed, never aggregated; cohort completeness is gated separately.
            failed.append({'run_id':r['run_id'],'model':r['model'].get('architecture'),'seed':r['seed'],'purpose':r['config']['purpose'],'error':r['errors'][0] if r['errors'] else None}); continue
        try: validate_record(r); records.append(r)
        except Exception as exc:
            # Records written by earlier code revisions predate the current schema; they are kept as evidence but never aggregated.
            (superseded if r['source_revision']!=current_now else rejected).append({'path':str(p),'run_id':r['run_id'],'reason':str(exc),'source_revision':r['source_revision']})
    current=current_now; groups=defaultdict(list)
    for r in records:
        # Selected seed changes independently. Never mix sources, settings, splits, or purposes.
        cfg={k:v for k,v in r['config'].items() if k not in ('selected_seed','selected_model','models','blocked_models')}
        key=(r['config']['purpose'],r['config'].get('cohort'),r['model']['architecture'],lineage(r['dataset']['manifest']),r['source_revision'],digest(cfg))
        groups[key].append(r)
    summary=[]
    for key,rs in groups.items():
        seeds=[r['seed'] for r in rs]
        if len(seeds)!=len(set(seeds)):
            rejected.append({'group':list(key),'reason':'Duplicate seed; select explicit run IDs before analysis'}); continue
        per_run=[run_values(r) for r in rs]; expected=len(rs[0]['config']['seeds']); metrics={}
        for metric in [*BASE_METRICS,*PILLARS,*DERIVED_METRICS]:
            vals=[v[metric] for v in per_run if v[metric] is not None]
            metrics[metric]=mean_ci(vals,expected_runs=expected)
            if metric=='error_detection_auroc': metrics[metric]['n_with_errors']=len(vals)
        corr={}
        for c in rs[0]['metrics']['corruptions']:
            name=f"{c['kind']}_{c['severity']:g}"
            corr[name]=mean_ci([next(x['accuracy'] for x in r['metrics']['corruptions'] if x['kind']==c['kind'] and x['severity']==c['severity']) for r in rs],expected_runs=expected)
        epochs=len(json.loads((Path('artifacts/revision')/rs[0]['run_id']/'history.json').read_text()))
        history=[{'epoch':e+1,'validation_accuracy':mean_ci([json.loads((Path('artifacts/revision')/r['run_id']/'history.json').read_text())[e]['validation']['accuracy'] for r in rs]),
                  'training_loss':mean_ci([json.loads((Path('artifacts/revision')/r['run_id']/'history.json').read_text())[e]['training_loss'] for r in rs])} for e in range(epochs)]
        summary.append({'purpose':key[0],'cohort':key[1],'model':key[2],'split_lineage':key[3],'source_revision':key[4],'protocol_hash':key[5],
                        'current_source':key[4]==current,'epochs':rs[0]['config']['epochs'],'input_representation':rs[0]['config']['input_representation'],
                        'bands':rs[0]['config']['bands'],'seeds':seeds,'run_ids':[r['run_id'] for r in rs],'evaluation_split':'test',
                        'metrics':metrics,'corruption_accuracy':corr,'validation_history':history})
    out={'current_source_revision':current,'groups':summary,'rejected':rejected,'failed_runs':failed,'superseded_unvalidated':superseded,'records':[r['run_id'] for r in records]}
    write_json('results/processed/revision_summary.json',out)
    # Sensitivity only within exact protocol, source and split; no smoke/benchmark mixing.
    schemes=json.loads(Path('configs/evaluation/revision_v1.json').read_text())['weights']
    complete=[g for g in summary if g['current_source'] and g['purpose'] in ('controlled_benchmark','multispectral_foundation_benchmark') and g['metrics']['accuracy']['n']==len(g['seeds'])]
    sens=[]
    comparisons=defaultdict(dict)
    for g in complete:
        if g['purpose']=='controlled_benchmark': comparisons[(g['purpose'],g['split_lineage'],g['source_revision'],g['protocol_hash'])][g['model']]={p:g['metrics'][p]['mean'] for p in PILLARS}
    for k,ms in comparisons.items(): sens.append({'protocol':list(k),'confounded_cross_input':False,'models':sorted(ms),'analysis':sensitivity(ms,schemes)})
    allm={g['model']:{p:g['metrics'][p]['mean'] for p in PILLARS} for g in complete}
    rgb_models={m for ms in comparisons.values() for m in ms}
    if set(allm)-rgb_models and len(allm)>1:
        sens.append({'protocol':['cross_input_all_models'],'confounded_cross_input':True,'models':sorted(allm),
                     'note':'Models differ in pretraining, input channels and corruption units; ranks are descriptive only.','analysis':sensitivity(allm,schemes)})
    write_json('results/processed/weight_sensitivity.json',sens)
    posthoc_explanation(complete,schemes)
    paired_comparisons(records,complete)
    print('Validated revision groups:',len(summary),'current:',sum(g['current_source'] for g in summary),'failed:',len(failed),'rejected:',len(rejected),'superseded:',len(superseded))
    return out

def posthoc_explanation(complete,schemes):
    """Post-hoc sensitivity: replace the explanation component by its control-normalized version. Not a primary result."""
    import numpy as np
    from scipy.stats import rankdata
    from src.cropxai.metrics import composite
    rows={}
    for g in complete:
        pil={p:g['metrics'][p]['mean'] for p in PILLARS}; alt=dict(pil); alt['explanation']=g['metrics']['explanation_control_normalized']['mean']
        rows[g['model']]={'primary_explanation':pil['explanation'],'posthoc_explanation':alt['explanation'],
                          'primary_composite':composite(pil,schemes['default']),'posthoc_composite':composite(alt,schemes['default']),
                          'cohort':g['cohort']}
    names=sorted(rows)
    for key in ('primary_composite','posthoc_composite'):
        r=rankdata([-rows[m][key] for m in names])
        for m,v in zip(names,r): rows[m]['rank_'+key]=float(v)
    write_json('results/processed/posthoc_explanation_sensitivity.json',{'status':'POST-HOC; defined after inspecting results; default weights; primary composite unchanged','models':rows})

def paired_comparisons(records,complete):
    """Exploratory pairwise analysis on identical test items; seed-level pairing preserved."""
    import numpy as np
    by_model={}
    for g in complete:
        rs=[r for r in records if r['run_id'] in g['run_ids']]; by_model[g['model']]={'group':g,'runs':{r['seed']:r for r in rs}}
    names=sorted(by_model); result={'statistical_scope':'exploratory; training-seed and test-item uncertainty on one fixed random split; no geographic variance','pairs':[]}
    # Per-item correctness averaged over seeds, aligned by sample ID.
    items={}
    for m,d in by_model.items():
        acc=[]; ids=None
        for seed,r in sorted(d['runs'].items()):
            z=np.load(Path('artifacts/revision')/r['run_id']/'predictions.npz',allow_pickle=False); sid=z['sample_ids'].tolist()
            if ids is None: ids=sid
            elif ids!=sid: raise ValueError('Seeds of one model evaluated different items')
            acc.append((z['probabilities'].argmax(1)==z['labels']).astype(float))
        items[m]=(ids,np.mean(acc,axis=0))
    npairs=max(1,len(names)*(len(names)-1)//2); level=bonferroni_level(.95,npairs)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if items[a][0]!=items[b][0]: result['pairs'].append({'a':a,'b':b,'error':'different test item sets; not compared'}); continue
            seed_level={}
            for metric in ['accuracy','macro_f1','composite_trust',*PILLARS]:
                va={s:(r['metrics']['pillars'][metric] if metric in PILLARS else r['metrics'][metric]) for s,r in by_model[a]['runs'].items()}
                vb={s:(r['metrics']['pillars'][metric] if metric in PILLARS else r['metrics'][metric]) for s,r in by_model[b]['runs'].items()}
                seed_level[metric]=paired_seed_difference(va,vb)
            result['pairs'].append({'a':a,'b':b,'cross_input':by_model[a]['group']['input_representation']!=by_model[b]['group']['input_representation'],
                'accuracy_item_bootstrap':bootstrap_paired_difference(items[a][1],items[b][1],level=level),'bonferroni_comparisons':npairs,
                'seed_level_differences':seed_level})
    result['accuracy_item_bootstrap_per_model']={m:bootstrap_mean(v[1]) for m,v in items.items()}
    write_json('results/processed/paired_comparisons.json',result)

if __name__=='__main__': analyze()
