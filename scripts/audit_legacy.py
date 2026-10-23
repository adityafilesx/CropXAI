"""Reconstruct the submitted Table 1 from the legacy SQLite stores and flag contaminated records.

Read-only on both databases. Output: docs/revision/evidence/legacy_audit.json
Findings are computed, not asserted: each one carries the run IDs it rests on.
"""
import sys, json, sqlite3
from contextlib import closing
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cropxai.provenance import write_json

def _reviewer_db():
    # The stale reviewer snapshot was moved (not deleted) by scripts/package_revision.py
    for c in ('reviewer_package/legacy_stale_reviewer_db/experiment_db.sqlite', 'reviewer_package/experiment_db.sqlite', 'legacy_stale_reviewer_db/experiment_db.sqlite'):
        if Path(c).exists(): return c
    raise FileNotFoundError('stale reviewer-package database not found')
DBS = ['results/experiment_db.sqlite', _reviewer_db()]
# Values printed in the submitted Table I (docs/revision/evidence/original_manuscript.txt).
SUBMITTED = {'Prithvi-100M': {'accuracy': .9530, 'f1': .9529}, 'ResNet-18': {'accuracy': .9153, 'f1': .9145},
             'ViT-Base': {'accuracy': .9541, 'f1': .9540}}
FALLBACK_VALUES = (0.5, 0.65)  # constants returned by the original trust.py on ImportError / missing layer


def load(db):
    with closing(sqlite3.connect(f'file:{db}?mode=ro', uri=True)) as c:
        c.row_factory = sqlite3.Row
        return [dict(r) for r in c.execute(
            'SELECT e.run_id,e.model,e.epochs,e.seed,e.finetune_mode,m.metric_name,m.metric_value '
            'FROM experiments e JOIN metrics m USING(run_id) ORDER BY e.model,e.epochs,e.seed')]


def main():
    rows = load(DBS[0]); runs = defaultdict(dict)
    for r in rows:
        runs[(r['model'], r['epochs'], r['seed'], r['run_id'])][r['metric_name']] = r['metric_value']
    out = {'source': DBS[0], 'n_runs': len(runs)}

    # 1. Does the submitted Table I equal the unweighted mean over every run of the model, 1- and 5-epoch alike?
    recon = {}
    for model, sub in SUBMITTED.items():
        sel = [(k, v) for k, v in runs.items() if k[0] == model]
        entry = {'n_runs': len(sel), 'epochs_pooled': sorted({k[1] for k, _ in sel}),
                 'seeds_pooled': sorted({k[2] for k, _ in sel}), 'run_ids': [k[3] for k, _ in sel]}
        for metric in ('accuracy', 'f1'):
            pooled = float(np.mean([v[metric] for _, v in sel]))
            five = [v[metric] for k, v in sel if k[1] == 5]
            entry[metric] = {'submitted': sub[metric], 'pooled_all_runs': pooled,
                             'abs_diff_pooled': abs(pooled - sub[metric]),
                             'five_epoch_only': float(np.mean(five)) if five else None,
                             'n_five_epoch': len(five)}
        entry['matches_pooled_to_4dp'] = all(round(entry[m]['pooled_all_runs'], 4) == sub[m] or
                                             abs(entry[m]['pooled_all_runs'] - sub[m]) < 6e-5 for m in ('accuracy', 'f1'))
        recon[model] = entry
    out['table1_reconstruction'] = recon

    five={}
    for model in sorted({k[0] for k in runs}):
        v=[x['accuracy'] for k,x in runs.items() if k[0]==model and k[1]==5]
        if v: five[model]={'n_runs':len(v),'accuracy_mean':float(np.mean(v))}
    out['five_epoch_accuracy_all_models']=five
    # 2. Runs whose trust_T3 equals a fallback constant exactly.
    out['fallback_trust_runs'] = [{'model': k[0], 'epochs': k[1], 'seed': k[2], 'run_id': k[3], 'trust_T3': v['trust_T3']}
                                  for k, v in sorted(runs.items()) if v.get('trust_T3') in FALLBACK_VALUES]

    # 3. Does 'Prithvi-100M' reproduce 'ViT-Base' bit-for-bit for shared (epochs, seed) cells?
    by = {(k[0], k[1], k[2]): v for k, v in runs.items()}
    identical = []
    for (m, e, s), v in by.items():
        if m == 'Prithvi-100M' and ('ViT-Base', e, s) in by:
            w = by[('ViT-Base', e, s)]
            shared = sorted(set(v) & set(w) - {'final_loss'})
            identical.append({'epochs': e, 'seed': s, 'metrics_compared': shared,
                              'identical': {k: v[k] == w[k] for k in shared}})
    out['prithvi_vs_vit_identity'] = identical

    # 4. Cohort composition per model (what was averaged).
    comp = defaultdict(lambda: defaultdict(list))
    for (m, e, s, rid) in runs: comp[m][e].append(s)
    out['cohort_composition'] = {m: {str(e): sorted(s) for e, s in d.items()} for m, d in comp.items()}

    # 5. Reviewer-package database: record count and overlap with the main store.
    rp = load(DBS[1]); rp_runs = {(r['model'], r['epochs'], r['seed']) for r in rp}
    out['reviewer_package_db'] = {'n_runs': len(rp_runs), 'cells_missing_from_main': sorted(map(list, rp_runs - set(by))),
                                  'cells_in_main_not_in_package': len(set(by) - rp_runs)}
    # 6. Same run_id, different metrics: legacy run_id is sha256(model|dataset|seed|epochs|mode|git_sha)[:16] and metrics were written
    #    with INSERT OR REPLACE, so re-running a configuration overwrote the earlier values under an unchanged run row.
    import hashlib
    def meta_and_metrics(db):
        with closing(sqlite3.connect(f'file:{db}?mode=ro', uri=True)) as c:
            c.row_factory = sqlite3.Row
            ex = {r['run_id']: dict(r) for r in c.execute('SELECT * FROM experiments')}
            mt = defaultdict(dict)
            for r in c.execute('SELECT run_id,metric_name,metric_value FROM metrics'): mt[r['run_id']][r['metric_name']] = r['metric_value']
            fg = [dict(r) for r in c.execute('SELECT figure_name,source_run_ids,generated_at FROM figures')]
        return ex, mt, fg
    exm, mtm, fgm = meta_and_metrics(DBS[0]); exr, mtr, fgr = meta_and_metrics(DBS[1])
    fp_ok = all(hashlib.sha256(f"{r['model']}|{r['dataset']}|{r['seed']}|{r['epochs']}|{r['finetune_mode']}|{r['git_sha']}".encode()).hexdigest()[:16] == rid for rid, r in {**exm, **exr}.items())
    diverged = []
    for rid in sorted(set(mtm) & set(mtr)):
        d = {k: {'main': mtm[rid][k], 'reviewer_package': mtr[rid][k]} for k in sorted(set(mtm[rid]) & set(mtr[rid])) if mtm[rid][k] != mtr[rid][k]}
        if d: diverged.append({'run_id': rid, 'model': exm[rid]['model'], 'epochs': exm[rid]['epochs'], 'seed': exm[rid]['seed'],
                               'same_experiment_row': exm[rid]['timestamp'] == exr[rid]['timestamp'], 'differing_metrics': d})
    out['database_divergence'] = {'run_id_is_config_fingerprint': fp_ok, 'runs_with_differing_metrics': diverged,
        'figures_main': fgm, 'figures_reviewer_package': fgr,
        'experiment_timestamps': {rid: exm[rid]['timestamp'] for r in diverged for rid in [r['run_id']]},
        'mechanism': 'INSERT OR REPLACE on metrics + deterministic run_id: a re-execution replaced metrics, keeping the first execution row and timestamp'}
    # 7. What the preserved submitted figure files themselves print (documentary link to the reviewer-package values).
    import subprocess
    from src.cropxai.provenance import file_hash
    figs = {}
    for name in ('baseline_results', 'training_curve'):
        f = Path('paper/figures') / f'{name}.pdf'
        txt = subprocess.run(['pdftotext', str(f), '-'], capture_output=True, text=True).stdout
        figs[name] = {'path': str(f), 'sha256': file_hash(f), 'text': ' '.join(txt.split())}
    out['submitted_figure_files'] = figs
    write_json('docs/revision/evidence/legacy_audit.json', out)
    print(json.dumps({'table1_matches_pooled': {m: e['matches_pooled_to_4dp'] for m, e in recon.items()},
                      'fallback_runs': len(out['fallback_trust_runs']),
                      'prithvi_vit_identical_cells': sum(all(i['identical'].values()) for i in identical),
                      'diverged_runs': [(d['model'], d['epochs'], d['seed']) for d in out['database_divergence']['runs_with_differing_metrics']]}, indent=1))


if __name__ == '__main__': main()
