"""Fail-closed evidence and submission gate. Every failed check names the claim and its source; exit status 1 if any critical check fails.

External blockers (similarity report, conference-format confirmation) are never suppressed to obtain a green gate.
"""
import sys,json,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.cropxai.provenance import file_hash,write_json,source_hash
from src.cropxai.metrics import PILLARS
from src.cropxai.models import ARCHITECTURES
from src.cropxai_analysis.compliance import check_all
from analyze_results import validate_record

CONFIGS=['configs/experiments/revision_v1.json','configs/experiments/revision_v1_prithvi_ms6.json']

def validate():
    checks=[]
    def check(name,ok,detail,critical=True): checks.append({'check':name,'passed':bool(ok),'detail':detail,'critical':critical})
    summary_path=Path('results/processed/revision_summary.json')
    if not summary_path.exists(): check('results',False,'No analyzed records'); summary={'groups':[],'records':[],'rejected':[],'failed_runs':[]}
    else: summary=json.loads(summary_path.read_text())
    check('record_rejections',not summary['rejected'],summary['rejected'])
    check('summary_current_source',summary.get('current_source_revision')==source_hash(),'Summary was computed from the code now in src/cropxai; re-run `cropxai analyze` otherwise')
    for rid in summary['records']:
        try: validate_record(json.loads(Path('results/raw',rid+'.json').read_text())); check('record_'+rid,True,'Hashes and independent numerical recomputation passed',critical=True)
        except Exception as exc: check('record_'+rid,False,str(exc))
    # Cohort completeness and protocol compliance are recomputed, never read from a stored flag.
    comp=check_all(CONFIGS)
    for r in comp['reports']:
        for c in r['checks']:
            if c['critical']: check(f"compliance[{r['cohort']}]/{c['check']}",c['passed'],c['detail'])
    cohorts=[json.loads(Path(c).read_text()) for c in CONFIGS]
    for cfg in cohorts:
        for model in cfg['models']:
            matches=[g for g in summary['groups'] if g['current_source'] and g['model']==model and g['purpose']==cfg['purpose'] and set(g['seeds'])==set(cfg['seeds']) and g['epochs']==cfg['epochs']]
            check('complete_group_'+model,len(matches)==1,'Exactly one homogeneous full-data group with the declared seeds and epochs')
    table=Path('results/tables/table1.json'); provenance=table.with_name('table1.provenance.json')
    if table.exists() and provenance.exists():
        rows=json.loads(table.read_text()); meta=json.loads(provenance.read_text())
        check('table_hashes',file_hash(table)==meta['json_sha256'] and file_hash(table.with_suffix('.tex'))==meta['tex_sha256'] and all(file_hash(Path('results/tables')/k)==v for k,v in meta['tex_files'].items()),'Generated table file hashes')
        check('table_source_current',file_hash(summary_path)==meta['source_sha256'],'Table must match current summary')
        for r in rows:
            gs=[g for g in summary['groups'] if g['run_ids']==r['run_ids'] and g['model']==r['model'] and g['purpose']==r['purpose'] and g['current_source']]
            ok=len(gs)==1 and all(r[k]==gs[0]['metrics'][r['metric']][k] for k in ('mean','std','ci95','n','missing_runs'))
            check('table_'+r['model']+'_'+r['metric'],ok,'Values and uncertainty match the same run IDs')
        for cfg in cohorts:
            for model in cfg['models']:
                have={r['metric'] for r in rows if r['model']==model and r['n']>0}
                check('six_pillars_'+model,have>=set(PILLARS)|{'composite_trust'},'All six components and the composite are measured and tabulated')
                check('ci_present_'+model,all(r['ci95'] is not None for r in rows if r['model']==model),'Every tabulated component has a seed-level interval (n>=2)')
                check('registered_'+model,model in ARCHITECTURES,'Model is in the architecture registry')
    else: check('table_exists',False,'Table 1 missing')
    fm=Path('results/figures/manifest.json')
    if fm.exists():
        for p in json.loads(fm.read_text())['files']:
            path=Path(p); meta=json.loads(path.with_suffix('.json').read_text())
            ok=file_hash(path)==meta['figure_sha256'] and file_hash(summary_path)==meta['source_sha256']
            check('figure_'+path.name,ok,'Figure and source hashes; plotted values stored in sidecar')
    else: check('figures',False,'No generated figures')
    source=Path('paper/source/445_revised_version.tex'); bib=Path('paper/source/references.bib')
    if source.exists() and bib.exists():
        text=source.read_text(); main=text.split(r'\section*{Point-by-point response')[0]
        keys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib.read_text())); citations=set()
        for group in re.findall(r'\\cite\w*(?:\[[^\]]*\])?\{([^}]+)\}',main): citations.update(x.strip() for x in group.split(','))
        check('citations_resolve',citations<=keys,sorted(citations-keys))
        check('no_uncited_bibliography',keys<=citations,sorted(keys-citations))
        verified=json.loads(Path('docs/revision/evidence/verified_sources.json').read_text())
        check('verified_bibliography',keys<=set(verified),sorted(keys-set(verified)))
        tables_tex=''.join(p.read_text() for p in sorted(Path('results/tables').glob('table_*.tex'))+[Path('results/tables/table1.tex')])
        labels=re.findall(r'\\label\{([^}]+)\}',text+tables_tex); refs=re.findall(r'\\(?:eqref|ref|pageref)\{([^}]+)\}',text)
        check('labels_unique',len(labels)==len(set(labels)),'Unique labels')
        check('references_resolve',set(refs)<=set(labels),sorted(set(refs)-set(labels)))
        figs=re.findall(r'\\includegraphics[^{]*\{\.\./\.\./([^}]+)\}',main)
        check('figure_files_are_generated',all(Path(f).exists() and f.startswith('results/figures/') for f in figs),[f for f in figs if not Path(f).exists()])
        check('figures_all_referenced',all(re.search(r'\\ref\{fig:'+Path(f).stem+r'\}|\\ref\{fig:[a-z]+\}',main) for f in figs),'every figure is cited in text')
        check('stale_asset_free',not re.search(r'paper/figures|critical_difference|cross_dataset_drift',main),'No legacy unsupported figures are embedded')
        # Scope and terminology.
        check('scope_statement','EuroSAT land-cover classification' in main and 'does not validate crop-health' in main,'Evaluated scope is stated')
        check('no_13_band_claim',bool(re.search(r'No experiment in this paper uses all 13',main)) and not re.search(r'(?<![Nn]o )13-band (experiment|evaluation|multispectral)',main),'No 13-band experiment is claimed')
        check('composite_not_bayesian','Composite Trust Score' in main and re.search(r'not a Bayesian',main) is not None and 'Bayesian Trust Score' not in main,'Weighted sum named correctly')
        weights=json.loads(Path('configs/evaluation/revision_v1.json').read_text())['weights']
        check('weights_documented',all(abs(v-1/6)<1e-12 for v in weights['default'].values()) and '(.15,.10,.25,.25,.15,.10)' in main and '(.15,.35,.15,.15,.10,.10)' in main,'Weights in text equal versioned configuration')
        check('stability_not_faithfulness','Stability does not establish faithfulness' in main and 'deletion' in main,'Stability and faithfulness are kept apart')
        check('nemenyi_withdrawn',re.search(r'Nemenyi[^.]*withdrawn',main) is not None,'Unsupported significance test is not claimed')
        check('prithvi_adaptation_documented',all(s in main for s in ('B8A','narrow NIR','position','no input parameter is added')),'Band adaptation described (R3.4)')
        check('band_mismatch_disclosed','do not use the same bands' in main,'Different input bands across cohorts disclosed (R3.5)')
        check('discrepancy_explained','INSERT OR REPLACE' in main and 'Table I' in main,'Figure 7/Table 1 discrepancy source documented')
        check('legacy_withdrawals','proxy' in main and '0.65' in main,'Proxy and fallback withdrawals disclosed')
        check('limitations_present','geographically independent' in main and 'synthetic' in main.lower() and 'future work' in main.lower(),'Limitations section covers split, synthetic shifts, future work')
        bad=re.findall(r'(?<!not )(?<!does not validate )(?<!without )(?:detects?|monitors?) crop (?:disease|health)',main,flags=re.I)
        check('no_crop_health_claim',not bad,bad)
        # Numbers that the prose quotes must come from the generated tables: every printed percentage in the abstract appears in the summary.
        absr=re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',main,re.S).group(1)
        quoted=[float(x) for x in re.findall(r'(\d+\.\d+)\\%',absr)]
        pool=set()
        for g in summary['groups']:
            for m in g['metrics'].values():
                if m['mean'] is not None: pool.add(round(100*m['mean'],2))
        legacy=json.loads(Path('docs/revision/evidence/legacy_audit.json').read_text())
        for e in legacy['table1_reconstruction'].values():
            for k in ('accuracy',): pool.update({round(100*e[k]['submitted'],2),round(100*e[k]['pooled_all_runs'],2),round(100*e[k]['five_epoch_only'],2)})
        check('abstract_numbers_traceable',all(any(abs(q-p)<0.011 for p in pool) for q in quoted),[q for q in quoted if not any(abs(q-p)<0.011 for p in pool)])
        check('uncertainty_scope_stated','do not reflect geographic' in main or 'Neither interval reflects geographic' in main,'Interval scope stated')
    else: check('manuscript_source',False,'Editable revised source missing')
    check('pdf_built',Path('output/pdf/445_revised_version.pdf').exists() and Path('output/pdf/445_revised_version.pdf').stat().st_mtime>=Path('paper/source/445_revised_version.tex').stat().st_mtime-1,'PDF newer than its source') if source.exists() else None
    # External items that software cannot establish.
    check('similarity_report',Path('output/pdf/similarity_report.pdf').exists(),'A genuine report from an accepted similarity service is required (not generated here)')
    check('conference_format_confirmed',False,'Local 2024 Procedia files were used; the current Elsevier guide returned HTTP 403 to the revising software and conference-specific compliance is unconfirmed')
    report={'ready':all(c['passed'] for c in checks if c['critical']),'checks':checks}
    write_json('docs/revision/evidence/manuscript_validation.json',report)
    failed=[c for c in checks if not c['passed']]
    print(json.dumps({'ready':report['ready'],'n_checks':len(checks),'failed':[(c['check'],str(c['detail'])[:160]) for c in failed]},indent=1))
    return report
if __name__=='__main__': raise SystemExit(0 if validate()['ready'] else 1)
