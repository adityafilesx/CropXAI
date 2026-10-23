"""Figures and tables derived only from validated machine-readable records of the current source revision.

Every figure writes a JSON sidecar holding the plotted numbers and the SHA-256 of its source summary, so that
`validate_manuscript.py` can detect a stale or hand-edited figure. Color follows the model (fixed order), never rank.
"""
import sys, json, os
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MPLCONFIGDIR',str(Path('tmp/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.cropxai.metrics import PILLARS, calibration_bins
from src.cropxai.provenance import write_json,file_hash
from analyze_results import analyze,validate_record

# Validated categorical order (adjacent pairs): blue, orange, aqua, yellow. Marker shape is a redundant encoding.
MODEL_ORDER=['ResNet-18','ViT-Base','ConvNeXt-T','Prithvi-EO-1.0-100M']
COLOR={'ResNet-18':'#2a78d6','ViT-Base':'#eb6834','ConvNeXt-T':'#1baf7a','Prithvi-EO-1.0-100M':'#eda100'}
MARK={'ResNet-18':'o','ViT-Base':'s','ConvNeXt-T':'^','Prithvi-EO-1.0-100M':'D'}
SHORT={'ResNet-18':'ResNet-18','ViT-Base':'ViT-Base','ConvNeXt-T':'ConvNeXt-T','Prithvi-EO-1.0-100M':'Prithvi (6-band)'}
INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'
LABELS={'performance':'Performance (macro-F1)','explanation':'Explanation (1 - deletion AUC)','calibration':'Calibration (1 - ECE)','robustness':'Robustness (corrupted accuracy)',
        'uncertainty':'Uncertainty (1 - Brier/2)','representation':'Representation ((silhouette+1)/2)','composite_trust':'Composite Trust Score'}
PRIMARY=('controlled_benchmark','multispectral_foundation_benchmark')

def number(x,d=4): return r'\NC{}' if x is None else f'{x:.{d}f}'
def tex(s): return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def ci_text(ci,d=4): return r'\NC{}' if ci is None else f'[{ci[0]:.{d}f}, {ci[1]:.{d}f}]'

def complete_groups(summary):
    gs=[g for g in summary['groups'] if g['current_source'] and g['purpose'] in PRIMARY and g['metrics']['accuracy']['n']==len(g['seeds'])==g['metrics']['accuracy']['n']]
    gs=[g for g in gs if g['metrics']['accuracy']['missing_runs']==0]
    return sorted(gs,key=lambda g: MODEL_ORDER.index(g['model']) if g['model'] in MODEL_ORDER else 99)

def style():
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,
                         'text.color':INK,'axes.titlesize':8.5,'axes.titleweight':'normal','legend.frameon':False,'legend.fontsize':7.5,'grid.color':GRID,'grid.linewidth':.6,
                         'axes.grid':True,'axes.axisbelow':True,'lines.linewidth':1.4,'lines.markersize':5,'pdf.fonttype':42,'savefig.facecolor':'white','figure.facecolor':'white'})

def load_runs(g):
    return [json.loads(Path('results/raw',rid+'.json').read_text()) for rid in g['run_ids']]

def generate():
    summary=analyze(); groups=complete_groups(summary)
    if not groups: raise SystemExit('No complete current-source group; nothing to plot. Run the campaigns and `cropxai analyze`.')
    paired=json.loads(Path('results/processed/paired_comparisons.json').read_text()); sens=json.loads(Path('results/processed/weight_sensitivity.json').read_text())
    out=Path('results/figures'); tables=Path('results/tables'); out.mkdir(exist_ok=True,parents=True); tables.mkdir(exist_ok=True,parents=True)
    for f in out.glob('*'): f.unlink()   # figures are regenerated wholesale; stale files must not survive
    source='results/processed/revision_summary.json'; source_sha=file_hash(source); files=[]; style()
    ids=[rid for g in groups for rid in g['run_ids']]; names=[g['model'] for g in groups]
    def save(name,fig,data,run_ids,caption):
        path=out/(name+'.pdf'); fig.savefig(path,bbox_inches='tight'); plt.close(fig)
        write_json(path.with_suffix('.json'),{'source':source,'source_sha256':source_sha,'run_ids':run_ids,'values':data,'caption_note':caption,'figure_sha256':file_hash(path),'script_sha256':file_hash(__file__)})
        files.append(str(path))
    def legend(ax,loc='best'): ax.legend(handles=[plt.Line2D([],[],color=COLOR[m],marker=MARK[m],lw=0,label=SHORT[m]) for m in names],loc=loc,ncol=1)
    runs={g['model']:load_runs(g) for g in groups}
    for rs in runs.values():
        for r in rs: validate_record(r)

    # F1 accuracy with seed t-interval (thick) and test-item bootstrap interval (thin)
    fig,ax=plt.subplots(figsize=(4.6,2.4)); vals=[]
    boot=paired['accuracy_item_bootstrap_per_model']
    for i,g in enumerate(groups):
        a=g['metrics']['accuracy']; ci=a['ci95']; b=boot[g['model']]['ci']
        ax.plot([a['mean']],[i],MARK[g['model']],color=COLOR[g['model']],zorder=3)
        ax.plot(ci,[i,i],color=COLOR[g['model']],lw=3,solid_capstyle='butt',zorder=2); ax.plot(b,[i+.18,i+.18],color=INK2,lw=1,zorder=2)
        vals.append({'model':g['model'],'mean':a['mean'],'seed_ci95':ci,'item_bootstrap_ci95':b,'n_seeds':a['n']})
    ax.set_yticks(range(len(groups)),[SHORT[m] for m in names]); ax.invert_yaxis(); ax.set_xlabel('Test accuracy'); ax.grid(axis='y',visible=False)
    ax.set_title('Thick: 95% t-interval over seeds. Thin gray: 95% bootstrap over test images.',loc='left',color=INK2,fontsize=7)
    save('accuracy',fig,vals,ids,'seed t interval and test-item bootstrap')

    # F2 six pillars + composite, small multiples
    fig,axes=plt.subplots(2,4,figsize=(7.2,3.9),sharey=False); data={}
    for ax,metric in zip(axes.flat,(*PILLARS,'composite_trust')):
        for i,g in enumerate(groups):
            s=g['metrics'][metric]; ci=s['ci95']
            ax.plot([i],[s['mean']],MARK[g['model']],color=COLOR[g['model']],zorder=3)
            if ci: ax.plot([i,i],ci,color=COLOR[g['model']],lw=2.5,solid_capstyle='butt',zorder=2)
            data.setdefault(metric,[]).append({'model':g['model'],'mean':s['mean'],'ci95':ci,'n':s['n']})
        ax.set_title(LABELS[metric],loc='left',fontsize=7); ax.set_xticks(range(len(groups)),['']*len(groups)); ax.set_xlim(-.6,len(groups)-.4); ax.grid(axis='x',visible=False)
        lo=min(d['ci95'][0] if d['ci95'] else d['mean'] for d in data[metric]); hi=max(d['ci95'][1] if d['ci95'] else d['mean'] for d in data[metric]); pad=max(.02,(hi-lo)*.25); ax.set_ylim(max(0,lo-pad),min(1,hi+pad))
    axes.flat[-1].axis('off'); legend(axes.flat[-1],'center')
    fig.suptitle('Component scores in [0,1] (higher is better); bars are 95% t-intervals over seeds. Note the per-panel y-axes.',x=.01,ha='left',fontsize=7.5,color=INK2)
    fig.tight_layout(rect=(0,0,1,.95)); save('pillars',fig,data,ids,'six pillars + composite')

    # F3 pooled reliability diagram (all seeds pooled, same bins as ECE)
    fig,ax=plt.subplots(figsize=(3.4,3.4)); ax.plot([0,1],[0,1],':',color=INK2,lw=1); data=[]
    for g in groups:
        conf=[];corr=[]
        for r in runs[g['model']]:
            z=np.load(Path('artifacts/revision')/r['run_id']/'predictions.npz',allow_pickle=False); p=z['probabilities']; conf.append(p.max(1)); corr.append(p.argmax(1)==z['labels'])
        conf=np.concatenate(conf); corr=np.concatenate(corr)
        bins=calibration_bins(np.column_stack([conf,1-conf]),np.where(corr,0,1),15)   # two-class view reproduces top-label binning exactly
        pts=[(b['confidence'],b['accuracy'],b['n']) for b in bins if b['n']]
        ax.plot([p[0] for p in pts],[p[1] for p in pts],marker=MARK[g['model']],color=COLOR[g['model']],ms=4)
        data.append({'model':g['model'],'bins':[{'confidence':a,'accuracy':b,'n':n} for a,b,n in pts],'pooled_ece':sum(n/len(conf)*abs(b-a) for a,b,n in pts)})
    ax.set(xlabel='Mean confidence',ylabel='Observed accuracy',xlim=(.2,1.02),ylim=(.2,1.02)); legend(ax,'upper left'); ax.set_title('Reliability, 15 bins, seeds pooled',loc='left',fontsize=7.5)
    save('reliability',fig,data,ids,'reliability pooled over seeds');

    # F4 deletion curves
    fig,axes=plt.subplots(1,len(groups),figsize=(1.9*len(groups)+.4,2.3),sharey=True); axes=np.atleast_1d(axes); data=[]
    for ax,g in zip(axes,groups):
        entry={'model':g['model']}
        for order,ls,col,lab in (('salient','-',COLOR[g['model']],'most salient first'),('random','--',INK2,'random'),('least',':',INK2,'least salient first')):
            curves=[]
            for r in runs[g['model']]:
                e=json.loads((Path('artifacts/revision')/r['run_id']/'explanations.json').read_text()); d=e['deletion'][order]; curves.append(np.mean(d['curves'],axis=0)); fr=d['fractions']
            m=np.mean(curves,axis=0); ax.plot(fr,m,ls,color=col,label=lab); entry[order]={'fractions':fr,'mean_probability':m.tolist()}
        data.append(entry); ax.set_title(SHORT[g['model']],fontsize=7.5,loc='left'); ax.set_xlabel('Fraction of patches deleted'); ax.set_ylim(0,1)
    axes[0].set_ylabel('Fixed-target probability'); axes[0].legend(fontsize=6.5); save('deletion',fig,data,ids,'deletion curves, mean over seeds and 100 images')

    # F5 robustness
    fig,axes=plt.subplots(1,2,figsize=(5.4,2.5),sharey=True); data=[]
    for ax,kind in zip(axes,('gaussian_noise','brightness')):
        for g in groups:
            sev=sorted({float(k.rsplit('_',1)[1]) for k in g['corruption_accuracy'] if k.startswith(kind)})
            mean=[g['metrics']['accuracy']['mean']]; lo=[g['metrics']['accuracy']['ci95'][0]]; hi=[g['metrics']['accuracy']['ci95'][1]]
            for s in sev:
                c=g['corruption_accuracy'][f'{kind}_{s:g}']; mean.append(c['mean']); lo.append(c['ci95'][0]); hi.append(c['ci95'][1])
            x=[0]+sev; ax.fill_between(x,lo,hi,color=COLOR[g['model']],alpha=.18,lw=0); ax.plot(x,mean,marker=MARK[g['model']],color=COLOR[g['model']],ms=4)
            data.append({'model':g['model'],'kind':kind,'severity':x,'accuracy':mean,'ci_low':lo,'ci_high':hi})
        ax.set_title(kind.replace('_',' '),loc='left',fontsize=8); ax.set_xlabel('Severity (unit scale of the input)')
    axes[0].set_ylabel('Test accuracy'); legend(axes[0],'lower left'); save('robustness',fig,data,ids,'accuracy under corruption')

    # F6 training history: validation per epoch and final test
    fig,ax=plt.subplots(figsize=(4.2,2.8)); data=[]
    for g in groups:
        h=g['validation_history']; ep=[e['epoch'] for e in h]; m=np.array([e['validation_accuracy']['mean'] for e in h]); s=np.array([e['validation_accuracy']['std'] or 0 for e in h])
        ax.fill_between(ep,m-s,m+s,color=COLOR[g['model']],alpha=.18,lw=0); ax.plot(ep,m,marker=MARK[g['model']],color=COLOR[g['model']],ms=4)
        t=g['metrics']['accuracy']['mean']; ax.plot([ep[-1]+.25],[t],marker=MARK[g['model']],mfc='white',color=COLOR[g['model']],ms=5,lw=0)
        data.append({'model':g['model'],'epoch':ep,'validation_accuracy_mean':m.tolist(),'validation_accuracy_sd':s.tolist(),'final_test_accuracy':t})
    ax.set(xlabel='Epoch',ylabel='Accuracy',xticks=list(range(1,len(ep)+1))); legend(ax,'lower right')
    ax.set_title('Filled: validation (mean +/- SD over seeds). Open marker: final test accuracy.',loc='left',fontsize=7,color=INK2); save('training_history',fig,data,ids,'validation vs test')

    # F7 weight sensitivity (all complete models; flagged confounded when inputs differ)
    allsens=[s for s in sens if s['protocol'][0]=='cross_input_all_models'] or [s for s in sens if not s['confounded_cross_input']]
    if allsens:
        S=allsens[0]['analysis']; models=[m for m in MODEL_ORDER if m in S[0]['scores']]; named=[r for r in S if not r['scheme'].startswith('perturbation_')]; pert=[r for r in S if r['scheme'].startswith('perturbation_')]
        fig,(a1,a2)=plt.subplots(1,2,figsize=(7,2.7),gridspec_kw={'width_ratios':[1.1,1]}); data={'confounded_cross_input':allsens[0]['confounded_cross_input'],'named':[],'rank_counts':{}}
        for j,r in enumerate(named):
            for m in models:
                a1.plot([r['scores'][m]],[j],MARK[m],color=COLOR[m],ms=5)
            data['named'].append({'scheme':r['scheme'],'scores':r['scores'],'ranks':r['ranks']})
        a1.set_yticks(range(len(named)),[r['scheme'].replace('_',' ') for r in named]); a1.invert_yaxis(); a1.set_xlabel('Composite Trust Score'); a1.grid(axis='y',visible=False)
        k=len(models); width=.8/k
        for i,m in enumerate(models):
            counts=[sum(1 for r in pert if round(r['ranks'][m])==rk) for rk in range(1,k+1)]; data['rank_counts'][m]=counts
            a2.bar(np.arange(1,k+1)+(i-(k-1)/2)*width,counts,width*.9,color=COLOR[m])
        a2.set(xlabel='Rank (1 = highest composite)',ylabel='Perturbed weight draws (of 50)',xticks=range(1,k+1)); a2.grid(axis='x',visible=False); a2.set_ylim(0,len(pert)*1.45); legend(a2,'upper center')
        save('sensitivity',fig,data,ids,'weight sensitivity')

    # F8 paired differences
    pairs=[p for p in paired['pairs'] if 'accuracy_item_bootstrap' in p]
    if pairs:
        fig,axes=plt.subplots(1,2,figsize=(7,2.3+.12*len(pairs)),sharey=True); data=[]
        for ax,(label,get) in zip(axes,(('Accuracy difference (test-item bootstrap)',lambda p:p['accuracy_item_bootstrap']),('Composite difference (seed-level t-interval)',lambda p:p['seed_level_differences']['composite_trust']))):
            for i,p in enumerate(pairs):
                d=get(p); ci=d['ci']; mean=d['mean']; col=INK2 if p['cross_input'] else INK
                ax.plot(ci,[i,i],color=col,lw=2.5,solid_capstyle='butt'); ax.plot([mean],[i],'o',color=col,ms=4)
            ax.axvline(0,color=INK2,lw=.8); ax.set_title(label,loc='left',fontsize=7.5); ax.set_xlabel('A - B')
        axes[0].set_yticks(range(len(pairs)),[f"{SHORT[p['a']]} - {SHORT[p['b']]}"+(' *' if p['cross_input'] else '') for p in pairs]); axes[0].invert_yaxis()
        for p in pairs: data.append({'a':p['a'],'b':p['b'],'cross_input':p['cross_input'],'accuracy_bootstrap':p['accuracy_item_bootstrap'],'composite_seed':p['seed_level_differences']['composite_trust']})
        fig.text(.01,-.02,'* different input representation (confounded). Bonferroni-adjusted level for accuracy; unadjusted 95% for seed-level; exploratory.',fontsize=6.5,color=INK2)
        fig.tight_layout(); save('paired_differences',fig,data,ids,'pairwise differences')

    # F9 confusion matrices (one explicitly identified seed)
    fig,axes=plt.subplots(1,len(groups),figsize=(2.6*len(groups),2.9)); axes=np.atleast_1d(axes); data=[]
    for ax,g in zip(axes,groups):
        r=sorted(runs[g['model']],key=lambda r:r['seed'])[0]; cm=np.asarray(r['metrics']['confusion_matrix']); nm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        ax.imshow(nm,vmin=0,vmax=1,cmap='Blues'); ax.grid(False); labels=list(r['class_to_idx']); short=[l[:4] for l in labels]
        ax.set_xticks(range(len(labels)),short,rotation=90,fontsize=6); ax.set_yticks(range(len(labels)),short,fontsize=6); ax.set_title(f"{SHORT[g['model']]}, seed {r['seed']}",fontsize=7.5,loc='left')
        data.append({'model':g['model'],'run_id':r['run_id'],'seed':r['seed'],'counts':cm.tolist(),'labels':labels})
    axes[0].set_ylabel('True class'); fig.supxlabel('Predicted class',fontsize=7.5); fig.tight_layout(); save('confusion',fig,data,[d['run_id'] for d in data],'row-normalized confusion, one seed')

    write_tables(groups,paired,sens,tables,source,source_sha)
    write_json(out/'manifest.json',{'files':files,'source_sha256':source_sha,'run_ids':ids,'models':names})
    print('Generated',len(files),'figures for',names)

def write_tables(groups,paired,sens,tables,source,source_sha):
    rows=[]
    for g in groups:
        for metric in (*PILLARS,'composite_trust'):
            rows.append({'model':g['model'],'metric':metric,'purpose':g['purpose'],'cohort':g['cohort'],'split':'test','run_ids':g['run_ids'],**g['metrics'][metric]})
    write_json(tables/'table1.json',rows)
    def longtable(rowlist,header,label,caption,cols):
        L=[r'\begin{longtable}{'+cols+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule '+header+r' \\',r'\midrule\endfirsthead',r'\toprule '+header+r' \\',r'\midrule\endhead']
        L+=rowlist+[r'\bottomrule',r'\end{longtable}']; return L
    # Table 1: components with uncertainty
    body=[]
    for m in (*PILLARS,'composite_trust'):
        cells=[]
        for g in groups:
            s=g['metrics'][m]; cells.append(f"{s['mean']:.3f} $\\pm$ {s['std']:.3f}" if s['std'] is not None else f"{s['mean']:.3f}")
        body.append(tex(LABELS[m])+' & '+' & '.join(cells)+r' \\')
    ncols='l'+'c'*len(groups)
    lines=longtable(body,'Component & '+' & '.join(tex(SHORT[g['model']]) for g in groups),'tab:trust',
        f"Six normalized components and Composite Trust Score on the test split (mean $\\pm$ sample SD over $n={groups[0]['metrics']['accuracy']['n']}$ independent seeds; five epochs). 95\\% intervals are in Table~\\ref{{tab:ci}}.",ncols)
    (tables/'table1.tex').write_text('\n'.join(lines)+'\n')
    body=[]
    for g in groups:
        for m in (*PILLARS,'composite_trust'):
            s=g['metrics'][m]; body.append(f"{tex(SHORT[g['model']])} & {tex(LABELS[m])} & {number(s['mean'])} & {number(s['std'])} & {ci_text(s['ci95'])} & {s['n']}/{s['missing_runs']} "+r'\\')
    (tables/'table_ci.tex').write_text('\n'.join(longtable(body,r'Model & Component & Mean & SD & 95\% CI & $n$/missing','tab:ci',r'Seed-level mean, sample SD and 95\% Student-$t$ interval for every component. NC = not computed.','llrrlr'))+'\n')
    # Table 2: raw metrics
    spec=[('accuracy','Accuracy'),('macro_f1','Macro-F1'),('ece','ECE (15 bins)'),('nll','NLL'),('brier','Brier'),('normalized_entropy','Norm. entropy'),('error_detection_auroc','Error-detection AUROC'),
          ('stability_ssim','Stability (SSIM)'),('deletion_auc','Deletion AUC, salient'),('deletion_auc_random','Deletion AUC, random'),('deletion_auc_least','Deletion AUC, least salient'),('aopc_salient','AOPC, salient'),('silhouette','Silhouette'),('accuracy_retention','Accuracy retention')]
    body=[]
    for k,lab in spec:
        cells=[]
        for g in groups:
            s=g['metrics'][k]; cells.append(r'\NC{}' if s['mean'] is None else (f"{s['mean']:.4f} $\\pm$ {s['std']:.4f}" if s['std'] is not None else f"{s['mean']:.4f}"))
        body.append(tex(lab)+' & '+' & '.join(cells)+r' \\')
    (tables/'table_metrics.tex').write_text('\n'.join(longtable(body,'Metric & '+' & '.join(tex(SHORT[g['model']]) for g in groups),'tab:metrics','Underlying metrics (mean $\\pm$ SD over seeds). Direction: higher is better except ECE, NLL, Brier and deletion AUC of the salient order; entropy is descriptive.',ncols))+'\n')
    # Table 3: corruptions
    conds=list(groups[0]['corruption_accuracy']); body=[]
    for c in conds:
        cells=[f"{g['corruption_accuracy'][c]['mean']:.3f}" for g in groups]; body.append(tex(c.replace('_',' '))+' & '+' & '.join(cells)+r' \\')
    body.append('clean & '+' & '.join(f"{g['metrics']['accuracy']['mean']:.3f}" for g in groups)+r' \\')
    (tables/'table_corruption.tex').write_text('\n'.join(longtable(body,'Condition & '+' & '.join(tex(SHORT[g['model']]) for g in groups),'tab:corruption','Mean test accuracy under synthetic corruptions (severity in the unit scale of each input representation, so Prithvi and RGB severities are not equivalent).',ncols))+'\n')
    # Table 4: paired
    body=[]
    for p in paired['pairs']:
        if 'accuracy_item_bootstrap' not in p: continue
        b=p['accuracy_item_bootstrap']; c=p['seed_level_differences']['composite_trust']
        body.append(f"{tex(SHORT[p['a']])} $-$ {tex(SHORT[p['b']])}{' *' if p['cross_input'] else ''} & {b['mean']:+.4f} & {ci_text(b['ci'])} & {c['mean']:+.4f} & {ci_text(c['ci'])} "+r'\\')
    (tables/'table_paired.tex').write_text('\n'.join(longtable(body,r'A $-$ B & $\Delta$acc & Bonf. CI (items) & $\Delta T$ & 95\% CI (seeds)','tab:paired','Exploratory paired differences. Accuracy: percentile bootstrap over test images (Bonferroni-adjusted level, seed-averaged correctness). Composite: Student-$t$ over paired seeds. * different input representation.','lrlrl'))+'\n')
    write_json(tables/'table1.provenance.json',{'source':source,'source_sha256':source_sha,'json_sha256':file_hash(tables/'table1.json'),'tex_sha256':file_hash(tables/'table1.tex'),
        'tex_files':{p.name:file_hash(p) for p in sorted(tables.glob('table_*.tex'))}})

if __name__=='__main__': generate()
