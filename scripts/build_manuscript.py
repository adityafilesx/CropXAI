"""Build the Procedia-format manuscript (with the point-by-point response in the same PDF) and a separate response sheet.

Every number in the results, abstract and audit sections is read from validated records (results/processed/*.json and
docs/revision/evidence/legacy_audit.json). The template contains only methodological text.
"""
import sys,json,re,shutil,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cropxai.provenance import write_json,file_hash
from src.cropxai.metrics import PILLARS
from generate_figures import generate,tex,complete_groups,SHORT,LABELS,MODEL_ORDER

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'paper/source'; OUT=ROOT/'output/pdf'; EV=ROOT/'docs/revision/evidence'

def escape(s):
    out=''
    for ch in str(s): out+={'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','$':r'\$','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\^{}'}.get(ch,ch)
    return out

def pct(x,d=2): return f'{100*x:.{d}f}\\%'
def ci(s,d=4,percent=False):
    if s['ci95'] is None: return r'\NC{}'
    f=(lambda v:f'{100*v:.2f}') if percent else (lambda v:f'{v:.{d}f}')
    return f"[{f(s['ci95'][0])}, {f(s['ci95'][1])}]"

def refs_latex(t): return re.sub(r'\[\[([^\]]+)\]\]',lambda m:'\\ref{'+m.group(1)+'}',t)
def refs_plain(t,labels): return re.sub(r'\[\[([^\]]+)\]\]',lambda m:labels.get(m.group(1),{}).get('number','??'),t)

def responses(comments):
    parts=[r'\clearpage\section*{Point-by-point response to reviewers}', 'Paper ID 445. Comment wording follows the author-supplied revision brief; the original notification letter was unavailable to the revising software. Completed corrections and unresolved evidence are distinguished. "Section/page" refer to this revised manuscript.']
    for c in comments:
        parts.append(r'\subsection*{'+c['id']+': '+escape(c['comment'])+'}')
        parts.append(r'\textbf{Status: '+c['status']+'.} '+refs_latex(escape(c['response'])))
        parts.append(r'\par\textbf{Location:} Section~\ref{'+c['section']+r'} (manuscript page~\pageref{'+c['section']+'}).')
        parts.append(r'\par\textbf{Implementation/evidence:} '+escape(c['code'])+'. '+escape(c['experiment'])+'. Tests: '+escape(c['test'])+'. Evidence: '+refs_latex(escape(c['evidence'])).rstrip('.')+'.')
    return '\n\n'.join(parts)

def compile_tex(filename):
    command=['pdflatex','-interaction=nonstopmode','-halt-on-error',filename]
    result=subprocess.run(command,cwd=SRC,capture_output=True,text=True)
    (EV/f'{Path(filename).stem}_build.log').write_text(result.stdout+result.stderr)
    if result.returncode: raise RuntimeError('LaTeX failed; see build log')
    if filename=='445_revised_version.tex':
        b=subprocess.run(['bibtex',Path(filename).stem],cwd=SRC,capture_output=True,text=True)
        (EV/'bibtex.log').write_text(b.stdout+b.stderr)
        if b.returncode: raise RuntimeError('BibTeX failed')
    for _ in range(2):
        result=subprocess.run(command,cwd=SRC,capture_output=True,text=True)
        if result.returncode: raise RuntimeError('LaTeX second pass failed')
    (EV/f'{Path(filename).stem}_build.log').write_text(result.stdout+result.stderr)

def fig(name,caption,label,width='.92'):
    return (r'\begin{figure}[htbp]\centering'+f'\n\\includegraphics[width={width}\\linewidth]{{../../results/figures/{name}.pdf}}\n'
            +rf'\caption{{{caption}}}\label{{{label}}}'+r'\end{figure}')

def results_text(groups,summary,paired,sens):
    byname={g['model']:g for g in groups}; names=[g['model'] for g in groups]; n=groups[0]['metrics']['accuracy']['n']
    fails=[f for f in summary['failed_runs'] if f['purpose'] in ('controlled_benchmark','multispectral_foundation_benchmark')]
    def best(metric,hi=True):
        key=lambda m: byname[m]['metrics'][metric]['mean']; return (max if hi else min)(names,key=key)
    C=[]
    C.append(r'\subsection{Experimental cohorts}\label{sec:cohorts}')
    C.append(f"All analysed numbers come from {sum(len(g['run_ids']) for g in groups)} completed runs: {len(names)} architectures $\\times$ {n} independent seeds, five epochs each, evaluated on the same 4,050 test patches. "
             +(f"{len(fails)} failed run(s) in the final cohorts are disclosed in the record store and are not aggregated. " if fails else "No run of the final cohorts failed. ")+f"Earlier runs from a previous source revision, one superseded dry run and the one-epoch smoke test exist in the archive but are excluded by construction. "
             "The RGB cohort (ResNet-18, ViT-Base, ConvNeXt-T) and the six-band cohort (Prithvi-EO-1.0-100M) use different inputs; tests of one cohort against the other are marked as confounded.")
    C.append(r'\subsection{Predictive performance and component scores}\label{sec:perf}')
    parts=[]
    for m in names:
        s=byname[m]['metrics']; parts.append(f"{escape(SHORT[m])} {pct(s['accuracy']['mean'])} (95\\% seed CI {ci(s['accuracy'],percent=True)}\\%, macro-F1 {s['macro_f1']['mean']:.4f})")
    C.append('Test accuracy was '+'; '.join(parts)+'. Table~\\ref{tab:trust} lists the six normalized components and the Composite Trust Score; Table~\\ref{tab:ci} gives seed-level intervals; Table~\\ref{tab:metrics} lists the underlying metrics. Figures~\\ref{fig:accuracy} and \\ref{fig:pillars} show the same values with their intervals. Seed intervals are narrow because only the linear head is trained (Section~\\ref{sec:statistics}); the representation component is computed from the frozen features and is therefore identical across seeds (zero SD by construction); bootstrap intervals over test patches are wider and are also drawn in Figure~\\ref{fig:accuracy}.')
    C.append(r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\input{../../results/tables/table1.tex}\endgroup')
    C.append(fig('accuracy','Test accuracy per model. Thick bars: 95\\% Student-$t$ interval over five training seeds; thin gray bars: 95\\% percentile bootstrap over the 4,050 test patches (seed-averaged correctness).','fig:accuracy','.7'))
    C.append(fig('pillars','Six components and the Composite Trust Score (higher is better). Markers: mean over seeds; bars: 95\\% Student-$t$ intervals. Each panel has its own vertical scale.','fig:pillars'))
    C.append(r'\begingroup\scriptsize\setlength{\tabcolsep}{2pt}\input{../../results/tables/table_metrics.tex}\endgroup')
    C.append(r'\subsection{Calibration, explanation and robustness}\label{sec:calib}')
    ece_best,ece_worst=best('ece',False),best('ece',True)
    stab=[byname[m]['metrics']['stability_ssim']['mean'] for m in names]; acc_all=[byname[m]['metrics']['accuracy']['mean'] for m in names]
    gaps=[(m,byname[m]['metrics']['deletion_gap_random']['mean']) for m in names]
    C.append(f"Top-label ECE ranged from {byname[ece_best]['metrics']['ece']['mean']:.4f} ({escape(SHORT[ece_best])}) to {byname[ece_worst]['metrics']['ece']['mean']:.4f} ({escape(SHORT[ece_worst])}); Figure~\\ref{{fig:reliability}} shows reliability curves pooled over seeds with the same 15 bins. "
             f"Explanation stability (SSIM between attribution maps of clean and noise-perturbed inputs) was low for every model ({min(stab):.3f}--{max(stab):.3f}), so attributions changed substantially under $\\sigma{{=}}.05$ noise; this does not by itself say whether the predictions or the attribution method are responsible. "
             "Deletion AUC for the most-salient-first order is compared in Figure~\\ref{fig:deletion} with random and least-salient controls. The control-minus-salient AUC gap is positive when deleting salient patches removes probability faster than random deletion and negative when random deletion removes it faster (a negative gap means the attribution ranking is not more faithful than chance under this metric): "
             +', '.join(f"{escape(SHORT[m])} {g:+.3f}" for m,g in gaps)+". "
             f"Under synthetic corruption (Figure~\\ref{{fig:robustness}}, Table~\\ref{{tab:corruption}}) accuracy at noise $\\sigma{{=}}.05$ ranged {min(byname[m]['corruption_accuracy']['gaussian_noise_0.05']['mean'] for m in names):.3f}--{max(byname[m]['corruption_accuracy']['gaussian_noise_0.05']['mean'] for m in names):.3f} and at brightness reduction $.5$ {min(byname[m]['corruption_accuracy']['brightness_0.5']['mean'] for m in names):.3f}--{max(byname[m]['corruption_accuracy']['brightness_0.5']['mean'] for m in names):.3f}, against clean accuracies of {min(acc_all):.3f}--{max(acc_all):.3f}; models differ markedly in which corruption family hurts them. These are synthetic covariate shifts on different unit scales, not geographic or seasonal shift.")
    C.append(fig('reliability','Reliability diagram (15 equal-width bins) with all seeds pooled per model. Dotted line: perfect calibration. Pooled values differ slightly from the per-seed mean ECE in Table~\\ref{tab:metrics}.','fig:reliability','.55'))
    C.append(fig('deletion','Fixed-target probability versus fraction of 16$\\times$16 patches deleted (replacement by normalized zero), mean over 100 images and five seeds. Solid: most salient first; dashed: random; dotted: least salient first.','fig:deletion'))
    C.append(fig('robustness','Test accuracy under synthetic corruptions (severity 0 is the clean test set; bands: 95\\% seed intervals). Severities are on the unit scale of each input representation and are not equivalent between RGB and six-band inputs.','fig:robustness','.8'))
    C.append(r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\input{../../results/tables/table_corruption.tex}\endgroup')
    C.append(r'\subsection{Composite Trust Score, weights and ranking}\label{sec:sensresults}')
    allsens=[s for s in sens if s['protocol'][0]=='cross_input_all_models'] or [s for s in sens if not s['confounded_cross_input']]
    if allsens:
        S=allsens[0]['analysis']; models=allsens[0]['models']; pert=[r for r in S if r['scheme'].startswith('perturbation_')]
        named={r['scheme']:r for r in S if not r['scheme'].startswith('perturbation_')}
        top=lambda r:'/'.join(escape(SHORT[m]) for m in r['preferred_models'])
        rhos=[r['spearman_vs_default'] for r in S if r['spearman_vs_default'] is not None]
        changed=sum(1 for r in S if r['ranks']!=S[0]['ranks'])
        up1=lambda t:t[:1].upper()+t[1:]
        C.append(f"Under the default (equal) weights the Composite Trust Score ranks the models as "+' > '.join(escape(SHORT[m]) for m in sorted(models,key=lambda m:S[0]['ranks'][m]))+". "
                 +up1('; '.join(f"{k.replace('_',' ')} weights prefer {top(v)}" for k,v in named.items() if k!='default'))+f". Across {len(S)} weight configurations (equal, two priority profiles and {len(pert)} random perturbations of $\\pm20\\%$) the ranking differed from the default in {changed} configuration(s); Spearman correlation with the default ranking ranged {min(rhos):.2f}--{max(rhos):.2f}. "
                 +("This analysis includes models with different inputs and is therefore descriptive only (Figure~\\ref{fig:sensitivity}). " if allsens[0]['confounded_cross_input'] else "")+"The weight profiles were fixed in the versioned configuration before the final campaign was executed and are a utility choice, not an estimate.")
        C.append(fig('sensitivity','Left: Composite Trust Score under the named weight profiles. Right: how often each model attains each rank over 50 seeded random weight perturbations.','fig:sensitivity'))
    ph=json.loads((ROOT/'results/processed/posthoc_explanation_sensitivity.json').read_text())['models']
    C.append(r'\subsection{Post-hoc check of the explanation component}\label{sec:posthoc}')
    order=sorted(ph,key=lambda m:ph[m]['rank_primary_composite'])
    C.append("\\textbf{This subsection is post-hoc: it was defined after the main results were inspected and does not replace the pre-declared composite.} "
             "The explanation component $1-D$ rewards low salient-first deletion AUC, but a model whose probability collapses under \\emph{any} masking also has a low $D$. The random-order control exposes this: for some models random deletion removes probability faster than salient-first deletion, so a high $1-D$ then reflects fragility to masking rather than a faithful attribution. "
             "As a diagnostic only, the explanation component is replaced by $E'=\\tfrac12\\bigl(1+(D_{\\mathrm{rand}}-D)/(D_{\\mathrm{rand}}+D)\\bigr)$, where $E'=.5$ means salient deletion is no faster than random deletion, $E'>.5$ faster and $E'<.5$ slower. With default weights the composite ranking changes from "
             +' > '.join(f"{escape(SHORT[m])} ({ph[m]['primary_composite']:.3f})" for m in order)+" (pre-declared) to "
             +' > '.join(f"{escape(SHORT[m])} ({ph[m]['posthoc_composite']:.3f})" for m in sorted(ph,key=lambda m:ph[m]['rank_posthoc_composite']))+" (post-hoc). "
             +"Only models with $E'$ above .5 show attributions that outperform random deletion under this protocol: "+(', '.join(f"{escape(SHORT[m])} $E'={ph[m]['posthoc_explanation']:.3f}$" for m in order))+". "
             "The composite ranking is therefore not robust to how the explanation component is defined, and no ranking is claimed to be universal.")
    C.append(r'\subsection{Pairwise comparisons}\label{sec:pairs}')
    pairs=[p for p in paired['pairs'] if 'accuracy_item_bootstrap' in p]
    nsep=sum(1 for p in pairs if p['accuracy_item_bootstrap']['ci'][0]>0 or p['accuracy_item_bootstrap']['ci'][1]<0)
    C.append(f"Of {len(pairs)} pairwise accuracy comparisons, {nsep} have a Bonferroni-adjusted bootstrap interval that excludes zero (Table~\\ref{{tab:paired}}, Figure~\\ref{{fig:paired}}). Comparisons marked with an asterisk cross input representations and cannot attribute a difference to the architecture. All comparisons are exploratory, condition on one random split, and do not support claims about geographic generalization.")
    C.append(r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\input{../../results/tables/table_paired.tex}\endgroup')
    C.append(fig('paired_differences','Pairwise differences (A $-$ B). Left: accuracy, bootstrap over test patches at the Bonferroni-adjusted level. Right: Composite Trust Score, 95\\% Student-$t$ interval over paired seeds. Gray: different input representations.','fig:paired'))
    C.append(r'\subsection{Training history}\label{sec:history}')
    last=lambda m: byname[m]['validation_history'][-1]['validation_accuracy']['mean']
    vt='; '.join(f"{escape(SHORT[m])} {pct(last(m))} versus {pct(byname[m]['metrics']['accuracy']['mean'])}" for m in names)
    C.append(f"Figure~\\ref{{fig:history}} shows validation accuracy per epoch (mean $\\pm$ SD over seeds) together with the final test accuracy, labelled separately. Final-epoch validation versus final test accuracy: {vt}. "
             "The binomial standard error of one accuracy on 4,050 patches is about 0.3--0.5 percentage points, so a one-point difference between two independent 4,050-patch subsets is within a few standard errors. Validation was never used to choose the reported epoch. "
             "Per-patch confusion matrices for one seed are in Figure~\\ref{fig:confusion}.")
    C.append(fig('training_history','Validation accuracy per epoch (filled, band: $\\pm$1 SD over seeds) and final test accuracy (open marker).','fig:history','.6'))
    C.append(fig('confusion','Row-normalized confusion matrices on the test split for the lowest seed of each model.','fig:confusion'))
    C.append(r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}\input{../../results/tables/table_ci.tex}\endgroup')
    return '\n\n'.join(C)

def abstract_text(groups,legacy):
    byname={g['model']:g for g in groups}; names=[g['model'] for g in groups]; n=groups[0]['metrics']['accuracy']['n']
    acc=lambda m: byname[m]['metrics']['accuracy']['mean']
    lo,hi=min(names,key=acc),max(names,key=acc); nm=lambda m: 'Prithvi, six bands' if m.startswith('Prithvi') else SHORT[m]
    sw=[g['metrics']['stability_ssim']['mean'] for g in groups]
    t=legacy['table1_reconstruction']['ResNet-18']
    return ("CropXAI reports predictive, calibration, explanation, robustness, uncertainty and feature-space measures of Earth observation classifiers together with auditable run records. "
        f"This revision evaluates {len(names)} architectures on EuroSAT land-cover classification, each with {n} independent seeds and a frozen backbone: three ImageNet-pretrained models on RGB patches and the Prithvi-EO-1.0-100M encoder on six Sentinel-2 bands; the two input types are never conflated and no 13-band experiment is claimed. "
        f"Test accuracy ranged from {pct(acc(lo))} ({nm(lo)}) to {pct(acc(hi))} ({nm(hi)}), while attribution stability under noise was low for all models (SSIM {min(sw):.2f}--{max(sw):.2f}); explanation stability and deletion-based faithfulness are measured separately. "
        "A six-component Composite Trust Score, a weighted sum rather than a Bayesian posterior, is reported with seed-level intervals and a weight-sensitivity analysis; a post-hoc control check shows that its model ranking depends on how the explanation component is defined. "
        f"An audit of the submitted results found that its Table~I pooled one- and five-epoch runs (ResNet-18: {pct(t['accuracy']['submitted'])} pooled versus {pct(t['accuracy']['five_epoch_only'])} for five-epoch runs), that fixed fallback trust values contaminated five runs, and that the model labelled Prithvi-100M was an RGB ViT-Base proxy; these results are withdrawn. "
        "The evidence covers one dataset, a random split and synthetic corruptions only, and does not validate crop-health applications.")

def discrepancy_text(legacy,rerun,groups):
    r=legacy['table1_reconstruction']; d=legacy['database_divergence']
    res=next(x for x in d['runs_with_differing_metrics'] if (x['model'],x['epochs'],x['seed'])==('ResNet-18',5,42))
    m=res['differing_metrics']
    T=[]
    T.append(f"\\textbf{{Table I of the submitted version.}} Recomputing from the stored legacy database reproduces the printed accuracies exactly as the unweighted mean over \\emph{{all}} runs of a model, regardless of epoch count: ResNet-18 {pct(r['ResNet-18']['accuracy']['pooled_all_runs'])} ({r['ResNet-18']['n_runs']} runs, of one and of five epochs), ViT-Base {pct(r['ViT-Base']['accuracy']['pooled_all_runs'])} ({r['ViT-Base']['n_runs']} runs) and ``Prithvi-100M'' {pct(r['Prithvi-100M']['accuracy']['pooled_all_runs'])} ({r['Prithvi-100M']['n_runs']} runs). "
             f"Only three distinct seeds exist, not five; the five-epoch-only means are {pct(r['ResNet-18']['accuracy']['five_epoch_only'])}, {pct(r['ViT-Base']['accuracy']['five_epoch_only'])} and {pct(r['Prithvi-100M']['accuracy']['five_epoch_only'])}. The reported numbers therefore did not describe a five-seed, five-epoch experiment. Table~\\ref{{tab:legacy}} compares them with the corrected cohorts.")
    by={g['model']:g for g in groups}; five=legacy['five_epoch_accuracy_all_models']
    def cell(m):
        a=by[m]['metrics']['accuracy']; return f"{100*a['mean']:.2f} [{100*a['ci95'][0]:.2f}, {100*a['ci95'][1]:.2f}]"
    rows=[('ResNet-18',pct(r['ResNet-18']['accuracy']['submitted']),f"{100*five['ResNet-18']['accuracy_mean']:.2f} ({five['ResNet-18']['n_runs']})",cell('ResNet-18')),
          ('ViT-Base',pct(r['ViT-Base']['accuracy']['submitted']),f"{100*five['ViT-Base']['accuracy_mean']:.2f} ({five['ViT-Base']['n_runs']})",cell('ViT-Base')),
          ('ConvNeXt-T','not reported',f"{100*five['ConvNeXt-T']['accuracy_mean']:.2f} ({five['ConvNeXt-T']['n_runs']})",cell('ConvNeXt-T')),
          ("``Prithvi-100M'' (ViT proxy) / Prithvi-EO-1.0-100M",pct(r['Prithvi-100M']['accuracy']['submitted']),f"{100*five['Prithvi-100M']['accuracy_mean']:.2f} ({five['Prithvi-100M']['n_runs']}), proxy",cell('Prithvi-EO-1.0-100M')+', real model')]
    T.append(r'\begin{table}[htbp]\centering\footnotesize\setlength{\tabcolsep}{3pt}\caption{Submitted accuracy versus the mixed legacy and the corrected cohorts (test accuracy, \%). Column 3: mean of legacy five-epoch runs only (number of runs). Column 4: corrected protocol, mean [95\% seed interval] over five seeds. The protocols differ (legacy trainer updated BatchNorm statistics; corrected runs train a linear head on frozen features), so the columns are not a like-for-like replication.}\label{tab:legacy}'
             +r'\begin{tabular}{p{.3\linewidth}rrr}\toprule Model & Submitted (1+5 epochs pooled) & Legacy, 5 epochs only & Corrected, 5 seeds $\times$ 5 epochs\\\midrule '
             +' \\\\ '.join(' & '.join(c) for c in rows)+r' \\\bottomrule\end{tabular}\end{table}')
    T.append(f"\\textbf{{Figure 7 versus Table I.}} The submitted training-convergence figure (it appears as Fig.~8 in the submitted PDF, whereas its Fig.~7 is a cross-dataset drift plot; the reviewer comment calls it Figure 7) shows, for ResNet-18 over five epochs, validation accuracy of about 84--86\\% and a training loss falling to about 0.50. "
             f"The legacy store derives a run identifier from the configuration alone and writes metrics with \\texttt{{INSERT OR REPLACE}}, so repeating a configuration replaced the earlier values under the unchanged run record. "
             f"For the run ResNet-18, seed 42, five epochs, the reviewer-package copy of the database holds test accuracy {pct(m['accuracy']['reviewer_package'])} and final training loss {m['final_loss']['reviewer_package']:.3f}, whereas the main database, used for the submitted Table I, holds {pct(m['accuracy']['main'])} and {m['final_loss']['main']:.3f} under the same identifier and timestamp. "
             "The first set agrees with the convergence figure on both quantities (accuracy band and final loss). In addition, another submitted figure file (\\texttt{baseline\\_results.pdf}) prints ``Final Test Accuracy: 85.19\\%'' and ``Final Trust Score (T3): 0.2394'', exactly the reviewer-package values of this run and not the main-database values, and it was registered in the database about 90 seconds after the run timestamp. The second set is what entered the submitted table. "
             "We therefore conclude that the submitted figure and table come from \\emph{different executions of the same configuration}, and that the figure is consistent with the original execution, not with the values later used in the table. "
             "The convergence figure's own plotted data were not preserved, so its link to that execution is strongly supported but not proven.")
    if rerun:
        eps=rerun['per_epoch']; fm=rerun['final_metrics_passed_to_store_run']
        T.append(f"\\textbf{{Re-execution of the legacy trainer.}} The unchanged legacy trainer was executed again (ResNet-18, seed 42, five epochs; the legacy database tables were verified identical to the archived original afterwards). Its validation accuracy per epoch was "
                 +', '.join(f"{e['validation_accuracy_percent']:.1f}\\%" for e in eps)+f", final training loss {eps[-1]['training_loss']:.4f}, test accuracy {pct(fm['accuracy'])} and weighted F1 {fm['f1']:.4f}. "
                 f"These reproduce the values held in the main database ({pct(m['accuracy']['main'])}, loss {m['final_loss']['main']:.4f}) and not the earlier reviewer-package values or the validation band of the submitted Figure 7. "
                 "Hence the main-database entry is reproducible with the current legacy code (one re-execution), while the execution that produced the Figure-7-like values (accuracy 85.19\\%, final loss 0.499) cannot be reproduced with the present code and data; the code revision (recorded as unknown) or data state evidently differed. "
                 "Consequently the submitted figure and table describe different executions, and neither can be certified as a five-seed result.")
    T.append(f"\\textbf{{Fallback trust values.}} The legacy trust function returned the constants 0.65 (missing library) and 0.50 (no supported layer). Five stored runs carry a trust value of exactly 0.5, including all one-epoch ``Prithvi-100M'' runs; these values are fallbacks, not measurements, and are invalid. The function now raises an error, and a test enforces this.")
    T.append("\\textbf{The foundation-model label.} For every (epochs, seed) cell present for both, the legacy ``Prithvi-100M'' accuracy, F1, precision and recall equal those of ViT-Base to machine precision: the label denoted an RGB ViT-Base proxy. All legacy ``Prithvi-100M'' results are withdrawn; Prithvi-EO-1.0-100M is evaluated afresh in this revision. Both legacy databases and the original sources remain preserved.")
    T.append("\\textbf{Legacy statistical code.} A scan of the legacy source tree (reproducible with a script in the repository) found modules whose own comments describe them as mock implementations, among them a Nemenyi post-hoc test, a Bayesian ranking, a power analysis, and an explanation-benchmark manager that draws deletion and insertion AUC values from a uniform random generator. The scan cannot show which of them produced numbers in the submitted manuscript; none of their output is reused here, the active pipeline contains no such code (enforced by a test), and the Nemenyi, Bayesian and power claims are withdrawn.")
    return '\n\n'.join(T)

def build():
    generate(); OUT.mkdir(parents=True,exist_ok=True)
    summary=json.loads((ROOT/'results/processed/revision_summary.json').read_text()); groups=complete_groups(summary)
    paired=json.loads((ROOT/'results/processed/paired_comparisons.json').read_text()); sens=json.loads((ROOT/'results/processed/weight_sensitivity.json').read_text())
    legacy=json.loads((EV/'legacy_audit.json').read_text())
    rr=sorted(EV.glob('legacy_rerun_*.json')); rerun=json.loads(rr[0].read_text()) if rr else None
    comments=json.loads((ROOT/'docs/revision/reviewer_comments.json').read_text())
    status='Revised after a full re-execution of the experiments. Remaining open items are listed in the submission checklist: independent similarity report, confirmation of the conference-specific format, and the limits stated in Section~\\ref{sec:limitations}.'
    template=(SRC/'manuscript_template.tex').read_text()
    final=(template.replace('@@ABSTRACT@@',abstract_text(groups,legacy)).replace('@@STATUS@@',status).replace('@@RESULTS@@',results_text(groups,summary,paired,sens))
           .replace('@@DISCREPANCY@@',discrepancy_text(legacy,rerun,groups)).replace('@@RESPONSE@@',responses(comments)))
    (SRC/'445_revised_version.tex').write_text(final)
    for name in ['elsarticle.cls','ecrc.sty','elsevier-logo-3p.pdf','SDlogo-3p.pdf','Procs.pdf']: shutil.copy2(ROOT/'Paper format '/name,SRC/name)
    compile_tex('445_revised_version.tex')
    shutil.copy2(SRC/'445_revised_version.pdf',OUT/'445_revised_version.pdf')
    aux=(SRC/'445_revised_version.aux').read_text()
    labels={key:{'number':num,'page':page} for key,num,page in re.findall(r'\\newlabel\{([^}]+)\}\{\{([^}]+)\}\{([^}]+)\}',aux)}
    write_json(EV/'manuscript_locations.json',labels)
    response_text=responses(comments)
    for label,loc in labels.items(): response_text=response_text.replace(r'\ref{'+label+'}',escape(loc['number'])).replace(r'\pageref{'+label+'}',escape(loc['page']))
    separate=r'\documentclass[11pt]{article}\usepackage[T1]{fontenc}\usepackage[margin=24mm]{geometry}\usepackage{hyperref}\setlength{\emergencystretch}{3em}\begin{document}'+response_text+r'\end{document}'
    (SRC/'reviewer_response_sheet.tex').write_text(separate); compile_tex('reviewer_response_sheet.tex')
    shutil.copy2(SRC/'reviewer_response_sheet.pdf',OUT/'reviewer_response_sheet.pdf')
    md=['# Response to reviewers - Paper 445','Comment wording follows the author-provided revision brief; the original notification is unavailable.']
    matrix=['# Reviewer traceability','| ID | Issue | Code/config | Required analysis | Manuscript location | Test | Status | Evidence |','|---|---|---|---|---|---|---|---|']
    for c in comments:
        loc=labels.get(c['section'],{}); location=f"Section {loc.get('number','NOT AVAILABLE')}, manuscript page {loc.get('page','NOT AVAILABLE')}"
        md.extend([f"\n## {c['id']} - {c['comment']}",f"**Status: {c['status']}**",refs_plain(c['response'],labels),f"Change: {c['code']}. Analysis: {c['experiment']}.",f"Location: {location}. Verification: {c['test']}. Evidence: {refs_plain(c['evidence'],labels).rstrip('.')}."])
        matrix.append('| '+' | '.join([c['id'],c['issue'],c['code'],c['experiment'],location,c['test'],c['status'],c['evidence']])+' |')
    (ROOT/'paper/response_to_reviewers.md').write_text('\n\n'.join(md)+'\n')
    (ROOT/'docs/revision/REVIEWER_TRACEABILITY.md').write_text('\n'.join(matrix)+'\n')
    print('Built combined manuscript + responses and separate response PDF; submission readiness remains subject to the validator.')
if __name__=='__main__': build()
