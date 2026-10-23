"""Scientific metrics with explicit validation and no fabricated fallback values."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import t, rankdata, spearmanr
from sklearn.metrics import confusion_matrix, f1_score, silhouette_score, roc_auc_score

PILLARS = ('performance', 'explanation', 'calibration', 'robustness', 'uncertainty', 'representation')

def probabilities(p, y):
    p, y = np.asarray(p, dtype=float), np.asarray(y)
    if p.ndim != 2 or len(p) == 0 or p.shape[1] < 2 or y.shape != (len(p),):
        raise ValueError('Expected nonempty N x C probabilities and N labels, C >= 2')
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid probabilities')
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError('Invalid class labels')
    return p, y

def calibration_bins(p, y, bins=15):
    p, y = probabilities(p, y)
    if not isinstance(bins, int) or bins < 1: raise ValueError('bins must be positive integer')
    confidence, correct = p.max(1), p.argmax(1) == y
    indices = np.minimum((confidence * bins).astype(int), bins - 1)
    rows=[]
    for b in range(bins):
        mask=indices == b; n=int(mask.sum())
        rows.append({'lower':b/bins, 'upper':(b+1)/bins, 'n':n,
                     'confidence':float(confidence[mask].mean()) if n else None,
                     'accuracy':float(correct[mask].mean()) if n else None})
    return rows

def prediction_metrics(p, y, bins=15):
    p,y=probabilities(p,y); pred=p.argmax(1); c=p.shape[1]
    reliability=calibration_bins(p,y,bins)
    entropy=-(p*np.log(np.clip(p,1e-15,1))).sum(1)
    errors=(pred != y).astype(int)
    return {'accuracy':float((pred==y).mean()),
            'macro_f1':float(f1_score(y,pred,labels=np.arange(c),average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,labels=np.arange(c),average='weighted',zero_division=0)),
            'per_class_f1':f1_score(y,pred,labels=np.arange(c),average=None,zero_division=0).tolist(),
            'ece':float(sum(r['n']/len(y)*abs(r['accuracy']-r['confidence']) for r in reliability if r['n'])),
            'nll':float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean()),
            'brier':float(((p-np.eye(c)[y])**2).sum(1).mean()),
            'normalized_entropy':float(entropy.mean()/np.log(c)),
            'error_detection_auroc':float(roc_auc_score(errors,entropy)) if len(np.unique(errors))==2 else None,
            'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(c)).tolist(),
            'reliability_bins':reliability}

def stability(clean, perturbed):
    from skimage.metrics import structural_similarity
    a,b=np.asarray(clean,dtype=float),np.asarray(perturbed,dtype=float)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape)<3: raise ValueError('SSIM requires matching 2D maps >= 3')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(),b.min())<0 or max(a.max(),b.max())>1:
        raise ValueError('Attributions must be normalized to [0,1]')
    if np.ptp(a)==0 or np.ptp(b)==0: raise ValueError('Constant attribution map has no spatial explanation')
    win=min(7,min(a.shape)); win-=1-win%2
    return float(structural_similarity(a,b,data_range=1.,win_size=win))

def representation_quality(features, labels):
    x,y=np.asarray(features,dtype=float),np.asarray(labels)
    if x.ndim!=2 or len(y)!=len(x) or not np.isfinite(x).all(): raise ValueError('Invalid feature matrix')
    if not 1 < len(np.unique(y)) < len(y): raise ValueError('Silhouette needs 2..N-1 classes')
    if np.max(np.ptp(x,axis=0))==0: raise ValueError('Constant features are not informative')
    # L2 normalization removes overall magnitude; Euclidean silhouette is still dimension dependent.
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    if (norm==0).any(): raise ValueError('Zero embedding')
    s=float(silhouette_score(x/norm,y,metric='euclidean'))
    return {'silhouette':s,'representation':(s+1)/2}

def weights_valid(weights):
    if set(weights)!=set(PILLARS): raise ValueError('Exactly six named weights required')
    vals=np.asarray([weights[k] for k in PILLARS],float)
    if not np.isfinite(vals).all() or (vals<0).any() or not np.isclose(vals.sum(),1,atol=1e-10):
        raise ValueError('Weights must be finite, nonnegative and sum to one')
    return vals

def composite(scores, weights):
    w=weights_valid(weights)
    if set(scores)!=set(PILLARS) or any(scores[k] is None for k in PILLARS): raise ValueError('Missing pillar: no composite score')
    s=np.asarray([scores[k] for k in PILLARS],float)
    if not np.isfinite(s).all() or (s<0).any() or (s>1).any(): raise ValueError('All six pillars must be finite and in [0,1]')
    return float(s@w)

def pillar_scores(metrics, corruption_accuracies, representation):
    if len(corruption_accuracies)==0: raise ValueError('Robustness not computed')
    values=np.asarray(corruption_accuracies,float)
    if not np.isfinite(values).all() or (values<0).any() or (values>1).any(): raise ValueError('Invalid corruption accuracy')
    scores=dict(performance=metrics['macro_f1'], explanation=1-metrics['deletion_auc'],
                calibration=1-metrics['ece'],robustness=float(values.mean()),
                uncertainty=1-metrics['brier']/2,representation=representation)
    composite(scores,{k:1/6 for k in PILLARS})
    return scores

def mean_ci(values, expected_runs=None):
    x=np.asarray(values,float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('Finite independent run values required')
    n=len(x)
    if expected_runs is not None and (expected_runs<0 or n>expected_runs): raise ValueError('Invalid expected runs')
    out={'n':n,'missing_runs':None if expected_runs is None else expected_runs-n,
         'mean':float(x.mean()) if n else None,'std':None,'ci95':None,'method':'Student-t; independent seeds; ddof=1'}
    if n>=2:
        sd=float(x.std(ddof=1)); half=float(t.ppf(.975,n-1)*sd/math.sqrt(n))
        out.update(std=sd,ci95=[out['mean']-half,out['mean']+half])
    return out

def sensitivity(models, schemes, perturbations=50, seed=445):
    if not models: raise ValueError('No complete models')
    names=sorted(models); rng=np.random.default_rng(seed)
    base=schemes['default']; weights_valid(base)
    all_schemes=dict(schemes)
    for i in range(perturbations):
        w=np.array([base[k] for k in PILLARS])*rng.uniform(.8,1.2,len(PILLARS)); w/=w.sum()
        all_schemes[f'perturbation_{i:03d}']=dict(zip(PILLARS,w.tolist()))
    reference=rankdata([-composite(models[m],base) for m in names],method='average')
    result=[]
    for label,w in all_schemes.items():
        scores=[composite(models[m],w) for m in names]
        ranks=rankdata(-np.array(scores),method='average')
        rho=float(spearmanr(reference,ranks).statistic) if len(names)>1 and np.ptp(reference)>0 and np.ptp(ranks)>0 else None
        result.append({'scheme':label,'weights':w,'scores':dict(zip(names,scores)),
                       'ranks':dict(zip(names,ranks.tolist())),'spearman_vs_default':rho,
                       'preferred_models':[m for m,v in zip(names,scores) if np.isclose(v,max(scores),atol=1e-12)]})
    return result
