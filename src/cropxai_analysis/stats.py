"""Statistics for the revision: seed-level paired intervals and test-item bootstrap.

Statistical units are stated explicitly:
  * seed-level   - one value per independent training seed (n = number of seeds); captures optimizer/initialization noise only.
  * test-item    - one value per test image averaged over seeds; captures test-set sampling noise for this fixed split.
Neither captures geographic or temporal population variance. Comparisons are exploratory.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import t

def t_interval(values, level=.95):
    x=np.asarray(values,float)
    if x.ndim!=1 or not np.isfinite(x).all(): raise ValueError('Finite 1-D values required')
    n=len(x)
    if n<2: return {'n':n,'mean':float(x.mean()) if n else None,'sd':None,'ci':None,'level':level}
    sd=float(x.std(ddof=1)); half=float(t.ppf(.5+level/2,n-1)*sd/math.sqrt(n)); m=float(x.mean())
    return {'n':n,'mean':m,'sd':sd,'ci':[m-half,m+half],'level':level}

def paired_seed_difference(a, b, level=.95):
    """a, b: dict seed -> value. Differences are formed only for seeds present in both (pairing preserved)."""
    shared=sorted(set(a)&set(b))
    if len(shared)<2: raise ValueError('At least two shared seeds required for a paired interval')
    d=[a[s]-b[s] for s in shared]
    out=t_interval(d,level); out['seeds']=shared; out['sign_consistent']=bool(all(v>0 for v in d) or all(v<0 for v in d))
    return out

def bootstrap_mean(values, n_boot=10000, level=.95, seed=445):
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all(): raise ValueError('Finite 1-D values (n>=2) required')
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),(n_boot,len(x)))
    means=x[idx].mean(1); lo,hi=np.quantile(means,[(1-level)/2,.5+level/2])
    return {'n':len(x),'mean':float(x.mean()),'ci':[float(lo),float(hi)],'level':level,'n_boot':n_boot,'method':'percentile bootstrap over test items'}

def bootstrap_paired_difference(a, b, n_boot=10000, level=.95, seed=445):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape: raise ValueError('Paired item vectors must align (same test items, same order)')
    out=bootstrap_mean(a-b,n_boot,level,seed); out['method']='paired percentile bootstrap over test items'
    return out

def bonferroni_level(level, comparisons):
    if comparisons<1: raise ValueError('comparisons must be >= 1')
    return 1-(1-level)/comparisons

def rank_correlations(score_table):
    """score_table: scheme -> {model: score}. Spearman between default ranking and each scheme (None if undefined)."""
    from scipy.stats import rankdata, spearmanr
    models=sorted(next(iter(score_table.values())))
    base=rankdata([-score_table['default'][m] for m in models])
    out={}
    for k,v in score_table.items():
        r=rankdata([-v[m] for m in models])
        out[k]=float(spearmanr(base,r).statistic) if len(models)>1 and np.ptp(base)>0 and np.ptp(r)>0 else None
    return out
