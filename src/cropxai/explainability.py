"""Input-times-gradient attribution, fixed-target stability, patch deletion controls."""
import numpy as np
import torch
import torch.nn.functional as F
from .metrics import stability
from .inputs import RGB_SPEC

def attribution(model,x,target):
    model.eval(); z=x.detach().clone().requires_grad_(True)
    logits=model(z)
    grad=torch.autograd.grad(logits.gather(1,target[:,None]).sum(),z)[0]
    # Signed channel contributions are combined by absolute magnitude.
    maps=(grad*z).abs().sum(1)
    low=maps.flatten(1).min(1).values[:,None,None]; high=maps.flatten(1).max(1).values[:,None,None]
    if ((high-low)<=1e-12).any(): raise ValueError('Degenerate attribution map')
    return ((maps-low)/(high-low)).detach()

def corrupt(x, kind, severity, generator=None, spec=None):
    """Synthetic covariate shift in the unit scale of the input representation (default RGB); see inputs.InputSpec."""
    return (spec or RGB_SPEC).corrupt(x,kind,severity,generator)

def deletion(model,x,maps,targets,steps=20,patch_size=16,order='salient',seed=445,batch_size=32):
    if x.ndim!=4 or maps.shape!=(len(x),*x.shape[-2:]) or targets.shape!=(len(x),): raise ValueError('Attribution/input/target shapes differ')
    if steps<1 or patch_size<1 or x.shape[-1]%patch_size or x.shape[-2]%patch_size: raise ValueError('Image size must be divisible by patch size; positive steps')
    if not torch.isfinite(maps).all(): raise ValueError('Nonfinite attribution')
    h,w=x.shape[-2:]; ph,pw=h//patch_size,w//patch_size; n=ph*pw
    pooled=F.avg_pool2d(maps.abs()[:,None],patch_size).flatten(1)
    if order=='salient': ranking=torch.argsort(pooled,dim=1,descending=True,stable=True)
    elif order=='least': ranking=torch.argsort(pooled,dim=1,descending=False,stable=True)
    elif order=='random':
        gen=torch.Generator().manual_seed(seed)
        ranking=torch.stack([torch.randperm(n,generator=gen) for _ in x]).to(x.device)
    else: raise ValueError('Unknown deletion order')
    counts=np.unique(np.rint(np.linspace(0,n,steps+1)).astype(int)); fractions=counts/n
    curves=[]; model.eval()
    with torch.no_grad():
        for i in range(len(x)):
            images=[]
            for count in counts:
                mask=torch.ones(n,device=x.device,dtype=x.dtype); mask[ranking[i,:count]]=0
                mask=mask.reshape(ph,pw).repeat_interleave(patch_size,0).repeat_interleave(patch_size,1)
                images.append(x[i]*mask) # normalized zero = ImageNet mean RGB
            probs=[]
            for start in range(0,len(images),batch_size):
                q=model(torch.stack(images[start:start+batch_size])).softmax(1)[:,targets[i]]
                probs.extend(q.cpu().tolist())
            curves.append(probs)
    curves=np.asarray(curves); auc=np.trapezoid(curves,x=fractions,axis=1)
    return {'fractions':fractions.tolist(),'curves':curves.tolist(),'auc_per_sample':auc.tolist(),'auc':float(auc.mean()),
            'aopc':float((curves[:,0,None]-curves[:,1:]).mean()),'order':order,
            'baseline':'normalized zero (per-channel mean of the input statistics)','target':'fixed clean predicted class probability','patch_size':patch_size}

def explain(model,x,seed=445,noise_std=.05,steps=20,patch_size=16):
    model.eval()
    with torch.no_grad(): targets=model(x).argmax(1)
    maps=attribution(model,x,targets)
    noisy=corrupt(x,'gaussian_noise',noise_std,torch.Generator().manual_seed(seed),getattr(model,'input_spec',None))
    noisy_maps=attribution(model,noisy,targets)
    ssims=[stability(a.cpu().numpy(),b.cpu().numpy()) for a,b in zip(maps,noisy_maps)]
    curves={o:deletion(model,x,maps,targets,steps,patch_size,o,seed) for o in ('salient','random','least')}
    return {'method':'absolute input-times-gradient','targets':targets.cpu().tolist(),
            'noise_std_rgb':noise_std,'stability_per_sample':ssims,'stability_ssim':float(np.mean(ssims)),
            'deletion':curves,'deletion_auc':curves['salient']['auc']},maps.cpu().numpy()
