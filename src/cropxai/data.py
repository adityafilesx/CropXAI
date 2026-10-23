"""Preserve current splits, hash inputs, and reject overlap before training."""
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import random, json
import numpy as np
from PIL import Image
from .provenance import file_hash, digest, write_json
from .inputs import RGB_SPEC, MS6_SPEC, spec_for_bands, PRITHVI_TIF_INDEX, TIF_BAND_ORDER

RGB_BANDS=list(RGB_SPEC.bands)
MEAN=list(RGB_SPEC.mean); STD=list(RGB_SPEC.std)

def validate_bands(bands, channels):
    """Bands must be a registered representation (RGB or the six Prithvi bands) with the matching channel count.

    Missing bands are never inferred, padded or substituted.
    """
    spec=spec_for_bands(bands)
    if channels!=spec.channels: raise ValueError(f'{spec.name} has {spec.channels} channels, not {channels}')
    return spec

def audit_dataset(root='data/processed/eurosat', output=None):
    root=Path(root); rows=[]; by_hash=defaultdict(list); by_pixels=defaultdict(list); counts={}; errors=[]
    classes=sorted(p.name for p in (root/'train').iterdir() if p.is_dir())
    if len(classes)<2: raise ValueError('Need at least two classes')
    for split in ('train','val','test'):
        split_classes=sorted(p.name for p in (root/split).iterdir() if p.is_dir())
        if split_classes!=classes: errors.append(f'class mapping differs in {split}')
        count=Counter()
        for p in sorted((root/split).rglob('*.jpg')):
            rel=str(p.relative_to(root)); sha=file_hash(p)
            with Image.open(p) as im:
                im.load(); mode=im.mode; size=list(im.size)
                pixels=digest({'mode':mode,'size':size,'pixels_sha':__import__('hashlib').sha256(im.tobytes()).hexdigest()})
            if mode!='RGB' or size!=[64,64]: errors.append(f'Unexpected image metadata {rel}: {mode}, {size}')
            row={'path':rel,'split':split,'class':p.parent.name,'label':classes.index(p.parent.name),'sha256':sha,'pixel_sha256':pixels,'mode':mode,'size':size}
            rows.append(row); by_hash[sha].append(rel); by_pixels[pixels].append(rel); count[p.parent.name]+=1
        counts[split]=dict(sorted(count.items()))
        if set(count)!=set(classes): errors.append(f'missing class data in {split}')
    duplicates=[v for v in by_pixels.values() if len(v)>1]
    overlap=[v for v in duplicates if len({p.split('/')[0] for p in v})>1]
    if overlap: errors.append('Pixel-identical images overlap splits')
    result={'schema_version':1,'root':str(root),'dataset':'EuroSAT-RGB','release':'NOT AVAILABLE; local content identified by hashes',
            'classes':classes,'bands':RGB_BANDS,'counts':counts,'n':len(rows),'split_id':digest(rows),
            'split_method':'preserved legacy class-wise random split; not geographically independent',
            'file_duplicates':[v for v in by_hash.values() if len(v)>1],'pixel_duplicates':duplicates,
            'cross_split_duplicates':overlap,'errors':errors,'samples':rows}
    if output: write_json(output,result)
    return result

def stratified_indices(labels, per_class=None, seed=445):
    labels=np.asarray(labels); rng=np.random.default_rng(seed); indices=[]
    for cls in sorted(np.unique(labels)):
        idx=np.flatnonzero(labels==cls); idx=rng.permutation(idx)
        if per_class is not None: idx=idx[:per_class]
        indices.extend(idx.tolist())
    return sorted(indices)

def deterministic_split(paths, seed=42):
    paths=sorted(paths); random.Random(seed).shuffle(paths); n=len(paths); a=int(.7*n); b=a+int(.15*n)
    return {'train':paths[:a],'val':paths[a:b],'test':paths[b:]}

class TifFolder:
    """Sentinel-2 GeoTIFF view of an existing JPEG split (same classes, same order, same sample IDs).

    Reads the registered bands from the 13-band EuroSAT TIFFs, normalizes with the representation's fixed
    pretraining statistics (never fitted on evaluation data), and bilinearly upsamples 64->size pixels.
    """
    def __init__(self, jpeg_split_root, tif_root, spec, size=224, flip=False):
        from torchvision import datasets
        base=datasets.ImageFolder(jpeg_split_root)  # scans names only; no image is decoded
        self.class_to_idx=base.class_to_idx; self.targets=base.targets; self.spec=spec; self.size=size; self.flip=flip
        self.samples=[(str(Path(tif_root)/Path(p).parent.name/(Path(p).stem+'.tif')),t) for p,t in base.samples]
        self.jpeg_samples=base.samples
        self.index=[i+1 for i in PRITHVI_TIF_INDEX]  # rasterio band numbers are 1-based
        mean=np.array(spec.mean,np.float32)[:,None,None]; std=np.array(spec.std,np.float32)[:,None,None]
        self._mean,self._std=mean,std
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        import torch, rasterio
        import torch.nn.functional as F
        path,target=self.samples[i]
        with rasterio.open(path) as src:
            if src.count!=13 or src.dtypes[0]!='uint16': raise ValueError(f'Unexpected GeoTIFF layout: {path}')
            dn=src.read(self.index).astype(np.float32)
        x=torch.from_numpy((dn-self._mean)/self._std)[None]
        x=F.interpolate(x,size=(self.size,self.size),mode='bilinear',align_corners=False)[0]
        if self.flip and torch.rand(1).item()<.5: x=x.flip(-1)
        return x,target

def loaders(root, size=224, batch_size=32, seed=42, per_class=None, training=True, bands=None, tif_root=None):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, Subset
    result={}; selections={}; mapping=None
    spec=spec_for_bands(bands) if bands is not None else RGB_SPEC
    for split in ('train','val','test'):
        if spec is RGB_SPEC:
            ops=[transforms.Resize((size,size),antialias=True)]
            if split=='train' and training: ops.append(transforms.RandomHorizontalFlip())
            ops.extend([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
            ds=datasets.ImageFolder(Path(root)/split,transform=transforms.Compose(ops))
        elif spec is MS6_SPEC:
            if tif_root is None: raise ValueError('tif_root required for multispectral input')
            ds=TifFolder(Path(root)/split,tif_root,spec,size,flip=(split=='train' and training))
        else: raise ValueError('Unsupported input representation')
        if mapping is None: mapping=ds.class_to_idx
        if mapping!=ds.class_to_idx: raise ValueError('Different class mappings')
        indices=stratified_indices(ds.targets, (per_class or {}).get(split),seed=445)
        ids=ds.jpeg_samples if spec is MS6_SPEC else ds.samples
        selections[split]=[str(Path(ids[i][0]).relative_to(root)) for i in indices]
        result[split]=DataLoader(Subset(ds,indices),batch_size=batch_size,shuffle=(split=='train' and training),num_workers=0,generator=torch.Generator().manual_seed(seed))
    return result,mapping,selections

def band_order_evidence(tif_paths, expected_order=TIF_BAND_ORDER):
    """Spectral sanity checks that test the assumed 13-band storage order of the EuroSAT GeoTIFFs.

    The files carry no band metadata, so the order is inferred from physics and then asserted:
    B8A (865 nm) must track B08 (842 nm) almost perfectly, B10 (cirrus) must be near zero, and B11 > B12 on average.
    """
    import rasterio
    stack=np.stack([rasterio.open(p).read().astype(np.float64) for p in tif_paths])  # N,13,H,W
    flat=stack.transpose(1,0,2,3).reshape(13,-1); corr=np.corrcoef(flat); idx={b:i for i,b in enumerate(expected_order)}
    means=flat.mean(1)
    checks={'corr_B8A_B08':float(corr[idx['B8A'],idx['B08']]),'corr_B09_B08':float(corr[idx['B09'],idx['B08']]),
            'mean_B10_cirrus_DN':float(means[idx['B10']]),'mean_B11_DN':float(means[idx['B11']]),'mean_B12_DN':float(means[idx['B12']]),
            'corr_B11_B12':float(corr[idx['B11'],idx['B12']])}
    checks['passed']=bool(checks['corr_B8A_B08']>.95 and checks['corr_B8A_B08']>checks['corr_B09_B08']+.1
                          and checks['mean_B10_cirrus_DN']<100 and checks['mean_B11_DN']>checks['mean_B12_DN'])
    return {'assumed_order':list(expected_order),'band_means_DN':dict(zip(expected_order,means.round(1).tolist())),'checks':checks}

def audit_ms_dataset(rgb_manifest, tif_root, output=None, pair_samples=200, seed=445):
    """Hash every GeoTIFF of the audited JPEG split and verify the JPEG<->GeoTIFF pairing and layout."""
    import rasterio
    rgb=json.loads(Path(rgb_manifest).read_text()); tif_root=Path(tif_root); rows=[]; errors=[]
    for r in rgb['samples']:
        stem=Path(r['path']).stem; p=tif_root/r['class']/(stem+'.tif')
        if not p.is_file(): errors.append(f'missing GeoTIFF {p}'); continue
        with rasterio.open(p) as s:
            if (s.count,s.width,s.height,s.dtypes[0])!=(13,64,64,'uint16'): errors.append(f'unexpected layout {p}: {s.count},{s.width},{s.height},{s.dtypes[0]}')
        rows.append({'path':f"{r['class']}/{stem}.tif",'jpeg_path':r['path'],'split':r['split'],'class':r['class'],'label':r['label'],'sha256':file_hash(p)})
    rng=np.random.default_rng(seed); pick=rng.choice(len(rows),min(pair_samples,len(rows)),replace=False); textured=[]; jmeans=[]; tmeans=[]
    for i in sorted(pick):
        with Image.open(Path(rgb['root'])/rgb['samples'][i]['path']) as im: jpg=np.asarray(im.convert('RGB'),float)
        with rasterio.open(tif_root/rows[i]['path']) as s: tif=s.read([4,3,2]).astype(float)  # B04,B03,B02 -> 1-based 4,3,2
        jmeans.append(jpg.mean((0,1))); tmeans.append(tif.mean((1,2)))
        # Per-image correlation is only informative when the 8-bit JPEG has texture; flat water/forest patches
        # are dominated by quantization noise (std ~ 1 gray level) and are checked through the channel means instead.
        if min(jpg[...,c].std() for c in range(3))>=5:
            textured.append(min(float(np.corrcoef(jpg[...,c].ravel(),tif[c].ravel())[0,1]) for c in range(3)))
    jm,tm=np.array(jmeans),np.array(tmeans); mean_corr=[float(np.corrcoef(jm[:,c],tm[:,c])[0,1]) for c in range(3)]
    if len(textured)<20 or min(textured)<.7 or min(mean_corr)<.95:
        errors.append('JPEG and GeoTIFF do not describe the same patches (textured min corr < 0.7 or channel-mean corr < 0.95)')
    pairing_summary={'n_checked':len(pick),'n_textured':len(textured),'textured_min_corr':min(textured) if textured else None,
                     'textured_median_corr':float(np.median(textured)) if textured else None,'channel_mean_corr_RGB':mean_corr}
    sample=[str(tif_root/rows[i]['path']) for i in rng.choice(len(rows),min(400,len(rows)),replace=False)]
    evidence=band_order_evidence(sample)
    if not evidence['checks']['passed']: errors.append('Assumed GeoTIFF band order failed spectral sanity checks')
    result={'schema_version':1,'root':str(tif_root),'dataset':'EuroSAT-MS-Sentinel2-L1C-GeoTIFF','release':'NOT AVAILABLE; local content identified by hashes',
            'classes':rgb['classes'],'bands':list(MS6_SPEC.bands),'source_tif_index':list(PRITHVI_TIF_INDEX),'n':len(rows),'counts':rgb['counts'],
            'split_id':digest(rows),'rgb_split_id':rgb['split_id'],'rgb_manifest':str(rgb_manifest),
            'split_method':rgb['split_method']+'; GeoTIFF rows follow the JPEG split by file stem',
            'jpeg_tif_pairing':pairing_summary,
            'band_order_evidence':evidence,'errors':errors,'samples':rows}
    if output: write_json(output,result)
    return result
