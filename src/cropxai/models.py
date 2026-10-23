"""Explicit architectures; no substitutes for unavailable foundation models."""
from pathlib import Path
import numpy as np
import torch
from torchvision import models
from .data import validate_bands, RGB_BANDS
from .inputs import RGB_SPEC, MS6_SPEC, PRITHVI_HLS_BANDS, PRITHVI_S2_BANDS, PRITHVI_TIF_INDEX
from .provenance import file_hash

SPECS={
 'ResNet-18':('resnet18','resnet18-f37072fd.pth','ResNet18_Weights.IMAGENET1K_V1'),
 'ResNet-50':('resnet50','resnet50-0676ba61.pth','ResNet50_Weights.IMAGENET1K_V1'),
 'ConvNeXt-T':('convnext_tiny','convnext_tiny-983f1562.pth','ConvNeXt_Tiny_Weights.IMAGENET1K_V1'),
 'ViT-Base':('vit_b_16','vit_b_16-c867db91.pth','ViT_B_16_Weights.IMAGENET1K_V1')}

PRITHVI='Prithvi-EO-1.0-100M'
PRITHVI_REVISION='f3a9ea7a1723621b0aeceb4de993093704d712c5'
PRITHVI_SHA256='7fac0c8a8693198e32a055e0c5a967f8b005f382182b63df1b29fdcd5c880731'
PRITHVI_CHECKPOINT=Path('artifacts/pretrained/prithvi_eo_1.0_100m/Prithvi_EO_V1_100M.pt')
ARCHITECTURES=tuple(SPECS)+(PRITHVI,)

def classifier(model):
    if hasattr(model,'fc'): return model.fc
    if hasattr(model,'heads'): return model.heads.head
    return model.classifier[-1]

class PrithviClassifier(torch.nn.Module):
    """Prithvi-EO-1.0 ViT encoder (single time step) + linear head on mean-pooled final patch tokens."""
    def __init__(self, encoder, num_classes):
        super().__init__(); self.encoder=encoder; self.fc=torch.nn.Linear(encoder.embed_dim,num_classes)
    def forward(self,x):
        tokens=self.encoder.forward_features(x)[-1]  # B,1+196,768 after the final LayerNorm
        return self.fc(tokens[:,1:].mean(1))         # CLS token is not trained for classification in MAE

def _create_prithvi(num_classes, pretrained, bands, cache):
    if tuple(bands)!=PRITHVI_S2_BANDS:
        raise ValueError(f'{PRITHVI} requires bands {list(PRITHVI_S2_BANDS)} (HLS {list(PRITHVI_HLS_BANDS)}); band sets are never padded or inferred')
    from .vendor.prithvi_mae import PrithviViT
    encoder=PrithviViT(img_size=224,patch_size=(1,16,16),num_frames=1,in_chans=6,embed_dim=768,depth=12,num_heads=12,mlp_ratio=4.)
    meta={'architecture':PRITHVI,'constructor':'src.cropxai.vendor.prithvi_mae.PrithviViT (encoder only, num_frames=1)',
          'weights':None,'checkpoint_sha256':None,'bands':list(PRITHVI_S2_BANDS),'hls_band_names':list(PRITHVI_HLS_BANDS),
          'source_tif_index':list(PRITHVI_TIF_INDEX),'channels':6,'size':224,'input_spec':MS6_SPEC.name,
          'channel_adaptation':'none: six pretrained input channels mapped one-to-one from Sentinel-2 bands; no new input parameters',
          'temporal_adaptation':'single acquisition (T=1); 3D sin-cos position embedding rebuilt for one frame',
          'pooling':'mean of final-layer patch tokens (CLS excluded)','pretraining':'random initialization'}
    if pretrained:
        path=Path(cache) if cache else PRITHVI_CHECKPOINT
        if not path.is_file(): raise FileNotFoundError(f'Prithvi checkpoint missing: {path}; download revision {PRITHVI_REVISION}')
        sha=file_hash(path)
        if sha!=PRITHVI_SHA256: raise ValueError('Prithvi checkpoint hash differs from the pinned revision')
        sd=torch.load(path,map_location='cpu',weights_only=True)
        enc={k[len('encoder.'):]:v for k,v in sd.items() if k.startswith('encoder.')}
        pos=enc.pop('pos_embed')  # (1, 1+3*196, 768): CLS + three frames
        encoder.load_state_dict({**enc,'pos_embed':encoder.pos_embed.detach().clone()},strict=True)
        # The one-frame table must equal CLS + frame 0 of the pretrained three-frame table, else T=1 is not a valid use.
        if not torch.allclose(encoder.pos_embed,pos[:,:encoder.pos_embed.shape[1]],atol=1e-6):
            raise ValueError('Rebuilt one-frame position embedding differs from checkpoint frame 0')
        meta.update(weights=f'ibm-nasa-geospatial/Prithvi-EO-1.0-100M@{PRITHVI_REVISION}',checkpoint_sha256=sha,checkpoint_path=str(path),
                    pretraining='MAE on HLS (contiguous US), six bands, three-frame clips',pos_embed_equals_checkpoint_frame0=True)
    model=PrithviClassifier(encoder,num_classes); model.input_spec=MS6_SPEC
    return model,meta

def create_model(name, num_classes, pretrained=True, bands=RGB_BANDS, cache=None):
    spec=validate_bands(bands,len(bands))
    if name==PRITHVI: return _create_prithvi(num_classes,pretrained,bands,cache)
    if name not in SPECS: raise ValueError(f'{name} is unavailable in the validated registry; proxy substitution is prohibited')
    if spec is not RGB_SPEC: raise ValueError(f'{name} was pretrained on RGB and is not adapted to bands {list(bands)}')
    constructor,filename,weight_id=SPECS[name]
    model=getattr(models,constructor)(weights=None)
    metadata={'architecture':name,'constructor':'torchvision.models.'+constructor,'weights':None,'checkpoint_sha256':None,
              'bands':list(bands),'channels':3,'size':224,'input_spec':RGB_SPEC.name,'channel_adaptation':'none','pretraining':'random initialization'}
    if pretrained:
        path=Path(cache or (Path(torch.hub.get_dir())/'checkpoints'))/filename
        if not path.is_file(): raise FileNotFoundError(f'Pretrained checkpoint not cached: {path}; no automatic download')
        sha=file_hash(path)
        if not sha.startswith(filename.rsplit('-',1)[1].split('.')[0]): raise ValueError('Checkpoint hash does not match torchvision identifier')
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        metadata.update(weights=weight_id,checkpoint_sha256=sha,checkpoint_path=str(path),pretraining='ImageNet-1K')
    head=classifier(model); replacement=torch.nn.Linear(head.in_features,num_classes)
    if hasattr(model,'fc'): model.fc=replacement
    elif hasattr(model,'heads'): model.heads.head=replacement
    else: model.classifier[-1]=replacement
    model.input_spec=RGB_SPEC
    return model,metadata

def freeze_backbone(model):
    for p in model.parameters(): p.requires_grad_(False)
    for p in classifier(model).parameters(): p.requires_grad_(True)
    # Keep all pretrained BN buffers/dropout fixed during linear probing.
    model.eval()
