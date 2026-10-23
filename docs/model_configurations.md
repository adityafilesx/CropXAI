# Model-input audit

| Registry name | Actual architecture | Representation/bands/order | Channels, tensor, resolution | Normalization | Checkpoint/adaptation | Status |
|---|---|---|---|---|---|---|
| ResNet-18 | torchvision resnet18 | EuroSAT RGB; B04/B03/B02 | 3, Bx3x224x224 | ImageNet mean/SD | resnet18-f37072fd.pth, V1; no channel change; new 10-class head | Supported; checkpoint hash recorded per new run |
| ResNet-50 | torchvision resnet50 | same RGB protocol | same | same | resnet50-0676ba61.pth; no channel change | Supported loader; not in new main campaign |
| ViT-Base | torchvision vit_b_16 | same RGB protocol | same; 16x16 token patches | same | vit_b_16-c867db91.pth, V1; no channel change | Supported; original proxy records quarantined |
| ConvNeXt-T | torchvision convnext_tiny | same RGB protocol | same | same | convnext_tiny-983f1562.pth, V1; no channel change | Supported |
| Prithvi-100M (legacy label) | ViT-Base proxy in active old registry | RGB, no HLS adaptation | Bx3x224x224 | ImageNet | No actual Prithvi checkpoint identified | Excluded as Prithvi evidence; loader now rejects it |
| Prithvi-EO-1.0-100M | Vendored `PrithviViT` encoder (Apache-2.0), single time step, mean-pooled patch tokens + linear head | EuroSAT Sentinel-2 GeoTIFF; B02,B03,B04,B8A,B11,B12 = Prithvi HLS B02..B07 (GeoTIFF indices 1,2,3,12,10,11) | 6, Bx6x224x224 (T=1 added inside the encoder) | Prithvi pretraining mean/std (DN), taken from the checkpoint's config.json | `Prithvi_EO_V1_100M.pt`, HF revision f3a9ea7a, SHA-256 7fac0c8a...; no channel adaptation; one-frame position embedding verified equal to checkpoint frame 0 | Supported; five-seed cohort executed. See MODEL_INPUT_AUDIT.md |
| SatMAE/DOFA/ScaleMAE/SegFormer legacy names | proxies | no verified architecture-specific input record | NOT AVAILABLE | NOT AVAILABLE | rejected; no scientific results attributed to these models | Unsupported |

The RGB models use three channels; Prithvi uses six Sentinel-2 bands from the GeoTIFF release. The cohorts therefore do not share bands; they share the split and test items. The band-to-channel mapping, the empirical GeoTIFF band order and the domain differences (apparent Level-1C top-of-atmosphere vs HLS surface reflectance, 64->224 upsampling) are documented in `docs/revision/MODEL_INPUT_AUDIT.md`, generated from run records. Legacy records lack input hashes. New runs record per-file selection, split, checkpoint and input-spec identities and assert tensor shapes against the registered representation. No 13-band experiment exists.
