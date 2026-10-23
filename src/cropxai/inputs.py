"""Input representations. One spec per tensor layout; it owns channels, normalization and corruption.

RGB   : EuroSAT-RGB JPEG, ImageNet statistics, 'unit scale' = pixel/255.
MS6   : EuroSAT Sentinel-2 GeoTIFF, the six bands Prithvi-EO-1.0 was pretrained on, pretraining statistics,
        'unit scale' = reflectance = DN/10000. Normalization is never fitted on the evaluation data.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

# Order of the 13 bands inside the EuroSAT GeoTIFFs shipped in data/raw/eurosat (verified empirically:
# scripts/audit_bands.py; the files carry no band metadata and the order differs from the nominal
# B01..B08,B8A,B09..B12 listing - B8A is stored last).
TIF_BAND_ORDER = ('B01','B02','B03','B04','B05','B06','B07','B08','B09','B10','B11','B12','B8A')
# Prithvi-EO-1.0 channel names are HLS-L30 style; the Sentinel-2 equivalents are given in the same order.
PRITHVI_HLS_BANDS = ('B02','B03','B04','B05','B06','B07')
PRITHVI_S2_BANDS = ('B02','B03','B04','B8A','B11','B12')
PRITHVI_TIF_INDEX = tuple(TIF_BAND_ORDER.index(b) for b in PRITHVI_S2_BANDS)
# Pretraining statistics (DN units, HLS surface reflectance x 10000) from the checkpoint's config.json.
PRITHVI_MEAN = (775.2290211032589, 1080.992780391705, 1228.5855250417867, 2497.2022620507532, 2204.2139147975554, 1610.8324823273745)
PRITHVI_STD = (1281.526139861424, 1270.0297974547493, 1399.4802505642526, 1368.3446143747644, 1291.6764008585435, 1154.505683480695)
RGB_BANDS = ('B04','B03','B02')
IMAGENET_MEAN = (.485,.456,.406); IMAGENET_STD = (.229,.224,.225)

@dataclass(frozen=True)
class InputSpec:
    name: str
    bands: tuple          # semantic names in tensor channel order
    mean: tuple           # in raw units (RGB: unit scale; MS6: DN)
    std: tuple
    raw_per_unit: float   # raw units per 'unit scale' (RGB 1.0; MS6 10000 DN per reflectance unit)
    clamp_max: float | None  # upper clamp on unit scale (RGB 1.0; reflectance unbounded)

    @property
    def channels(self): return len(self.bands)

    def _stats(self, x):
        m = x.new_tensor(self.mean)[None,:,None,None]; s = x.new_tensor(self.std)[None,:,None,None]
        return m, s

    def corrupt(self, x, kind, severity, generator=None):
        """Synthetic covariate shift applied on the unit scale, then re-normalized.

        gaussian_noise : unit-scale additive N(0, severity^2) per element (independent across channels).
        brightness     : multiply unit-scale values by (1 - severity).
        """
        if x.ndim != 4 or x.shape[1] != self.channels: raise ValueError(f'{self.name}: expected B,{self.channels},H,W tensor, got {tuple(x.shape)}')
        if severity < 0 or not np.isfinite(severity): raise ValueError('Invalid severity')
        m, s = self._stats(x)
        unit = (x * s + m) / self.raw_per_unit
        if kind == 'gaussian_noise':
            unit = unit + torch.randn(x.shape, generator=generator, device='cpu', dtype=x.dtype).to(x.device) * severity
        elif kind == 'brightness':
            unit = unit * (1 - severity)
        else:
            raise ValueError('Unsupported corruption')
        unit = unit.clamp(min=0) if self.clamp_max is None else unit.clamp(0, self.clamp_max)
        return (unit * self.raw_per_unit - m) / s

    def describe(self):
        return {'name': self.name, 'bands': list(self.bands), 'channels': self.channels, 'mean': list(self.mean), 'std': list(self.std),
                'raw_per_unit': self.raw_per_unit, 'clamp_max': self.clamp_max}

RGB_SPEC = InputSpec('eurosat_rgb_jpeg', RGB_BANDS, IMAGENET_MEAN, IMAGENET_STD, 1.0, 1.0)
MS6_SPEC = InputSpec('eurosat_s2_tif_prithvi6', PRITHVI_S2_BANDS, PRITHVI_MEAN, PRITHVI_STD, 10000.0, None)
SPECS = {s.name: s for s in (RGB_SPEC, MS6_SPEC)}

def spec_for_bands(bands):
    for s in SPECS.values():
        if tuple(bands) == s.bands: return s
    raise ValueError(f'No registered input representation has bands {list(bands)}; band sets are never inferred or padded')
