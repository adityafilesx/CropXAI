"""Second test module: multispectral inputs, statistics, compliance failures, fallback elimination, legacy reconstruction."""
import copy, json, math, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import torch

from src.cropxai.inputs import (RGB_SPEC, MS6_SPEC, TIF_BAND_ORDER, PRITHVI_TIF_INDEX, PRITHVI_S2_BANDS, PRITHVI_HLS_BANDS,
                                spec_for_bands, PRITHVI_MEAN, PRITHVI_STD)
from src.cropxai.data import validate_bands, band_order_evidence, TifFolder, loaders
from src.cropxai.models import create_model, PRITHVI, PRITHVI_CHECKPOINT, freeze_backbone, classifier, ARCHITECTURES
from src.cropxai.explainability import corrupt
from src.cropxai.metrics import PILLARS, composite
from src.cropxai.provenance import digest, file_hash, source_hash
from src.cropxai_analysis.stats import (t_interval, paired_seed_difference, bootstrap_mean, bootstrap_paired_difference,
                                       bonferroni_level, rank_correlations)
from src.cropxai_analysis.compliance import check_cohort

ROOT = Path(__file__).resolve().parents[1]


class InputSpecTests(unittest.TestCase):
    def test_band_order_and_prithvi_mapping(self):
        # The GeoTIFFs store B8A last (empirical audit); the nominal order would give [1,2,3,8,11,12].
        self.assertEqual(TIF_BAND_ORDER[-1], 'B8A')
        self.assertEqual(PRITHVI_TIF_INDEX, (1, 2, 3, 12, 10, 11))
        self.assertEqual(tuple(TIF_BAND_ORDER[i] for i in PRITHVI_TIF_INDEX), PRITHVI_S2_BANDS)
        self.assertEqual(PRITHVI_HLS_BANDS, ('B02', 'B03', 'B04', 'B05', 'B06', 'B07'))
        self.assertEqual(MS6_SPEC.channels, 6); self.assertEqual(RGB_SPEC.channels, 3)

    def test_bands_never_inferred(self):
        self.assertIs(validate_bands(list(PRITHVI_S2_BANDS), 6), MS6_SPEC)
        self.assertIs(validate_bands(['B04', 'B03', 'B02'], 3), RGB_SPEC)
        for bands, ch in [(['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'][::-1], 6), (['B02', 'B03', 'B04'], 3),
                          (list(PRITHVI_S2_BANDS), 3), (['B04', 'B03', 'B02', 'B08'], 4)]:
            with self.assertRaises(ValueError): validate_bands(bands, ch)
        with self.assertRaises(ValueError): spec_for_bands(['B01'])

    def test_rgb_spec_matches_historic_corruption(self):
        x = torch.rand(2, 3, 4, 4) * 2 - 1
        mean = torch.tensor(RGB_SPEC.mean)[None, :, None, None]; std = torch.tensor(RGB_SPEC.std)[None, :, None, None]
        g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
        rgb = (x * std + mean) + torch.randn(x.shape, generator=g1) * .1
        expected = (rgb.clamp(0, 1) - mean) / std
        self.assertTrue(torch.allclose(corrupt(x, 'gaussian_noise', .1, g2), expected, atol=1e-6))

    def test_ms6_corruption_units(self):
        mean = torch.tensor(PRITHVI_MEAN)[None, :, None, None]; std = torch.tensor(PRITHVI_STD)[None, :, None, None]
        dn = torch.full((1, 6, 4, 4), 2000.); x = (dn - mean) / std
        b = corrupt(x, 'brightness', .5, spec=MS6_SPEC); back = b * std + mean
        self.assertTrue(torch.allclose(back, dn * .5, atol=1e-2))              # reflectance scaled by (1-severity)
        n = corrupt(torch.zeros(1, 6, 64, 64), 'gaussian_noise', 0, spec=MS6_SPEC)
        self.assertTrue(torch.allclose(n, torch.zeros_like(n), atol=1e-5) or (n * std + mean).min() >= 0)
        noisy = corrupt(x, 'gaussian_noise', .05, torch.Generator().manual_seed(0), MS6_SPEC) * std + mean
        self.assertGreaterEqual(float(noisy.min()), 0)                         # reflectance never negative
        with self.assertRaises(ValueError): corrupt(torch.zeros(1, 3, 4, 4), 'brightness', .1, spec=MS6_SPEC)  # wrong channel count


class SyntheticGeoTiffTests(unittest.TestCase):
    """End-to-end band mapping: channel c of the loader must equal normalized stored band PRITHVI_TIF_INDEX[c]."""
    def setUp(self):
        import rasterio
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory(); root = Path(self.tmp.name)
        self.jpeg, self.tif = root / 'jpeg', root / 'tif'
        for split in ('train', 'val', 'test'):
            for cls in ('A', 'B'):
                (self.jpeg / split / cls).mkdir(parents=True); (self.tif / cls).mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (64, 64)).save(self.jpeg / split / cls / f'{cls}_{split}.jpg')
                data = np.stack([np.full((64, 64), 1000 + 111 * k, np.uint16) for k in range(13)])
                with rasterio.open(self.tif / cls / f'{cls}_{split}.tif', 'w', driver='GTiff', height=64, width=64, count=13, dtype='uint16') as dst: dst.write(data)
    def tearDown(self): self.tmp.cleanup()
    def test_channel_mapping(self):
        ds = TifFolder(self.jpeg / 'test', self.tif, MS6_SPEC, size=32)
        x, y = ds[0]
        self.assertEqual(tuple(x.shape), (6, 32, 32)); self.assertEqual(ds.class_to_idx, {'A': 0, 'B': 1})
        for c, idx in enumerate(PRITHVI_TIF_INDEX):
            expected = ((1000 + 111 * idx) - PRITHVI_MEAN[c]) / PRITHVI_STD[c]
            self.assertAlmostEqual(float(x[c].mean()), expected, places=4)
    def test_loaders_use_jpeg_sample_ids(self):
        ld, mapping, sel = loaders(str(self.jpeg), 32, 2, 1, None, False, bands=list(PRITHVI_S2_BANDS), tif_root=str(self.tif))
        self.assertEqual(sel['test'], ['test/A/A_test.jpg', 'test/B/B_test.jpg'])
        x, y = next(iter(ld['test'])); self.assertEqual(tuple(x.shape), (2, 6, 32, 32))
        with self.assertRaises(ValueError): loaders(str(self.jpeg), 32, 2, 1, None, False, bands=list(PRITHVI_S2_BANDS))  # tif_root required
    def test_corrupt_geotiff_layout_rejected(self):
        import rasterio
        bad = self.tif / 'A' / 'A_test.tif'
        with rasterio.open(bad, 'w', driver='GTiff', height=64, width=64, count=3, dtype='uint16') as dst: dst.write(np.zeros((3, 64, 64), np.uint16))
        with self.assertRaises(ValueError): TifFolder(self.jpeg / 'test', self.tif, MS6_SPEC, 32)[0]


class BandOrderEvidenceTests(unittest.TestCase):
    def _write(self, directory, order_swap=False):
        import rasterio
        rng = np.random.default_rng(0); paths = []
        for i in range(6):
            nir = rng.uniform(500, 3000, (64, 64)); base = {b: rng.uniform(300, 1500, (64, 64)) for b in TIF_BAND_ORDER}
            base['B08'] = nir; base['B8A'] = nir * 1.03 + rng.normal(0, 10, nir.shape)   # B8A tracks B08
            base['B09'] = rng.uniform(300, 900, nir.shape); base['B10'] = rng.uniform(5, 20, nir.shape)
            base['B11'] = rng.uniform(1200, 2400, nir.shape); base['B12'] = rng.uniform(600, 1200, nir.shape)
            stack = np.stack([base[b] for b in TIF_BAND_ORDER]).astype(np.uint16)
            if order_swap: stack[[8, 12]] = stack[[12, 8]]                                  # store B8A at nominal position 8
            p = Path(directory) / f'{i}.tif'
            with rasterio.open(p, 'w', driver='GTiff', height=64, width=64, count=13, dtype='uint16') as dst: dst.write(stack)
            paths.append(p)
        return paths
    def test_passes_for_documented_order_and_fails_when_swapped(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(band_order_evidence(self._write(d))['checks']['passed'])
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(band_order_evidence(self._write(d, order_swap=True))['checks']['passed'])


class PrithviModelTests(unittest.TestCase):
    def test_random_init_shape_and_freeze(self):
        torch.manual_seed(0); torch.set_num_threads(2)
        m, meta = create_model(PRITHVI, 10, False, list(PRITHVI_S2_BANDS)); freeze_backbone(m)
        with torch.no_grad(): out = m(torch.zeros(1, 6, 224, 224))
        self.assertEqual(tuple(out.shape), (1, 10)); self.assertIs(m.input_spec, MS6_SPEC)
        self.assertEqual(meta['channels'], 6); self.assertEqual(meta['hls_band_names'], list(PRITHVI_HLS_BANDS))
        self.assertEqual({n for n, p in m.named_parameters() if p.requires_grad}, {'fc.weight', 'fc.bias'})
        self.assertIs(classifier(m), m.fc)
    def test_wrong_bands_rejected(self):
        for bands in (['B04', 'B03', 'B02'], list(PRITHVI_S2_BANDS)[::-1]):
            with self.assertRaises(ValueError): create_model(PRITHVI, 10, False, bands)
        with self.assertRaises(ValueError): create_model('ResNet-18', 10, False, list(PRITHVI_S2_BANDS))  # RGB model never adapted silently
    def test_unlisted_names_rejected(self):
        for name in ('Prithvi-100M', 'SatMAE', 'DOFA'):
            self.assertNotIn(name, ARCHITECTURES)
            with self.assertRaises(ValueError): create_model(name, 10, False)
    @unittest.skipUnless((ROOT / PRITHVI_CHECKPOINT).exists(), 'Prithvi checkpoint not downloaded')
    def test_pretrained_checkpoint_identity_and_position_embedding(self):
        m, meta = create_model(PRITHVI, 10, True, list(PRITHVI_S2_BANDS), cache=ROOT / PRITHVI_CHECKPOINT)
        self.assertTrue(meta['pos_embed_equals_checkpoint_frame0'])
        self.assertEqual(meta['checkpoint_sha256'], '7fac0c8a8693198e32a055e0c5a967f8b005f382182b63df1b29fdcd5c880731')
        self.assertTrue(torch.isfinite(m.encoder.patch_embed.proj.weight).all())
        # weights are the pretrained ones, not the random initialization of the same architecture
        r, _ = create_model(PRITHVI, 10, False, list(PRITHVI_S2_BANDS))
        self.assertFalse(torch.equal(m.encoder.patch_embed.proj.weight, r.encoder.patch_embed.proj.weight))
    def test_tampered_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x.pt'; p.write_bytes(b'not a checkpoint')
            with self.assertRaises(ValueError): create_model(PRITHVI, 10, True, list(PRITHVI_S2_BANDS), cache=p)


class StatisticsTests(unittest.TestCase):
    def test_t_interval_known(self):
        r = t_interval([1, 2, 3, 4, 5]); self.assertEqual(r['mean'], 3); self.assertAlmostEqual(r['ci'][0], 1.0367568385, places=8)
        self.assertIsNone(t_interval([1.])['ci']); self.assertEqual(t_interval([.5, .5])['ci'], [.5, .5])
        with self.assertRaises(ValueError): t_interval([1, float('nan')])
    def test_paired_difference_preserves_pairing(self):
        a = {1: .9, 2: .8, 3: .7}; b = {1: .8, 2: .7, 3: .6, 4: .1}   # seed 4 unshared -> ignored
        r = paired_seed_difference(a, b); self.assertEqual(r['seeds'], [1, 2, 3]); self.assertAlmostEqual(r['mean'], .1)
        self.assertLess(r['sd'], 1e-9); self.assertTrue(r['sign_consistent'])
        with self.assertRaises(ValueError): paired_seed_difference({1: 1}, {1: 2})
    def test_bootstrap_deterministic_and_covers(self):
        x = np.r_[np.ones(900), np.zeros(100)]
        r1, r2 = bootstrap_mean(x, 2000, seed=3), bootstrap_mean(x, 2000, seed=3)
        self.assertEqual(r1, r2); self.assertLess(r1['ci'][0], .9); self.assertGreater(r1['ci'][1], .9); self.assertLess(r1['ci'][1] - r1['ci'][0], .06)
    def test_paired_bootstrap_requires_alignment_and_detects_shift(self):
        a = np.r_[np.ones(95), np.zeros(5)]; b = np.zeros(100)
        r = bootstrap_paired_difference(a, b, 2000); self.assertGreater(r['ci'][0], .85)
        with self.assertRaises(ValueError): bootstrap_paired_difference(a, b[:-1])
    def test_bonferroni_and_rank_correlation(self):
        self.assertAlmostEqual(bonferroni_level(.95, 3), 1 - .05 / 3)
        with self.assertRaises(ValueError): bonferroni_level(.95, 0)
        table = {'default': {'a': 3, 'b': 2, 'c': 1}, 'flip': {'a': 1, 'b': 2, 'c': 3}, 'tie': {'a': 1, 'b': 1, 'c': 1}}
        r = rank_correlations(table); self.assertAlmostEqual(r['default'], 1); self.assertAlmostEqual(r['flip'], -1); self.assertIsNone(r['tie'])


class SourceIdentityTests(unittest.TestCase):
    def test_analysis_edits_do_not_change_run_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'src/cropxai').mkdir(parents=True); (root / 'scripts').mkdir()
            (root / 'src/cropxai/a.py').write_text('x=1'); (root / 'scripts/analyze.py').write_text('y=1'); (root / 'src/cropxai/cli.py').write_text('z=1')
            h = source_hash(root)
            (root / 'scripts/analyze.py').write_text('y=2'); (root / 'src/cropxai/cli.py').write_text('z=2')
            self.assertEqual(h, source_hash(root))
            (root / 'src/cropxai/a.py').write_text('x=2'); self.assertNotEqual(h, source_hash(root))


def _record(rid, model, seed, root, *, epochs=5, weights='w', sha='s', bands=None, spec=RGB_SPEC, source='src1', split='split1', items=None, manifest=None):
    bands = list(bands or spec.bands); rd = root / 'artifacts' / 'revision' / rid; rd.mkdir(parents=True)
    items = items or ['test/A/1.jpg', 'test/A/2.jpg', 'test/B/3.jpg']
    n = len(items)
    (rd / 'samples.json').write_text(json.dumps({'test': items}))
    (rd / 'history.json').write_text(json.dumps([{'epoch': e + 1} for e in range(epochs)]))
    (rd / 'explanations.json').write_text(json.dumps({'sample_ids': items, 'targets': [0] * n, 'stability_per_sample': [.1 * (i + 1) for i in range(n)],
                                                      'deletion': {'salient': {'auc_per_sample': [.2] * n}}}))
    for f in ('predictions.npz', 'attributions.npz', 'checkpoint.pt'): (rd / f).write_bytes(rid.encode() + f.encode())
    pillars = {k: .5 for k in PILLARS}; w = {k: 1 / 6 for k in PILLARS}
    cfg = {'purpose': 'controlled_benchmark', 'cohort': 'toy', 'epochs': 5, 'batch_size': 32, 'learning_rate': .001, 'finetune_mode': 'frozen_backbone_eval', 'size': 224,
           'pretrained': True, 'feature_cache': True, 'bands': bands, 'input_representation': spec.name, 'evaluation_config': 'e.json',
           'seeds': [1, 2], 'models': ['ResNet-18', 'ViT-Base'], 'manifest': str(manifest), 'evaluation': {'weights': {'default': w}}}
    cfg = {**cfg, 'epochs': epochs}
    artifacts = {f'artifacts/revision/{rid}/{p.name}': file_hash(p) for p in sorted(rd.iterdir())}
    return {'schema_version': 2, 'run_id': rid, 'status': 'completed', 'config': cfg, 'config_hash': digest(cfg), 'source_revision': source, 'seed': seed,
            'model': {'architecture': model, 'weights': weights, 'checkpoint_sha256': sha, 'bands': bands, 'channels': len(bands), 'input_spec': spec.name},
            'input': {'channels': len(bands)}, 'dataset': {'split_id': split, 'manifest_sha256': file_hash(manifest)}, 'environment': {'packages': {'torch': '2'}},
            'artifacts': artifacts, 'metrics': {'accuracy': .9, 'macro_f1': .9, 'ece': .05, 'nll': .3, 'brier': .2, 'stability_ssim': .4, 'deletion_auc': .3,
                                                'silhouette': .2, 'pillars': pillars, 'composite_trust': composite(pillars, w)}}


class ComplianceTests(unittest.TestCase):
    """The checker must fail for every protocol violation and pass for a clean matrix."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.manifest = self.root / 'manifest.json'; self.manifest.write_text(json.dumps({'errors': []}))
        self.config = {'purpose': 'controlled_benchmark', 'cohort': 'toy', 'models': ['ResNet-18', 'ViT-Base'], 'seeds': [1, 2], 'epochs': 5, 'batch_size': 32,
                       'learning_rate': .001, 'finetune_mode': 'frozen_backbone_eval', 'size': 224, 'pretrained': True, 'feature_cache': True,
                       'bands': list(RGB_SPEC.bands), 'input_representation': RGB_SPEC.name, 'evaluation_config': 'e.json', 'manifest': str(self.manifest)}
        self.records = [_record(f'{m}{s}', m, s, self.root, sha=m, manifest=self.manifest) for m in self.config['models'] for s in self.config['seeds']]
    def tearDown(self): self.tmp.cleanup()
    def run_check(self, records=None, config=None, **kw):
        return check_cohort(config or self.config, records if records is not None else self.records, artifact_root=self.root, current_source='src1', **kw)
    def failing(self, report): return {c['check'] for c in report['checks'] if c['critical'] and not c['passed']}

    def test_clean_matrix_passes(self):
        report = self.run_check(); self.assertTrue(report['ok'], self.failing(report))
    def test_missing_seed_fails(self):
        self.assertIn('matrix_complete', self.failing(self.run_check(self.records[:-1])))
    def test_duplicate_run_fails(self):
        extra = copy.deepcopy(self.records[0]); extra['run_id'] = 'dup'
        self.assertIn('matrix_complete', self.failing(self.run_check(self.records + [extra])))
    def test_mixed_epochs_fail(self):
        bad = [copy.deepcopy(r) for r in self.records]; bad[0]['config']['epochs'] = 1; bad[0]['config_hash'] = digest(bad[0]['config'])
        self.assertIn('protocol_homogeneous', self.failing(self.run_check(bad)))
    def test_history_shorter_than_declared_fails(self):
        (self.root / 'artifacts/revision' / self.records[0]['run_id'] / 'history.json').write_text(json.dumps([{'epoch': 1}]))
        self.assertIn('epochs_completed', self.failing(self.run_check(verify_hashes=False)))
    def test_nan_or_missing_pillar_fails(self):
        for mutate in (lambda m: m['pillars'].__setitem__('uncertainty', float('nan')), lambda m: m['pillars'].pop('representation'),
                       lambda m: m.__setitem__('composite_trust', .99), lambda m: m.__setitem__('ece', None)):
            bad = copy.deepcopy(self.records); mutate(bad[1]['metrics'])
            self.assertIn('metrics_valid', self.failing(self.run_check(bad)))
    def test_tampered_artifact_fails(self):
        (self.root / 'artifacts/revision' / self.records[2]['run_id'] / 'predictions.npz').write_bytes(b'tampered')
        self.assertIn('artifacts_present', self.failing(self.run_check()))
    def test_missing_artifact_fails(self):
        (self.root / 'artifacts/revision' / self.records[0]['run_id'] / 'checkpoint.pt').unlink()
        self.assertIn('artifacts_present', self.failing(self.run_check()))
    def test_different_test_items_fail(self):
        extra = _record('odd', 'ViT-Base', 2, self.root, sha='ViT-Base', manifest=self.manifest, items=['test/A/9.jpg', 'test/A/2.jpg', 'test/B/3.jpg'])
        recs = [r for r in self.records if r['run_id'] != 'ViT-Base2'] + [extra]
        self.assertIn('identical_test_items', self.failing(self.run_check(recs)))
    def test_input_mismatch_fails(self):
        bad = copy.deepcopy(self.records); bad[0]['model']['bands'] = list(PRITHVI_S2_BANDS)
        self.assertIn('input_representation', self.failing(self.run_check(bad)))
    def test_missing_checkpoint_identity_fails(self):
        bad = copy.deepcopy(self.records); bad[0]['model']['checkpoint_sha256'] = None
        self.assertIn('model_identity', self.failing(self.run_check(bad)))
    def test_proxy_or_wrong_prithvi_hash_fails(self):
        bad = copy.deepcopy(self.records); bad[0]['model'].update(architecture='Prithvi-100M')
        self.assertIn('model_identity', self.failing(self.run_check(bad)))
        bad = copy.deepcopy(self.records); bad[0]['model'].update(architecture=PRITHVI, checkpoint_sha256='0' * 64)
        self.assertIn('model_identity', self.failing(self.run_check(bad)))
    def test_superseded_source_excluded_and_reported(self):
        bad = copy.deepcopy(self.records); bad[0]['source_revision'] = 'old'
        failing = self.failing(self.run_check(bad)); self.assertIn('source_current', failing); self.assertIn('matrix_complete', failing)
    def test_failed_runs_are_disclosed_not_aggregated(self):
        failed = copy.deepcopy(self.records[0]); failed['status'] = 'failed'; failed['run_id'] = 'f'
        report = self.run_check(self.records + [failed]); self.assertTrue(report['ok'])
        note = next(c for c in report['checks'] if c['check'] == 'failed_runs_disclosed'); self.assertEqual(note['detail'][0]['run_id'], 'f')
    def test_constant_stability_flagged(self):
        e = self.root / 'artifacts/revision' / self.records[0]['run_id'] / 'explanations.json'; d = json.loads(e.read_text()); d['stability_per_sample'] = [.65] * 3
        e.write_text(json.dumps(d)); self.assertIn('explanation_samples_complete', self.failing(self.run_check(verify_hashes=False)))
    def test_manifest_errors_fail(self):
        self.manifest.write_text(json.dumps({'errors': ['overlap']}))
        self.assertIn('manifest_valid', self.failing(self.run_check(verify_hashes=False)))


class FallbackEliminationTests(unittest.TestCase):
    def test_no_constant_return_in_legacy_trust(self):
        import re
        text = (ROOT / 'src/research/xai/trust.py').read_text()
        self.assertIsNone(re.search(r'return\s+0\.(65|5|50)\b', text))
        self.assertIn('NOT COMPUTED', text)
    def test_unsupported_model_raises_not_constant(self):
        from src.research.xai.trust import calculate_trust_score
        loader = [(torch.zeros(1, 3, 8, 8), torch.zeros(1))]
        with self.assertRaises(ValueError): calculate_trust_score(torch.nn.Linear(2, 2), loader, 'cpu')
    def test_active_pipeline_has_no_silent_exception_to_score(self):
        import re
        for f in (ROOT / 'src/cropxai').rglob('*.py'):
            if 'vendor' in f.parts: continue
            for m in re.finditer(r'except[^\n]*:\s*\n\s*(return|pass)', f.read_text()):
                self.fail(f'{f.name}: exception swallowed near {m.group(0)!r}')


    def test_active_path_has_no_mock_markers(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import scan_legacy_placeholders as scan, re
        bad = [x for x in scan.scan(scan.ACTIVE) if x[2] == 'marker' or re.search(r'np\.random\.', x[3])]
        self.assertEqual(bad, [])


@unittest.skipUnless((ROOT / 'results/experiment_db.sqlite').exists(), 'legacy database unavailable')
class LegacyReconstructionTests(unittest.TestCase):
    def test_submitted_table1_equals_pooled_mean_over_mixed_epochs(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import audit_legacy, io, contextlib
        with contextlib.redirect_stdout(io.StringIO()): audit_legacy.main()
        d = json.loads((ROOT / 'docs/revision/evidence/legacy_audit.json').read_text())
        for model, e in d['table1_reconstruction'].items():
            self.assertEqual(e['epochs_pooled'], [1, 5]); self.assertTrue(e['matches_pooled_to_4dp'], model)
            self.assertGreater(abs(e['accuracy']['five_epoch_only'] - e['accuracy']['submitted']), 5e-3)   # five-epoch-only differs
        self.assertEqual(len(d['fallback_trust_runs']), 5)
        self.assertTrue(all(all(i['identical'][k] for k in ('accuracy', 'f1', 'precision', 'recall')) for i in d['prithvi_vs_vit_identity']))
    def test_same_run_id_diverges_between_databases(self):
        d = json.loads((ROOT / 'docs/revision/evidence/legacy_audit.json').read_text())['database_divergence']
        self.assertTrue(d['run_id_is_config_fingerprint'])
        resnet = next(r for r in d['runs_with_differing_metrics'] if (r['model'], r['epochs'], r['seed']) == ('ResNet-18', 5, 42))
        self.assertTrue(resnet['same_experiment_row'])
        self.assertAlmostEqual(resnet['differing_metrics']['accuracy']['reviewer_package'], 0.8519, places=3)   # Figure 7 band (84-86%)
        self.assertAlmostEqual(resnet['differing_metrics']['accuracy']['main'], 0.9237, places=3)
        self.assertAlmostEqual(resnet['differing_metrics']['final_loss']['reviewer_package'], 0.499, places=3)  # figure's final loss ~0.50
    def test_submitted_baseline_figure_prints_reviewer_database_values(self):
        d = json.loads((ROOT / 'docs/revision/evidence/legacy_audit.json').read_text())['submitted_figure_files']
        self.assertIn('Final Test Accuracy: 85.19%', d['baseline_results']['text'])
        self.assertIn('Final Trust Score (T3): 0.2394', d['baseline_results']['text'])
        self.assertNotIn('92.37', d['baseline_results']['text'])
        self.assertIn('Training Convergence', d['training_curve']['text'])
    def test_reviewer_database_is_subset_of_main(self):
        d = json.loads((ROOT / 'docs/revision/evidence/legacy_audit.json').read_text())
        self.assertEqual(d['reviewer_package_db']['cells_missing_from_main'], [])


class CliTests(unittest.TestCase):
    def run_cli(self, *args): return subprocess.run([sys.executable, str(ROOT / 'cropxai'), *args], capture_output=True, text=True, cwd=ROOT)
    def test_help_and_subcommands(self):
        r = self.run_cli('--help'); self.assertEqual(r.returncode, 0)
        for cmd in ('campaign', 'compliance', 'analyze', 'figures', 'validate-manuscript', 'audit-legacy'): self.assertIn(cmd, r.stdout)
    def test_unknown_command_is_error(self): self.assertNotEqual(self.run_cli('nonsense').returncode, 0)
    def test_campaign_requires_action(self): self.assertNotEqual(self.run_cli('campaign').returncode, 0)
    def test_pyproject_entry_point_targets_active_cli(self):
        self.assertIn('src.cropxai.cli:main', (ROOT / 'pyproject.toml').read_text())


@unittest.skipUnless((ROOT / 'results/figures/manifest.json').exists() and (ROOT / 'results/tables/table1.json').exists(), 'generated assets unavailable')
class GeneratedAssetConsistencyTests(unittest.TestCase):
    """Plotted and tabulated numbers must equal the validated summary (no hand-edited or stale assets)."""
    def setUp(self):
        self.summary = json.loads((ROOT / 'results/processed/revision_summary.json').read_text())
        self.groups = {(g['model']): g for g in self.summary['groups'] if g['current_source'] and g['purpose'] in ('controlled_benchmark', 'multispectral_foundation_benchmark')}
    def test_table1_matches_summary(self):
        for row in json.loads((ROOT / 'results/tables/table1.json').read_text()):
            s = self.groups[row['model']]['metrics'][row['metric']]
            for k in ('mean', 'std', 'ci95', 'n', 'missing_runs'): self.assertEqual(row[k], s[k], (row['model'], row['metric'], k))
    def test_figure_sidecars_match_summary_and_hashes(self):
        manifest = json.loads((ROOT / 'results/figures/manifest.json').read_text())
        self.assertEqual(manifest['source_sha256'], file_hash(ROOT / 'results/processed/revision_summary.json'))
        for f in manifest['files']:
            meta = json.loads((ROOT / f).with_suffix('.json').read_text()); self.assertEqual(meta['figure_sha256'], file_hash(ROOT / f))
        acc = json.loads((ROOT / 'results/figures/accuracy.json').read_text())['values']
        for v in acc: self.assertEqual(v['mean'], self.groups[v['model']]['metrics']['accuracy']['mean'])
        pil = json.loads((ROOT / 'results/figures/pillars.json').read_text())['values']
        for metric, rows in pil.items():
            for v in rows: self.assertEqual(v['mean'], self.groups[v['model']]['metrics'][metric]['mean'])
    def test_every_group_has_all_pillars_and_matching_seed_count(self):
        for m, g in self.groups.items():
            self.assertEqual(g['metrics']['accuracy']['n'], len(g['seeds']), m)
            for p in PILLARS: self.assertEqual(g['metrics'][p]['n'], len(g['seeds']), (m, p))
    def test_single_epoch_setting_inside_primary_cohorts(self):
        self.assertEqual({g['epochs'] for g in self.groups.values()}, {5})


if __name__ == '__main__': unittest.main()
