"""Deterministic scientific tests; unittest requires no extra installation."""
import unittest, tempfile, sqlite3, json
from pathlib import Path
import numpy as np
import torch
from src.cropxai.metrics import prediction_metrics, calibration_bins, stability, representation_quality, composite, PILLARS, sensitivity, mean_ci, pillar_scores
from src.cropxai.data import validate_bands, deterministic_split, stratified_indices, audit_dataset, loaders
from src.cropxai.explainability import deletion, corrupt, attribution
from src.cropxai.provenance import digest, append_record, write_json
from src.cropxai.models import create_model, freeze_backbone

class MetricsTests(unittest.TestCase):
    def test_perfect(self):
        m=prediction_metrics(np.eye(3),np.arange(3))
        for k in ['accuracy','macro_f1','weighted_f1']: self.assertEqual(m[k],1)
        for k in ['ece','nll','brier','normalized_entropy']: self.assertEqual(m[k],0)
        self.assertIsNone(m['error_detection_auroc'])
    def test_uniform(self):
        m=prediction_metrics(np.ones((4,2))*.5,np.array([0,1,0,1]))
        self.assertAlmostEqual(m['nll'],np.log(2)); self.assertAlmostEqual(m['brier'],.5)
        self.assertEqual(m['ece'],0); self.assertAlmostEqual(m['normalized_entropy'],1)
    def test_ece_boundary_and_empty_bins(self):
        p=np.array([[1,0],[.5,.5],[0,1]]); y=np.array([0,1,1])
        b=calibration_bins(p,y,2); self.assertEqual(sum(r['n'] for r in b),3)
        self.assertIsNone(b[0]['accuracy']); self.assertAlmostEqual(prediction_metrics(p,y,2)['ece'],1/6)
    def test_invalid_probabilities(self):
        for p in [np.array([[.2,.2]]),np.array([[np.nan,1]]),np.array([[-1,2]])]:
            with self.assertRaises(ValueError): prediction_metrics(p,np.array([0]))
        with self.assertRaises(ValueError): prediction_metrics(np.eye(2),np.array([0,2]))
    def test_macro_includes_missing_classes(self):
        m=prediction_metrics(np.array([[1,0,0],[1,0,0]]),np.array([0,0]))
        self.assertAlmostEqual(m['macro_f1'],1/3)
    def test_nll_zero_true_probability(self):
        self.assertTrue(np.isfinite(prediction_metrics(np.array([[1,0]]),np.array([1]))['nll']))
    def test_stability(self):
        a=np.arange(64,dtype=float).reshape(8,8)/63
        self.assertAlmostEqual(stability(a,a),1)
        self.assertLess(stability(a,np.flipud(a)),1)
        with self.assertRaises(ValueError): stability(np.zeros((8,8)),np.zeros((8,8)))
    def test_representation(self):
        x=np.array([[1,.01],[1,-.01],[-1,.01],[-1,-.01]])
        self.assertGreater(representation_quality(x,[0,0,1,1])['representation'],.9)
        with self.assertRaises(ValueError): representation_quality(x,[0,0,0,0])
        with self.assertRaises(ValueError): representation_quality(np.ones((4,2)),[0,0,1,1])
    def test_score(self):
        w={k:1/6 for k in PILLARS}; s={k:.5 for k in PILLARS}
        self.assertAlmostEqual(composite(s,w),.5)
        for bad in [{**s,'uncertainty':None},{**s,'calibration':2},{**s,'representation':np.nan}]:
            with self.assertRaises(ValueError): composite(bad,w)
        del s['robustness']
        with self.assertRaises(ValueError): composite(s,w)
        with self.assertRaises(ValueError): composite({k:.5 for k in PILLARS},{k:.2 for k in PILLARS})
    def test_pillar_normalization(self):
        scores=pillar_scores({'macro_f1':.8,'deletion_auc':.3,'ece':.1,'brier':.4},[.6,.8],.7)
        self.assertAlmostEqual(scores['uncertainty'],.8); self.assertAlmostEqual(scores['robustness'],.7)
        with self.assertRaises(ValueError): pillar_scores({},[],.5)
    def test_ci(self):
        result=mean_ci([1,2,3,4,5],expected_runs=6)
        self.assertEqual(result['mean'],3); self.assertAlmostEqual(result['std'],np.sqrt(2.5)); self.assertEqual(result['missing_runs'],1)
        self.assertAlmostEqual(result['ci95'][0],1.0367568385,places=8)
        self.assertIsNone(mean_ci([1])['ci95']); self.assertIsNone(mean_ci([])['mean'])
        self.assertEqual(mean_ci([.5,.5])['ci95'],[.5,.5])
    def test_sensitivity(self):
        w={k:1/6 for k in PILLARS}; a={k:.5 for k in PILLARS}; b=dict(a); a['performance']=1; b['explanation']=1
        schemes={'default':w,'performance':{k:float(k=='performance') for k in PILLARS},'explanation':{k:float(k=='explanation') for k in PILLARS}}
        r=sensitivity({'a':a,'b':b},schemes,2)
        self.assertEqual(r[1]['preferred_models'],['a']); self.assertEqual(r[2]['preferred_models'],['b'])
        self.assertEqual(r,sensitivity({'a':a,'b':b},schemes,2))

class ExplanationTests(unittest.TestCase):
    class KnownModel(torch.nn.Module):
        def forward(self,x):
            score=x[:,0,0,0]*4
            return torch.stack([score,-score],1)
    def test_deletion_known_signal(self):
        m=self.KnownModel(); x=torch.ones(2,3,4,4); maps=torch.zeros(2,4,4); maps[:,0,0]=1
        target=torch.zeros(2,dtype=torch.long)
        high=deletion(m,x,maps,target,16,1,'salient'); low=deletion(m,x,maps,target,16,1,'least')
        self.assertLess(high['auc'],low['auc']); self.assertEqual(high['curves'][0][-1],.5)
        self.assertEqual(high['fractions'][0],0); self.assertEqual(high['fractions'][-1],1)
        self.assertEqual(len(high['curves']),2)
    def test_attribution_known_pixel(self):
        a=attribution(self.KnownModel(),torch.ones(1,3,4,4),torch.tensor([0]))
        self.assertEqual(a[0,0,0],1); self.assertEqual(a.sum(),1)
    def test_corruptions(self):
        x=torch.zeros(2,3,4,4)
        self.assertTrue(torch.allclose(corrupt(x,'gaussian_noise',0),x,atol=1e-6))
        a=corrupt(x,'gaussian_noise',.1,torch.Generator().manual_seed(42))
        b=corrupt(x,'gaussian_noise',.1,torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(a,b)); self.assertFalse(torch.equal(a,x))
        self.assertTrue(torch.all(corrupt(x,'brightness',.5)<x))
        with self.assertRaises(ValueError): corrupt(x,'not-real',.1)

class DataTests(unittest.TestCase):
    def test_bands(self):
        validate_bands(['B04','B03','B02'],3)
        with self.assertRaises(ValueError): validate_bands(['B02','B03','B04'],3)
        with self.assertRaises(ValueError): validate_bands(['B04','B03','B02'],6)
    def test_split(self):
        items=[str(i) for i in range(100)]; a=deterministic_split(items); b=deterministic_split(items[::-1])
        self.assertEqual(a,b); self.assertEqual([len(a[k]) for k in ('train','val','test')],[70,15,15])
        self.assertFalse(set(a['train'])&set(a['test'])); self.assertEqual(len(set(sum(a.values(),[]))),100)
    def test_stratified_selection(self):
        labels=np.repeat(range(3),10); idx=stratified_indices(labels,2)
        self.assertEqual(np.bincount(labels[idx]).tolist(),[2,2,2]); self.assertEqual(idx,stratified_indices(labels,2))
    def test_leakage_detection(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for split in ('train','val','test'):
                for c in ('a','b'):
                    p=root/split/c; p.mkdir(parents=True)
                    Image.new('RGB',(64,64),color='red' if c=='a' else 'blue').save(p/'1.jpg')
            r=audit_dataset(root)
            self.assertTrue(r['cross_split_duplicates']); self.assertTrue(r['errors'])

class ProvenanceTests(unittest.TestCase):
    def test_additive_immutable_store(self):
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'test.sqlite'
            with sqlite3.connect(db) as c: c.execute('CREATE TABLE legacy (value INTEGER)'); c.execute('INSERT INTO legacy VALUES (445)')
            r={'schema_version':2,'run_id':'abc','status':'completed','timestamp':'now','config':{},'config_hash':digest({}),'source_revision':'x','dataset':{},'model':{},'seed':42,'environment':{},'artifacts':{},'metrics':{}}
            append_record(r,db)
            with self.assertRaises(sqlite3.IntegrityError): append_record(r,db)
            with sqlite3.connect(db) as c: self.assertEqual(c.execute('SELECT value FROM legacy').fetchone()[0],445)
            write_json(Path(temp)/'record.json',r); self.assertEqual(json.loads((Path(temp)/'record.json').read_text()),r)
            r['metrics']={'a':float('nan')}
            with self.assertRaises(ValueError): append_record(r,db)
    def test_configuration_hash(self): self.assertEqual(digest({'b':2,'a':1}),digest({'a':1,'b':2}))

class ModelTests(unittest.TestCase):
    def test_proxy_rejected(self):
        with self.assertRaises(ValueError): create_model('Prithvi-100M',10,False)
    def test_cpu_shape_and_freeze(self):
        torch.set_num_threads(2); m,meta=create_model('ResNet-18',10,False); freeze_backbone(m)
        before=m.bn1.running_mean.clone()
        with torch.no_grad(): out=m(torch.zeros(2,3,224,224))
        self.assertEqual(out.shape,(2,10)); self.assertTrue(torch.equal(before,m.bn1.running_mean))
        self.assertTrue(all(not p.requires_grad for n,p in m.named_parameters() if not n.startswith('fc.')))

if __name__=='__main__': unittest.main()

class FeatureCacheTests(unittest.TestCase):
    def test_cached_matches_online_and_detects_corruption(self):
        import os
        from src.cropxai.feature_cache import cached_features,head_predict
        from torch.utils.data import DataLoader,TensorDataset
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.fc=torch.nn.Linear(3,2)
            def forward(self,x): return self.fc(x.mean((2,3)))
        model=Tiny().eval(); x=torch.rand(6,3,4,4); y=torch.arange(6)%2; loader=DataLoader(TensorDataset(x,y),batch_size=2)
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            try:
                f,labels,meta=cached_features(model,loader,'cpu',{'test':1})
                expected=model(x).softmax(1).detach().numpy()
                np.testing.assert_allclose(head_predict(model,f,'cpu'),expected,rtol=1e-6,atol=1e-7)
                f2,_,_=cached_features(model,loader,'cpu',{'test':1}); np.testing.assert_array_equal(f,f2)
                npz=Path(meta).with_suffix('.npz'); npz.write_bytes(b'corrupted')
                with self.assertRaises(ValueError): cached_features(model,loader,'cpu',{'test':1})
            finally: os.chdir(original)

class RealDataIntegration(unittest.TestCase):
    @unittest.skipUnless(Path('data/processed/eurosat/train').exists(),'EuroSAT unavailable')
    def test_real_input_shapes_and_deterministic_evaluation(self):
        ld,mapping,selected=loaders('data/processed/eurosat',batch_size=10,per_class={'train':1,'val':1,'test':1},training=False)
        x,y=next(iter(ld['test'])); x2,y2=next(iter(ld['test']))
        self.assertEqual(tuple(x.shape),(10,3,224,224)); self.assertEqual(len(mapping),10)
        self.assertTrue(torch.equal(x,x2)); self.assertTrue(torch.equal(y,y2))
        self.assertEqual(len(set(selected['test'])),10)
