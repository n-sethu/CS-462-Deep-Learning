"""Adaptation split, selection, freezing, and paired-contrast regression checks."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from src.adaptation import (noise_image, validation_objective, contrast_statistics,
                            bootstrap, fit, AdaptedBank, fit_mixture_mlp)
from src.validation import FEATURES


class AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads(Path('configs/adaptation.json').read_text())
        torch.set_num_threads(2)

    def test_noise_matches_formula_and_is_order_independent(self):
        c=self.config
        image=np.full((32,32,3),128,dtype=np.uint8)
        for severity in (1,3,5):
            rng=np.random.default_rng(np.random.SeedSequence([c['corruption_seed'],1,7,severity]))
            expected=(255*np.clip(image/255.+rng.normal(scale=c['gaussian_sigma'][str(severity)],size=image.shape),0,1)).astype(np.uint8)
            np.testing.assert_array_equal(noise_image(image,severity,7,1,c),expected)
        a=noise_image(image,3,7,1,c)
        noise_image(image,3,99,1,c)
        np.testing.assert_array_equal(a,noise_image(image,3,7,1,c))
        self.assertFalse(np.array_equal(a,noise_image(image,3,7,2,c)))

    def test_selection_uses_mean_condition_aurc_not_pooled(self):
        table=pd.DataFrame(dict(id=[0,1,0,1],error=[0,1,0,1],condition=['a','a','b','b']))
        score=np.array([.8,.9,.1,.2])
        self.assertEqual(validation_objective(table,score,['a','b']),.25)
        from src.metrics import metrics
        self.assertNotEqual(validation_objective(table,score,['a','b']),metrics(table.error,score,np.arange(4))['aurc'])

    def test_gamma_and_bootstrap_use_same_four_draws(self):
        errors=np.array([0,1,0,1,0,1])
        a=np.array([.1,.8,.3,.7,.5,.6]); b=a[::-1].copy()
        scores=dict(original_A=a,original_B=b,adapted_A=a,adapted_B=b)
        c=dict(self.config,bootstrap_resamples=20)
        rows=bootstrap(errors,scores,np.arange(6),c)
        gamma=next(r for r in rows if r['comparison']=='Gamma')
        self.assertEqual([gamma[k] for k in ['estimate','ci_low','ci_high']],[0,0,0])
        stats=contrast_statistics(errors,scores,np.arange(6),[.8,.9])
        self.assertEqual(stats['Gamma'],0)

    def test_scalers_train_only_selection_reload_and_freeze(self):
        rng=np.random.default_rng(5)
        def frame(ids,shift):
            frames=[]
            for condition in self.config['development_conditions']:
                x=rng.normal(size=(len(ids),5))+shift
                frames.append(pd.DataFrame(dict(id=ids,label=np.zeros(len(ids),int),prediction=np.tile([0,1],len(ids)//2),
                    error=np.tile([0,1],len(ids)//2),condition=condition,**{name:x[:,i] for i,name in enumerate(FEATURES['B'])})))
            return pd.concat(frames,ignore_index=True)
        dt,vt=frame(np.arange(8),0),frame(np.arange(20,28),10)
        c=dict(self.config,epochs=2,patience=2)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'registration.json').write_text('{}')
            (root/'prepared.json').write_text(json.dumps(dict(files={})))
            with patch('src.adaptation.check',return_value=c),patch('src.adaptation.development',side_effect=[dt,vt]):
                fit(root)
            bank=AdaptedBank(root)
            expected=pd.read_csv(root/'validation_scores.csv')
            for name,value in bank.predict(vt).items():
                np.testing.assert_allclose(value,expected[name],atol=1e-7)
            import joblib
            artifact=joblib.load(root/'detectors/B_lr.joblib')
            np.testing.assert_allclose(artifact['scaler'].mean_,dt[FEATURES['B']].mean())
            with patch('src.adaptation.check',return_value=c),patch('src.adaptation.development',side_effect=AssertionError('Frozen fit must not read data')):
                fit(root)
            checkpoint=root/'detectors/A_mlp_11.pt'
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Artifact changed'): AdaptedBank(root)

    def test_source_overlap_is_rejected(self):
        table=pd.DataFrame(dict(id=[1,2]))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); (root/'prepared.json').write_text('{"files":{}}')
            with patch('src.adaptation.check',return_value=self.config),patch('src.adaptation.development',return_value=table):
                with self.assertRaisesRegex(ValueError,'source leakage'): fit(root)


if __name__=='__main__': unittest.main()
