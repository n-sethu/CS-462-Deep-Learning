"""Numerical, leakage, freezing and depth tests for the new extension modules only."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy.linalg
import torch
from sklearn.covariance import LedoitWolf
from threadpoolctl import threadpool_limits

from src.extension_methods import fit_geometry, raw_distances, fit_normalization, make_features, StageCapture, extract_stages
from src.models import CIFARResNet18
from src.metrics import risk_curve
from src.run_extensions import bootstrap_contrasts, fit, freeze, ExtensionBank, verify_hashes, condition_metrics
from src.validation import sha, output_features


class GeometryTests(unittest.TestCase):
    def test_mahalanobis_matches_non_diagonal_hand_case(self):
        covariance=np.array([[4.,1.],[1.,2.]])
        geometry=dict(centroids=np.zeros((3,10,2)),cholesky=np.repeat(scipy.linalg.cholesky(covariance,lower=True)[None],3,axis=0))
        x=np.repeat(np.array([[[2.,3.]]]),3,axis=1)
        expected=np.sqrt(np.array([2.,3.])@np.linalg.solve(covariance,np.array([2.,3.])))
        np.testing.assert_allclose(raw_distances(x,np.array([4]),geometry),expected)

    def test_identity_covariance_equals_euclidean_and_predicted_class_is_used(self):
        rng=np.random.default_rng(1)
        x=rng.normal(size=(5,3,4))
        centroids=rng.normal(size=(3,10,4))
        predicted=np.array([9,0,3,6,1])
        euclidean=raw_distances(x,predicted,dict(centroids=centroids))
        mahalanobis=raw_distances(x,predicted,dict(centroids=centroids,cholesky=np.repeat(np.eye(4)[None],3,axis=0)))
        np.testing.assert_allclose(mahalanobis,euclidean)
        self.assertFalse(np.allclose(euclidean,raw_distances(x,(predicted+1)%10,dict(centroids=centroids))))

    def test_covariance_uses_class_centered_training_residuals(self):
        rng=np.random.default_rng(42)
        labels=np.repeat(np.arange(10),8)
        x=rng.normal(size=(80,3,4))+labels[:,None,None]*30
        with threadpool_limits(2):
            geometry=fit_geometry(x,labels,mahalanobis=True)
            for m in range(3):
                residual=x[:,m]-geometry['centroids'][m,labels]
                expected=LedoitWolf(assume_centered=True,store_precision=False).fit(residual)
                np.testing.assert_allclose(geometry['covariance'][m],expected.covariance_)
                np.testing.assert_allclose(geometry['cholesky'][m]@geometry['cholesky'][m].T,expected.covariance_)

    def test_detector_reference_and_no_evaluation_labels(self):
        rng=np.random.default_rng(7)
        x=rng.normal(size=(12,3,2))
        logits=rng.normal(size=(12,3,10))
        geometry=dict(centroids=rng.normal(size=(3,10,2)))
        reference=fit_normalization(x,logits,geometry)
        raw=raw_distances(x,output_features(logits)['prediction'],geometry)
        np.testing.assert_allclose(reference['mean'],raw.mean(0))
        np.testing.assert_allclose(reference['std'],raw.std(0))
        table=make_features(np.arange(12),logits,x,reference)
        self.assertNotIn('label',table)
        self.assertNotIn('error',table)
        np.testing.assert_allclose(table.mean_novelty.mean(),0,atol=1e-12)
        with self.assertRaises(TypeError):
            make_features(np.arange(12),logits,x,reference,labels=np.zeros(12))

    def test_non_positive_covariance_does_not_fall_back(self):
        with self.assertRaises(np.linalg.LinAlgError):
            fit_geometry(np.zeros((20,3,2)),np.tile(np.arange(10),2),mahalanobis=True)


class DepthTests(unittest.TestCase):
    def test_temporary_hooks_preserve_logits_and_shapes(self):
        torch.set_num_threads(2)
        torch.manual_seed(3)
        model=CIFARResNet18().eval()
        images=torch.randn(2,3,32,32)
        with torch.inference_mode():
            before,features=model(images,return_features=True)
            hooks=StageCapture(model)
            after=model(images)
            torch.testing.assert_close(before,after,atol=0,rtol=0)
            for stage,dim in [('stage1',64),('stage2',128),('stage4',512)]:
                self.assertEqual(hooks.values[stage].shape,(2,dim))
            torch.testing.assert_close(features,hooks.values['stage4'])
            hooks.close()
        for layer in (model.network.layer1,model.network.layer2,model.network.layer4):
            self.assertEqual(len(layer._forward_hooks),0)

    def test_extraction_rejects_wrong_logits_and_cleans_hooks(self):
        torch.set_num_threads(2)
        models=[CIFARResNet18().eval() for _ in range(3)]
        images=torch.randn(2,3,32,32)
        labels=torch.tensor([0,1])
        dataset=torch.utils.data.TensorDataset(images,labels)
        original=dict(id=np.arange(2),label=labels.numpy(),logits=np.zeros((2,3,10)),features=np.zeros((2,3,512)))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AssertionError):
                extract_stages(dataset,np.arange(2),models,'cpu',original,Path(directory)/'new.npz','hash')
            self.assertFalse((Path(directory)/'new.npz').exists())
        for model in models:
            self.assertEqual(len(model.network.layer1._forward_hooks),0)


class BootstrapTests(unittest.TestCase):
    def test_identical_measurements_have_zero_interaction(self):
        scores=dict(b=np.array([.2,.8,.1,.6]),c=np.array([.1,.9,.3,.7]))
        result=bootstrap_contrasts([0,1,0,1],scores,np.arange(4),dict(interaction={'b':0.,'c':0.}),resamples=40)
        self.assertEqual(result[0]['delta_aurc'],0)
        self.assertEqual(result[0]['ci_low'],0)
        self.assertEqual(result[0]['ci_high'],0)

    def test_cluster_sampling_keeps_copies_together(self):
        errors=np.array([0,1,1,0,0,1])
        a=np.arange(6)
        b=a[::-1]
        groups=np.tile([0,1],3)
        result=bootstrap_contrasts(errors,dict(a=a,b=b),np.arange(6),dict(diff={'a':1,'b':-1}),resamples=50,seed=4,groups=groups)[0]
        rng=np.random.default_rng(4)
        values=[]
        for _ in range(50):
            draws=rng.integers(2,size=2)
            ix=np.concatenate([np.flatnonzero(groups==g) for g in draws])
            values.append(risk_curve(errors[ix],a[ix],ix).mean()-risk_curve(errors[ix],b[ix],ix).mean())
        np.testing.assert_allclose([result['ci_low'],result['ci_high']],np.quantile(values,[.025,.975]))

    def test_paired_linear_contrast_uses_same_draws(self):
        scores=dict(a=np.array([.4,.1,.3,.2]),b=np.array([.2,.4,.1,.3]))
        results=bootstrap_contrasts([0,1,0,1],scores,np.arange(4),dict(forward={'a':1,'b':-1},reverse={'a':-1,'b':1}),resamples=50)
        self.assertAlmostEqual(results[0]['ci_low'],-results[1]['ci_high'])
        self.assertAlmostEqual(results[0]['ci_high'],-results[1]['ci_low'])


class SelectionTests(unittest.TestCase):
    def test_scalers_reload_and_fit_disabled_after_freeze(self):
        torch.set_num_threads(2)
        rng=np.random.default_rng(9)
        def frame(n,shift):
            x=rng.normal(size=(n,7))+shift
            return pd.DataFrame(dict(id=np.arange(n),error=np.tile([0,1],n//2),confidence=x[:,0],entropy=x[:,1],
                margin=x[:,2],mutual_information=x[:,3],mean_novelty=x[:,4],novelty_variation=x[:,5],max_novelty=x[:,6]))
        def fast_mlp(x,y,xv,yv,ids,seed,epochs):
            torch.manual_seed(seed)
            model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,1)).eval()
            with torch.inference_mode():score=model(torch.tensor(xv,dtype=torch.float32)).sigmoid().numpy().ravel()
            return model.state_dict(),score,1
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            study=root/'h7'
            dest=study/'mahalanobis'
            dest.mkdir(parents=True)
            dt,vt=frame(40,0),frame(20,10)
            dt.to_csv(dest/'detector_train_features.csv',index=False)
            vt.to_csv(dest/'val_features.csv',index=False)
            (study/'registration.json').write_text('{}')
            (study/'prepared.json').write_text(json.dumps(dict(files={})))
            (root/'original_files.json').write_text('{}')
            with patch('src.run_extensions.check_registration',return_value={}),patch('src.run_extensions.fit_mlp',side_effect=fast_mlp),threadpool_limits(2):
                fit(root,'h7')
                bank=ExtensionBank(dest)
                expected=pd.read_csv(dest/'validation_scores.csv')
                for name,score in bank.predict(vt).items():
                    np.testing.assert_allclose(score,expected[name],atol=2e-7,rtol=2e-6)
                np.testing.assert_allclose(bank.lr['B']['scaler'].mean_,dt[bank.lr['B']['features']].mean(0))
                freeze(root,'h7')
                with self.assertRaisesRegex(ValueError,'frozen'):
                    fit(root,'h7')

    def test_changed_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'artifact'
            path.write_bytes(b'frozen')
            files={str(path):sha(path)}
            verify_hashes(files)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                verify_hashes(files)

    def test_small_subgroup_keeps_counts_but_suppresses_auroc(self):
        frame=pd.DataFrame(dict(id=np.arange(4),error=[0,1,0,1],unanimous=[True]*4,confidence=[.95]*4))
        rows,_=condition_metrics('toy','toy',frame,dict(B_lr=np.array([.1,.9,.2,.8])),{})
        self.assertEqual(rows[-1]['errors'],2)
        self.assertIsNone(rows[-1]['auroc'])
        self.assertIsNotNone(rows[-1]['aurc'])


if __name__=='__main__':
    unittest.main()
