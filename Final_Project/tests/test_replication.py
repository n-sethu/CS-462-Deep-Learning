"""Replication isolation, seed, retention and aggregation checks."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.replicate import initialize,aggregate,completed


class ReplicationTests(unittest.TestCase):
    def test_new_seeds_are_unique_and_architecture_fixed(self):
        c=json.loads(Path('configs/replication.json').read_text())
        seeds=[s for e in c['member_seeds'] for s in e]
        self.assertEqual(len(c['member_seeds']),10)
        self.assertTrue(all(len(e)==3 for e in c['member_seeds']))
        self.assertEqual(len(set(seeds)),30)
        self.assertFalse(set(seeds)&{1,2,3})
        self.assertIn('unchanged',c['architecture'])

    def test_workspace_creation_and_snapshot_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder); repo=base/'repo'; repo.mkdir()
            for name in ['src','configs','splits','data/CIFAR-10-C']:(repo/name).mkdir(parents=True,exist_ok=True)
            (repo/'src/a.py').write_text('x=1')
            (repo/'splits/cifar10_seed42.npz').write_bytes(b'saved split')
            (repo/'data/CIFAR-10-C/gaussian_noise.npy').write_bytes(b'data')
            (repo/'configs/replication.json').write_text(Path('configs/replication.json').read_text())
            (repo/'configs/ensemble.json').write_text(Path('configs/ensemble.json').read_text())
            root=base/'runs'; initialize(repo,root)
            self.assertTrue((root/'ensemble_01/data').is_symlink())
            a=json.loads((root/'ensemble_01/configs/ensemble.json').read_text())
            b=json.loads((root/'ensemble_02/configs/ensemble.json').read_text())
            self.assertNotEqual(a['output_dir'],b['output_dir'])
            self.assertEqual(a['split_path'],b['split_path'])
            (repo/'src/a.py').write_text('x=2')
            self.assertEqual((root/'snapshot/src/a.py').read_text(),'x=1')
            initialize(repo,root)
            (root/'snapshot/src/a.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'Snapshot changed'):initialize(repo,root)

    def test_training_preserves_actual_seeds_in_compatible_slots(self):
        import torch
        from src.replicate import train
        with tempfile.TemporaryDirectory() as folder:
            work=Path(folder)
            (work/'identity.json').write_text(json.dumps(dict(member_seeds=[1011,1012,1013])))
            def fake_main():
                import sys
                seed=int(sys.argv[sys.argv.index('--seed')+1])
                path=work/'training'/f'seed_{seed}'; path.mkdir(parents=True)
                torch.save(dict(model_state_dict={'weight':torch.tensor([seed])},optimizer_state_dict={},scheduler_state_dict={},config=dict(seed=seed)),path/'best.pt')
            with patch('src.train.main',side_effect=fake_main): train(work)
            for slot,seed in enumerate([1011,1012,1013],1):
                saved=torch.load(work/f'training/seed_{slot}/best.pt',weights_only=False)
                self.assertEqual(saved['config']['seed'],seed)
                self.assertNotIn('optimizer_state_dict',saved)
                self.assertEqual(saved['model_state_dict']['weight'].item(),seed)
            with patch('src.train.main',side_effect=AssertionError('Must reuse completed members')): train(work)

    def test_aggregate_keeps_every_ensemble_and_excludes_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for i,effect in [(1,-.01),(2,.02)]:
                work=root/f'ensemble_{i:02d}'; final=work/'results/final'; final.mkdir(parents=True)
                with (final/'clean_bootstrap.csv').open('w') as f:
                    f.write('family,comparison,delta_aurc,ci_low,ci_high\n')
                    f.write(f'mlp_selected,B_minus_A,{effect},-.03,.03\n')
                    f.write(f'mlp_selected,C_minus_B,{-effect},-.03,.03\n')
                (work/'compact_complete.json').write_text('{}')
            aggregate(root)
            with (root/'ensemble_summary.csv').open() as f: rows=list(csv.DictReader(f))
            b=next(r for r in rows if r['comparison']=='B_minus_A')
            self.assertEqual(int(b['completed_ensembles']),2)
            self.assertAlmostEqual(float(b['mean']),.005)
            self.assertEqual(float(b['fraction_negative']),.5)
            with (root/'status.csv').open() as f: self.assertEqual(len(list(csv.DictReader(f))),10)


if __name__=='__main__':unittest.main()
