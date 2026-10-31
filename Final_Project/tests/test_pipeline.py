"""Offline checks: no CIFAR test data, downloads, or GPU required."""
import tempfile
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from src.data import make_loaders, split_indices
from src.models import CIFARResNet18
from src.train import run_epoch, seed_everything


class PipelineTests(unittest.TestCase):
    def test_splits_are_saved_stratified_and_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'splits.npz'
            targets = np.repeat(np.arange(10), 5000)
            original = split_indices(targets, path)
            loaded = split_indices(targets, path)
            for key in original:
                np.testing.assert_array_equal(original[key], loaded[key])
            with self.assertRaises(ValueError):
                split_indices(targets, path, seed=99)

    def test_only_training_partition_is_opened(self):
        class FakeCIFAR:
            def __init__(self, root, train, download, transform):
                self.targets = np.repeat(np.arange(10), 5000)
                self.transform = transform
                self.train = train

            def __len__(self):
                return len(self.targets)

            def __getitem__(self, index):
                return self.transform(Image.new('RGB', (32, 32))), int(self.targets[index])

        with tempfile.TemporaryDirectory() as directory, patch('src.data.datasets.CIFAR10', side_effect=FakeCIFAR) as factory:
            config = dict(data_dir=directory, split_path=f'{directory}/splits.npz',
                          split_seed=42, seed=1, batch_size=4, num_workers=0)
            train, val = make_loaders(config)
            self.assertTrue(all(call.kwargs['train'] for call in factory.call_args_list))
            self.assertEqual(len(train.dataset), 35000)
            self.assertEqual(len(val.dataset), 5000)
            self.assertIn('RandomCrop', repr(train.dataset.dataset.transform))
            self.assertNotIn('RandomCrop', repr(val.dataset.dataset.transform))
            a, _ = val.dataset[0]
            b, _ = val.dataset[0]
            torch.testing.assert_close(a, b)
            self.assertEqual(next(iter(train))[0].shape, (4, 3, 32, 32))

    def test_cli_writes_metrics_checkpoint_and_curves(self):
        from src.train import main
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(Path('configs/ensemble.json').read_text())
            config.update(epochs=1, device='cpu', output_dir=directory)
            config_path = Path(directory) / 'input.json'
            config_path.write_text(json.dumps(config))
            loader = DataLoader(TensorDataset(torch.randn(2, 3, 32, 32), torch.tensor([0, 1])), batch_size=2)
            with patch('sys.argv', ['train', '--config', str(config_path)]), patch('src.train.make_loaders', return_value=(loader, loader)):
                main()
            output = Path(directory) / 'seed_1'
            for name in ('best.pt', 'metrics.csv', 'curves.png', 'best_metrics.json', 'config.json', 'environment.json'):
                self.assertGreater((output / name).stat().st_size, 0)
            checkpoint = torch.load(output / 'best.pt', weights_only=False)
            self.assertEqual(checkpoint['epoch'], 1)
            self.assertEqual(checkpoint['config']['split_seed'], 42)
            with patch('sys.argv', ['train', '--config', str(config_path)]):
                with self.assertRaises(FileExistsError):
                    main()

    def test_training_evaluation_features_and_checkpoint(self):
        torch.set_num_threads(2)
        seed_everything(1)
        model = CIFARResNet18()
        images = torch.randn(2, 3, 32, 32)
        loader = DataLoader(TensorDataset(images, torch.tensor([1, 2])), batch_size=2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.network.fc.weight.detach().clone()
        loss, _ = run_epoch(model, loader, torch.nn.CrossEntropyLoss(), 'cpu', optimizer)
        self.assertTrue(np.isfinite(loss))
        self.assertFalse(torch.equal(before, model.network.fc.weight))
        run_epoch(model, loader, torch.nn.CrossEntropyLoss(), 'cpu')
        with torch.no_grad():
            logits, features = model(images, return_features=True)
        self.assertEqual(logits.shape, (2, 10))
        self.assertEqual(features.shape, (2, 512))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(model.state_dict(), path)
            restored = CIFARResNet18().eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                torch.testing.assert_close(logits, restored(images))


if __name__ == '__main__':
    unittest.main()
