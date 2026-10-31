"""CIFAR-10 training data; the official test split is never instantiated here."""
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def split_indices(targets, path, seed=42):
    """Persist and validate the fixed 35k/10k/5k stratified partition."""
    targets = np.asarray(targets, dtype=np.int64)
    if len(targets) != 50000:
        raise ValueError("Expected the 50,000 official CIFAR-10 training examples")
    path = Path(path)
    digest = hashlib.sha256(targets.tobytes()).hexdigest()
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if int(saved['seed']) != seed or str(saved['targets_sha256']) != digest:
                raise ValueError("Saved splits do not match the split seed or dataset; do not overwrite them")
            splits = {name: saved[name].copy() for name in ('train', 'detector_train', 'val')}
    else:
        train, heldout = train_test_split(np.arange(len(targets)), test_size=15000,
                                         stratify=targets, random_state=seed)
        detector, val = train_test_split(heldout, test_size=5000,
                                        stratify=targets[heldout], random_state=seed)
        splits = dict(train=train, detector_train=detector, val=val)
    joined = np.concatenate(list(splits.values()))
    if not np.array_equal(np.sort(joined), np.arange(50000)):
        raise ValueError("Splits must be disjoint and cover the training set exactly")
    for name, size in [('train', 35000), ('detector_train', 10000), ('val', 5000)]:
        indices = splits[name]
        if len(indices) != size or not np.array_equal(np.bincount(targets[indices], minlength=10), np.full(10, size // 10)):
            raise ValueError(f"Invalid size or class balance for {name}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents concurrent ensemble jobs from overwriting splits.
        with path.open('xb') as handle:
            np.savez(handle, **splits, seed=seed, targets_sha256=digest)
    return splits


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loaders(config, download=True):
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    augmentation = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    training = datasets.CIFAR10(config['data_dir'], train=True, download=download, transform=augmentation)
    validation = datasets.CIFAR10(config['data_dir'], train=True, download=False, transform=evaluation)
    indices = split_indices(training.targets, config['split_path'], config['split_seed'])
    options = dict(batch_size=config['batch_size'], num_workers=config['num_workers'],
                   worker_init_fn=seed_worker, pin_memory=torch.cuda.is_available())
    train = DataLoader(Subset(training, indices['train'].tolist()), shuffle=True,
                       generator=torch.Generator().manual_seed(config['seed']), **options)
    val = DataLoader(Subset(validation, indices['val'].tolist()), shuffle=False,
                     generator=torch.Generator().manual_seed(config['seed'] + 1), **options)
    return train, val
