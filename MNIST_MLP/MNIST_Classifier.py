"""Hyperparameter sweep for a high-accuracy MNIST multilayer perceptron.

Install dependencies with: python -m pip install torch torchvision
Run a small smoke test with: python MNIST_Classifier.py --smoke-test
Run the sweep with: python MNIST_Classifier.py --trials 8 --epochs 25
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
	import torch
	from torch import nn
	from torch.utils.data import DataLoader
	from torchvision import datasets, transforms
except ImportError as error:
	raise SystemExit(
		"This script requires PyTorch and torchvision. "
		"Install them with: python -m pip install torch torchvision"
	) from error


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DEFAULT_SEED = 2026


@dataclass(frozen=True)
class TrialConfig:
	hidden_sizes: tuple[int, ...]
	activation: str
	dropout: float
	weight_decay: float
	learning_rate: float
	label_smoothing: float
	mixup_alpha: float
	augmentation: bool


TRIALS = [
	TrialConfig((512, 256, 128), "gelu", 0.10, 1e-4, 1.5e-3, 0.05, 0.20, True),
	TrialConfig((512, 256, 128), "relu", 0.15, 3e-4, 1e-3, 0.05, 0.20, True),
	TrialConfig((768, 384, 192), "gelu", 0.10, 1e-4, 1e-3, 0.05, 0.00, True),
	TrialConfig((512, 256), "silu", 0.10, 3e-4, 1.5e-3, 0.00, 0.20, True),
	TrialConfig((768, 384, 192), "relu", 0.20, 1e-4, 7.5e-4, 0.05, 0.20, True),
	TrialConfig((512, 256, 128), "silu", 0.05, 1e-5, 1e-3, 0.05, 0.00, False),
	TrialConfig((1024, 512, 256), "gelu", 0.15, 3e-4, 7.5e-4, 0.05, 0.20, True),
	TrialConfig((512, 256, 128), "relu", 0.05, 1e-5, 1.5e-3, 0.00, 0.00, False),
]


def set_seed(seed: int) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def activation(name: str) -> nn.Module:
	activations: dict[str, nn.Module] = {
		"relu": nn.ReLU(),
		"gelu": nn.GELU(),
		"silu": nn.SiLU(),
	}
	return activations[name]


class MNISTMLP(nn.Module):
	def __init__(self, config: TrialConfig) -> None:
		super().__init__()
		layers: list[nn.Module] = [nn.Flatten()]
		input_size = 28 * 28
		for hidden_size in config.hidden_sizes:
			layers.extend(
				[
					nn.Linear(input_size, hidden_size),
					nn.BatchNorm1d(hidden_size),
					activation(config.activation),
					nn.Dropout(config.dropout),
				]
			)
			input_size = hidden_size
		layers.append(nn.Linear(input_size, 10))
		self.network = nn.Sequential(*layers)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.network(inputs)


def make_datasets(augmentation: bool) -> tuple[Any, Any, Any]:
	normalize = transforms.Normalize((0.1307,), (0.3081,))
	train_transforms: list[Any] = [transforms.ToTensor()]
	if augmentation:
		train_transforms.insert(0, transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)))
	train_transforms.append(normalize)
	train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.Compose(train_transforms))
	evaluation_train = datasets.MNIST(
		DATA_DIR, train=True, download=False, transform=transforms.Compose([transforms.ToTensor(), normalize])
	)
	test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), normalize]))
	split_generator = torch.Generator().manual_seed(DEFAULT_SEED)
	shuffled_indices = torch.randperm(len(train_full), generator=split_generator).tolist()
	train = torch.utils.data.Subset(train_full, shuffled_indices[:55_000])
	validation = torch.utils.data.Subset(evaluation_train, shuffled_indices[55_000:])
	return train, validation, test


def make_loader(dataset: Any, batch_size: int, shuffle: bool, seed: int) -> DataLoader[Any]:
	generator = torch.Generator().manual_seed(seed)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def mixup_batch(inputs: torch.Tensor, targets: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
	if alpha <= 0:
		return inputs, targets, targets, 1.0
	lam = float(torch.distributions.Beta(alpha, alpha).sample())
	permutation = torch.randperm(inputs.size(0), device=inputs.device)
	return inputs.mul(lam).add_(inputs[permutation].mul(1.0 - lam)), targets, targets[permutation], lam


def accuracy(model: nn.Module, loader: DataLoader[Any], device: torch.device) -> float:
	model.eval()
	correct = 0
	total = 0
	with torch.inference_mode():
		for inputs, targets in loader:
			predictions = model(inputs.to(device)).argmax(dim=1)
			correct += int((predictions == targets.to(device)).sum())
			total += targets.size(0)
	return correct / total


def train_model(
	config: TrialConfig,
	train_loader: DataLoader[Any],
	validation_loader: DataLoader[Any],
	device: torch.device,
	epochs: int,
	patience: int,
	verbose: bool = False,
) -> tuple[float, int, dict[str, torch.Tensor]]:
	model = MNISTMLP(config).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
	criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
	best_accuracy = 0.0
	best_epoch = 1
	best_state: dict[str, torch.Tensor] = copy.deepcopy(model.state_dict())
	stale_epochs = 0

	for epoch in range(1, epochs + 1):
		model.train()
		for inputs, targets in train_loader:
			inputs, targets = inputs.to(device), targets.to(device)
			mixed_inputs, first_targets, second_targets, lam = mixup_batch(inputs, targets, config.mixup_alpha)
			loss = lam * criterion(model(mixed_inputs), first_targets)
			loss += (1.0 - lam) * criterion(model(mixed_inputs), second_targets)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
			optimizer.step()
		scheduler.step()
		validation_accuracy = accuracy(model, validation_loader, device)
		if verbose:
			print(f"  epoch {epoch:02d}: validation accuracy={validation_accuracy:.4%}")
		if validation_accuracy > best_accuracy:
			best_accuracy = validation_accuracy
			best_epoch = epoch
			best_state = copy.deepcopy(model.state_dict())
			stale_epochs = 0
		else:
			stale_epochs += 1
			if stale_epochs >= patience:
				break
	return best_accuracy, best_epoch, best_state


def run_trial(index: int, config: TrialConfig, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
	set_seed(args.seed + index)
	train, validation, _ = make_datasets(config.augmentation)
	train_loader = make_loader(train, args.batch_size, True, args.seed + index)
	validation_loader = make_loader(validation, args.batch_size * 2, False, args.seed)
	validation_accuracy, best_epoch, _ = train_model(
		config, train_loader, validation_loader, device, args.epochs, args.patience, args.verbose
	)
	result = {"trial": index + 1, "validation_accuracy": validation_accuracy, "best_epoch": best_epoch, **asdict(config)}
	print(f"trial {index + 1}/{args.trials}: {validation_accuracy:.4%} at epoch {best_epoch} | {config}")
	return result


def final_train(best_result: dict[str, Any], args: argparse.Namespace, device: torch.device) -> float:
	config = TrialConfig(
		tuple(best_result["hidden_sizes"]), best_result["activation"], best_result["dropout"],
		best_result["weight_decay"], best_result["learning_rate"], best_result["label_smoothing"],
		best_result["mixup_alpha"], best_result["augmentation"],
	)
	set_seed(args.seed)
	train, _, test = make_datasets(config.augmentation)
	full_train = datasets.MNIST(
		DATA_DIR, train=True, download=False,
		transform=train.dataset.transform if hasattr(train, "dataset") else train.transform,
	)
	train_loader = make_loader(full_train, args.batch_size, True, args.seed)
	test_loader = make_loader(test, args.batch_size * 2, False, args.seed)
	model = MNISTMLP(config).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_result["best_epoch"])
	criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
	for _ in range(best_result["best_epoch"]):
		model.train()
		for inputs, targets in train_loader:
			inputs, targets = inputs.to(device), targets.to(device)
			mixed_inputs, first_targets, second_targets, lam = mixup_batch(inputs, targets, config.mixup_alpha)
			loss = lam * criterion(model(mixed_inputs), first_targets)
			loss += (1.0 - lam) * criterion(model(mixed_inputs), second_targets)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
			optimizer.step()
		scheduler.step()
	return accuracy(model, test_loader, device)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--trials", type=int, default=len(TRIALS), choices=range(1, len(TRIALS) + 1))
	parser.add_argument("--epochs", type=int, default=25)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
	parser.add_argument("--verbose", action="store_true")
	parser.add_argument("--smoke-test", action="store_true")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	device = choose_device()
	print(f"Using device: {device}")
	if args.smoke_test:
		model = MNISTMLP(TRIALS[0])
		sample = torch.randn(4, 1, 28, 28)
		print(f"Smoke test output shape: {tuple(model(sample).shape)}")
		return  
	results = [run_trial(index, TRIALS[index], args, device) for index in range(args.trials)]
	results.sort(key=lambda result: result["validation_accuracy"], reverse=True)
	best_result = results[0]
	print(f"\nBest validation result: {best_result['validation_accuracy']:.4%}")
	test_accuracy = final_train(best_result, args, device)
	best_result["test_accuracy"] = test_accuracy
	output_path = PROJECT_DIR / "mnist_sweep_results.json"
	output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
	print(f"Final test accuracy: {test_accuracy:.4%}")
	print(f"Saved sweep results to {output_path}")


if __name__ == "__main__":
	main()
