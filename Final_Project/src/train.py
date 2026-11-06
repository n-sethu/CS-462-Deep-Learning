"""Run from the repository root: python -m src.train --config configs/ensemble.json."""
import argparse
import csv
import json
import os
import random
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .data import make_loaders
from .models import CIFARResNet18


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def run_epoch(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = correct = count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
    return total_loss / count, correct / count


def plot_history(rows, destination):
    epochs = [r['epoch'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ('train_loss', 'val_loss'):
        axes[0].plot(epochs, [r[name] for r in rows], label=name)
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss')
    axes[0].legend()
    axes[1].plot(epochs, [r['val_accuracy'] for r in rows])
    axes[1].set(xlabel='Epoch', ylabel='Validation accuracy', ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/ensemble.json')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--prepare-data', action='store_true', help='Download train data and save splits, without training')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ('seed', 'epochs', 'device'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config['epochs'] < 1:
        raise ValueError('epochs must be positive')
    seed_everything(config['seed'])
    if args.prepare_data:
        train, val = make_loaders(config)
        print(f'Saved splits: train={len(train.dataset)}, detector_train=10000, val={len(val.dataset)}', flush=True)
        return
    device = config['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Run inside a Slurm GPU allocation or choose --device cpu.')
    config['resolved_device'] = device
    output = Path(config['output_dir']) / f"seed_{config['seed']}"
    output.mkdir(parents=True, exist_ok=True)
    # Refuse to replace an earlier experiment, including an interrupted run.
    with (output / 'config.json').open('x') as handle:
        json.dump(config, handle, indent=2)
    train, val = make_loaders(config)
    model = CIFARResNet18().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'],
                                momentum=config['momentum'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    criterion = nn.CrossEntropyLoss()
    (output / 'environment.json').write_text(json.dumps({
        'torch': torch.__version__, 'numpy': np.__version__,
        'cuda_runtime': torch.version.cuda, 'device': str(device),
    }, indent=2))
    history, best_loss = [], float('inf')
    with (output / 'metrics.csv').open('w', newline='') as handle:
        fields = ['epoch', 'learning_rate', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, config['epochs'] + 1):
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = run_epoch(model, train, criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, val, criterion, device)
            scheduler.step()
            row = dict(epoch=epoch, learning_rate=lr, train_loss=train_loss,
                       train_accuracy=train_acc, val_loss=val_loss, val_accuracy=val_acc)
            history.append(row)
            writer.writerow(row)
            handle.flush()
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                                scheduler_state_dict=scheduler.state_dict(), config=config, **row), output / 'best.pt')
                (output / 'best_metrics.json').write_text(json.dumps(row, indent=2))
            plot_history(history, output / 'curves.png')
            print(f'Epoch {epoch}/{config["epochs"]}: train_loss={train_loss:.4f} '
                  f'val_loss={val_loss:.4f} val_accuracy={val_acc:.2%}', flush=True)
    print(f'Best validation loss: {best_loss:.4f}; outputs: {output}', flush=True)


if __name__ == '__main__':
    main()
