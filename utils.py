"""Data loading, evaluation metrics, and result files."""

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(root, batch_size, workers, seed, download=False, rotation=0, dataset='cifar10'):
    if dataset == 'cifar100':
        dataset_class = datasets.CIFAR100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    else:
        dataset_class = datasets.CIFAR10
        dataset_class.url = 'https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz'
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    normalize = transforms.Normalize(mean, std)
    train_transform = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                            transforms.RandomRotation(rotation),
                                            transforms.ToTensor(), normalize,
                                        ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    train = dataset_class(root, train=True, download=download, transform=train_transform)
    validation = dataset_class(root, train=True, download=False, transform=test_transform)
    test = dataset_class(root, train=False, download=download, transform=test_transform)

    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    targets = np.asarray(train.targets)
    for label in range(len(train.classes)):
        indices = rng.permutation(np.flatnonzero(targets == label))
        val_count = len(indices) // 10
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    generator = torch.Generator().manual_seed(seed)
    options = dict(batch_size=batch_size, num_workers=workers,
                   pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker)
    return (
        DataLoader(Subset(train, train_indices), shuffle=True, generator=generator, **options),
        DataLoader(Subset(validation, val_indices), shuffle=False, **options),
        DataLoader(test, shuffle=False, **options),
        train.classes,
    )


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def classification_metrics(confusion, classes):
    """Each class is positive (1); all other classes are negative (0)."""
    matrix = np.asarray(confusion, dtype=np.int64)
    per_class = []
    for i, name in enumerate(classes):
        tp = int(matrix[i, i])
        fn = int(matrix[i].sum()) - tp
        fp = int(matrix[:, i].sum()) - tp
        tn = int(matrix.sum()) - tp - fn - fp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(dict(class_name=name, tp=tp, fp=fp, fn=fn, tn=tn,
                              precision=precision, recall=recall, f1=f1, support=tp + fn))
    return dict(
        accuracy=float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        macro_precision=float(np.mean([r['precision'] for r in per_class])),
        macro_recall=float(np.mean([r['recall'] for r in per_class])),
        macro_f1=float(np.mean([r['f1'] for r in per_class])),
        per_class=per_class, confusion_matrix=matrix.tolist(),
    )


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def write_artifacts(output, history, metrics):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'history.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    save_json(output / 'metrics.json', metrics)
    save_json(output / 'history.json', history)
