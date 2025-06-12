"""
Data utilities for CIFAR-10 semi-supervised learning
Handles data loading, splitting, and dataset creation
"""

import numpy as np
import torch
import math
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from PIL import Image
import json
import os

from augmentations import TransformTwice, TransformFixMatch, get_cifar10_transforms


def normalize_cifar10(x, mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616)):
    """Normalize CIFAR-10 data like in reference implementation"""
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x = x.astype(np.float32) / 255.0
    x = (x - mean) / std
    return x


def transpose_cifar10(x, source="NHWC", target="NCHW"):
    """Transpose CIFAR-10 data from NHWC to NCHW"""
    return x.transpose([source.index(d) for d in target])


class CIFAR10SSL(Dataset):
    """
    CIFAR-10 dataset for semi-supervised learning
    """

    def __init__(self, root, indexs=None, train=True, transform=None, target_transform=None, download=True):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Load CIFAR-10 dataset
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)

        if indexs is not None:
            self.data = self.dataset.data[indexs]
            self.targets = np.array(self.dataset.targets)[indexs]
        else:
            self.data = self.dataset.data
            self.targets = np.array(self.dataset.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)  # Convert to PIL Image like reference implementation

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR10Unlabeled(CIFAR10SSL):
    """
    CIFAR-10 unlabeled dataset (targets set to -1)
    """

    def __init__(self, root, indexs, train=True, transform=None, target_transform=None, download=True):
        super().__init__(root, indexs, train, transform, target_transform, download)
        # Set all targets to -1 for unlabeled data
        self.targets = np.array([-1 for _ in range(len(self.targets))])


def create_data_splits(labels, n_labeled_per_class, seed=42):
    """
    Create train/unlabeled/validation splits for CIFAR-10

    Args:
        labels: List of labels for the dataset
        n_labeled_per_class: Number of labeled samples per class
        seed: Random seed for reproducibility

    Returns:
        train_labeled_idxs, train_unlabeled_idxs, val_idxs
    """
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):  # CIFAR-10 has 10 classes
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)

        # Split: labeled, unlabeled, validation (500 per class for validation)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])

    # Shuffle the indices
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs


def create_ssl_splits_fixmatch(labels, n_labeled, seed=42):
    """
    Create splits for FixMatch (all data used as unlabeled)

    Args:
        labels: List of labels for the dataset
        n_labeled: Total number of labeled samples
        seed: Random seed for reproducibility

    Returns:
        labeled_idxs, unlabeled_idxs
    """
    np.random.seed(seed)
    labels = np.array(labels)
    labeled_idxs = []
    n_labeled_per_class = n_labeled // 10

    for i in range(10):  # CIFAR-10 has 10 classes
        idxs = np.where(labels == i)[0]
        idxs = np.random.choice(idxs, n_labeled_per_class, False)
        labeled_idxs.extend(idxs)

    labeled_idxs = np.array(labeled_idxs)
    unlabeled_idxs = np.array(range(len(labels)))  # All data as unlabeled

    np.random.shuffle(labeled_idxs)

    return labeled_idxs, unlabeled_idxs


def expand_labeled_dataset_fixmatch(labeled_idxs, batch_size, eval_step, num_labeled):
    """
    Expand labeled dataset for FixMatch like reference implementation

    Args:
        labeled_idxs: Original labeled indices
        batch_size: Batch size
        eval_step: Evaluation step interval
        num_labeled: Number of labeled samples

    Returns:
        Expanded labeled indices
    """
    # Calculate expansion factor like reference implementation
    num_expand_x = math.ceil(batch_size * eval_step / num_labeled)
    expanded_idxs = np.hstack([labeled_idxs for _ in range(num_expand_x)])
    np.random.shuffle(expanded_idxs)
    return expanded_idxs


def expand_labeled_dataset(labeled_idxs, batch_size, total_steps):
    """
    Expand labeled dataset to ensure enough samples for training
    Only expand if we have very few labeled samples (< batch_size)

    Args:
        labeled_idxs: Original labeled indices
        batch_size: Batch size
        total_steps: Total training steps

    Returns:
        Expanded labeled indices (only if necessary)
    """
    current_samples = len(labeled_idxs)

    # Only expand if we have fewer samples than batch size
    if current_samples >= batch_size:
        return labeled_idxs

    # Calculate minimum expansion needed
    min_expansion = (batch_size // current_samples) + 1
    expanded_idxs = np.tile(labeled_idxs, min_expansion)

    # Shuffle the expanded indices
    np.random.shuffle(expanded_idxs)
    return expanded_idxs


def get_cifar10_dataloaders_mixmatch(root, n_labeled, batch_size=64, num_workers=4, seed=42, total_steps=20000):
    """
    Get CIFAR-10 dataloaders for MixMatch (compatible with reference implementation)

    Args:
        root: Data root directory
        n_labeled: Number of labeled samples
        batch_size: Batch size
        num_workers: Number of worker processes
        seed: Random seed
        total_steps: Total training steps

    Returns:
        labeled_loader, unlabeled_loader, val_loader, test_loader
    """
    # CIFAR-10 normalization constants
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    # Transforms like reference implementation
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # Load base dataset to get labels
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True)

    # Create splits using FixMatch style for consistency
    labeled_idxs, unlabeled_idxs = create_ssl_splits_fixmatch(base_dataset.targets, n_labeled, seed)

    # For validation, use a subset of unlabeled data
    val_size = min(1000, len(unlabeled_idxs) // 10)
    val_idxs = unlabeled_idxs[:val_size]
    unlabeled_idxs = unlabeled_idxs[val_size:]

    # Expand labeled dataset if needed for small datasets
    if n_labeled < batch_size:
        labeled_idxs = expand_labeled_dataset_fixmatch(labeled_idxs, batch_size, 1000, n_labeled)

    # Create datasets
    labeled_dataset = CIFAR10SSL(root, labeled_idxs, train=True, transform=train_transform)
    unlabeled_dataset = CIFAR10SSL(root, unlabeled_idxs, train=True, transform=TransformTwice(train_transform))
    val_dataset = CIFAR10SSL(root, val_idxs, train=True, transform=test_transform)
    test_dataset = CIFAR10SSL(root, train=False, transform=test_transform)

    # Create dataloaders
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"#Labeled: {len(labeled_idxs)} #Unlabeled: {len(unlabeled_idxs)} #Val: {len(val_idxs)}")

    return labeled_loader, unlabeled_loader, val_loader, test_loader


def get_cifar10_dataloaders_fixmatch(root, n_labeled, batch_size=64, num_workers=4, seed=42, eval_step=1000, mu=7):
    """
    Get CIFAR-10 dataloaders for FixMatch (compatible with reference implementation)

    Args:
        root: Data root directory
        n_labeled: Number of labeled samples
        batch_size: Batch size
        num_workers: Number of worker processes
        seed: Random seed
        eval_step: Evaluation step interval
        mu: Unlabeled batch size multiplier

    Returns:
        labeled_loader, unlabeled_loader, test_loader
    """
    # CIFAR-10 normalization constants
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)

    # Transforms like reference implementation
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # Load base dataset to get labels
    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=True)

    # Create splits
    labeled_idxs, unlabeled_idxs = create_ssl_splits_fixmatch(base_dataset.targets, n_labeled, seed)

    # Expand labeled dataset like reference implementation
    if n_labeled < batch_size:
        labeled_idxs = expand_labeled_dataset_fixmatch(labeled_idxs, batch_size, eval_step, n_labeled)

    # Create datasets
    labeled_dataset = CIFAR10SSL(root, labeled_idxs, train=True, transform=train_transform)
    unlabeled_dataset = CIFAR10SSL(root, unlabeled_idxs, train=True, transform=TransformFixMatch(mean=mean, std=std))
    test_dataset = CIFAR10SSL(root, train=False, transform=test_transform)

    # Create dataloaders like reference implementation
    labeled_loader = DataLoader(
        labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    unlabeled_loader = DataLoader(
        unlabeled_dataset, batch_size=batch_size * mu, shuffle=True, num_workers=num_workers, drop_last=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"#Labeled: {len(labeled_idxs)} #Unlabeled: {len(unlabeled_idxs)}")

    return labeled_loader, unlabeled_loader, test_loader


def save_data_statistics(n_labeled, save_path):
    """Save data distribution statistics"""
    distribution = [0.1] * 10  # Equal distribution for CIFAR-10
    stats = {"distribution": distribution}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(stats, f)


if __name__ == "__main__":
    # Test data loading
    print("Testing MixMatch dataloaders...")
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_cifar10_dataloaders_mixmatch(
        "./data", n_labeled=40, batch_size=64
    )

    print("\nTesting FixMatch dataloaders...")
    labeled_loader, unlabeled_loader, test_loader = get_cifar10_dataloaders_fixmatch(
        "./data", n_labeled=40, batch_size=64
    )

    # Test a batch
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        print(f"Labeled batch shape: {inputs.shape}, targets: {targets.shape}")
        break

    for batch_idx, (inputs, targets) in enumerate(unlabeled_loader):
        if isinstance(inputs, tuple):
            print(f"Unlabeled batch shapes: {inputs[0].shape}, {inputs[1].shape}")
        else:
            print(f"Unlabeled batch shape: {inputs.shape}")
        break
