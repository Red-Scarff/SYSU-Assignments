"""
Utility functions for semi-supervised learning
Includes EMA, loss functions, evaluation metrics, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class EMAModel:
    """
    Exponential Moving Average model for better evaluation
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register model parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def mixup_data(x, y, alpha=1.0):
    """
    Apply mixup augmentation to data

    Args:
        x: Input data
        y: Labels (can be one-hot or hard labels)
        alpha: Mixup parameter

    Returns:
        mixed_x, mixed_y, lambda_mix
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    if y.dim() == 1:  # Hard labels
        y_a, y_b = y, y[index]
        mixed_y = (y_a, y_b, lam)
    else:  # Soft labels
        mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_x, mixed_y, lam


def sharpen(p, T):
    """
    Sharpen probability distribution

    Args:
        p: Probability distribution
        T: Temperature parameter

    Returns:
        Sharpened distribution
    """
    sharp_p = p ** (1.0 / T)
    sharp_p = sharp_p / sharp_p.sum(dim=1, keepdim=True)
    return sharp_p


def cross_entropy_loss(logits, targets, use_hard_labels=True, reduction="none"):
    """
    Cross entropy loss that handles both hard and soft labels

    Args:
        logits: Model predictions
        targets: Target labels (hard or soft)
        use_hard_labels: Whether to use hard labels
        reduction: Reduction method

    Returns:
        Loss value
    """
    if use_hard_labels:
        return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == "none":
            return nll_loss
        elif reduction == "mean":
            return nll_loss.mean()
        elif reduction == "sum":
            return nll_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {reduction}")


def consistency_loss(logits_s, logits_w, name="ce", T=1.0, p_cutoff=0.0, use_hard_labels=True):
    """
    Consistency loss for semi-supervised learning

    Args:
        logits_s: Strong augmentation logits
        logits_w: Weak augmentation logits
        name: Loss function name
        T: Temperature for sharpening
        p_cutoff: Confidence threshold
        use_hard_labels: Whether to use hard pseudo labels

    Returns:
        loss, mask, select, pseudo_lb
    """
    assert name in ["ce", "mse"]

    # Generate pseudo labels from weak augmentation
    logits_w = logits_w.detach()
    pseudo_label = torch.softmax(logits_w, dim=-1)

    # Apply temperature sharpening
    if T < 1.0:
        pseudo_label = sharpen(pseudo_label, T)

    # Get confidence and predicted class
    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

    # Create mask based on confidence threshold
    mask = max_probs.ge(p_cutoff).float()
    select = max_probs.ge(p_cutoff).long()

    if use_hard_labels:
        masked_loss = cross_entropy_loss(logits_s, max_idx, use_hard_labels, reduction="none") * mask
    else:
        pseudo_label = torch.softmax(logits_w / T, dim=-1)
        masked_loss = cross_entropy_loss(logits_s, pseudo_label, use_hard_labels, reduction="none") * mask

    return masked_loss.mean(), mask.mean(), select, max_idx


def interleave(x, size):
    """Interleave samples for batch normalization like reference implementation"""
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    """De-interleave samples like reference implementation"""
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on given data loader

    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to run evaluation on

    Returns:
        accuracy, loss
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = F.cross_entropy(outputs, targets)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

    accuracy = 100.0 * total_correct / total_samples
    avg_loss = total_loss / len(data_loader)

    return accuracy, avg_loss


def save_checkpoint(state, filename):
    """Save model checkpoint"""
    torch.save(state, filename)


def load_checkpoint(filename, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint.get("epoch", 0), checkpoint.get("best_acc", 0)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function
    between the initial lr set in the optimizer to 0, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test accuracy function
    output = torch.randn(10, 5)
    target = torch.randint(0, 5, (10,))
    acc = accuracy(output, target)
    print(f"Accuracy: {acc[0].item():.2f}%")

    # Test mixup
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    mixed_x, mixed_y, lam = mixup_data(x, y)
    print(f"Mixup lambda: {lam:.3f}")

    # Test sharpening
    p = torch.softmax(torch.randn(4, 10), dim=1)
    sharp_p = sharpen(p, T=0.5)
    print(f"Original entropy: {(-p * torch.log(p + 1e-8)).sum(1).mean():.3f}")
    print(f"Sharpened entropy: {(-sharp_p * torch.log(sharp_p + 1e-8)).sum(1).mean():.3f}")
