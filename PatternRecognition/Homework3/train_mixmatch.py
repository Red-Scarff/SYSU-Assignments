"""
MixMatch training script following reference implementation pattern
"""

import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import numpy as np

from models.wideresnet import create_wideresnet28_2
from data_utils import get_cifar10_dataloaders_mixmatch
from utils import evaluate_model, get_cosine_schedule_with_warmup, mixup_data, sharpen


class ModelEMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.999, device=None):
        from copy import deepcopy

        self.ema = deepcopy(model)
        if device is not None:
            self.ema.to(device)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in esd.keys():
                if k in msd:
                    esd[k].copy_(esd[k] * self.decay + (1.0 - self.decay) * msd[k])


def linear_rampup(current, rampup_length=1024):
    """Linear rampup function"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    """Calculate offsets for interleaving"""
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    """Interleave labeled and unlabeled samples"""
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def mixmatch_loss(
    model,
    x_labeled,
    y_labeled,
    x_unlabeled,
    y_unlabeled_guess,
    lambda_u=75,
    T=0.5,
    alpha=0.75,
    device="cuda",
    current_epoch=0,
):
    """
    MixMatch loss computation following the original paper

    Args:
        model: The neural network model
        x_labeled: Labeled input data
        y_labeled: Labeled targets (one-hot)
        x_unlabeled: Unlabeled input data
        y_unlabeled_guess: Pseudo labels for unlabeled data (one-hot)
        lambda_u: Weight for unlabeled loss
        T: Temperature for sharpening
        alpha: Mixup parameter
        device: Device to run on
        current_epoch: Current epoch for rampup

    Returns:
        total_loss, labeled_loss, unlabeled_loss
    """
    batch_size = x_labeled.size(0)

    # Ensure all tensors are on the same device
    x_labeled = x_labeled.to(device)
    y_labeled = y_labeled.to(device)
    x_unlabeled = x_unlabeled.to(device)
    y_unlabeled_guess = y_unlabeled_guess.to(device)

    # Concatenate all inputs and targets (labeled + unlabeled + unlabeled2)
    # For MixMatch, we duplicate unlabeled data to match the paper
    all_inputs = torch.cat([x_labeled, x_unlabeled, x_unlabeled], dim=0)
    all_targets = torch.cat([y_labeled, y_unlabeled_guess, y_unlabeled_guess], dim=0)

    # Apply mixup to all data
    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)  # Ensure l >= 0.5

    idx = torch.randperm(all_inputs.size(0))
    input_a, input_b = all_inputs, all_inputs[idx]
    target_a, target_b = all_targets, all_targets[idx]

    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b

    # Interleave labeled and unlabeled samples for correct BatchNorm
    mixed_input = list(torch.split(mixed_input, batch_size))
    mixed_input = interleave(mixed_input, batch_size)

    # Forward pass through model
    logits = [model(mixed_input[0])]
    for input_batch in mixed_input[1:]:
        logits.append(model(input_batch))

    # De-interleave
    logits = interleave(logits, batch_size)
    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)

    # Split targets
    mixed_targets_x = mixed_target[:batch_size]
    mixed_targets_u = mixed_target[batch_size:]

    # Compute losses following the original MixMatch paper
    # Labeled loss: cross-entropy with soft targets
    loss_x = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_targets_x, dim=1))

    # Unlabeled loss: MSE between predictions and soft targets
    probs_u = torch.softmax(logits_u, dim=1)
    loss_u = torch.mean((probs_u - mixed_targets_u) ** 2)

    # Linear rampup for unlabeled loss weight
    w = linear_rampup(current_epoch) * lambda_u

    # Total loss
    total_loss = loss_x + w * loss_u

    return total_loss, loss_x, loss_u


def generate_pseudo_labels(model, unlabeled_data, K=2, T=0.5, alpha=0.75, device="cuda"):
    """
    Generate pseudo labels for unlabeled data using MixMatch approach

    Args:
        model: The neural network model
        unlabeled_data: Unlabeled input data (multiple augmentations)
        K: Number of augmentations
        T: Temperature for sharpening
        alpha: Mixup parameter
        device: Device to run on

    Returns:
        averaged_unlabeled_data, sharpened_pseudo_labels
    """
    model.eval()

    with torch.no_grad():
        # For MixMatch, unlabeled_data is a tuple of (aug1, aug2) from TransformTwice
        if isinstance(unlabeled_data, (list, tuple)) and len(unlabeled_data) == 2:
            # Average predictions across the two augmentations
            aug1, aug2 = unlabeled_data
            aug1 = aug1.to(device)
            aug2 = aug2.to(device)

            pred1 = torch.softmax(model(aug1), dim=1)
            pred2 = torch.softmax(model(aug2), dim=1)

            # Average predictions
            avg_pred = (pred1 + pred2) / 2.0

            # Use first augmentation as representative data
            unlabeled_inputs = aug1
        else:
            # Single augmentation case
            unlabeled_inputs = unlabeled_data.to(device)
            avg_pred = torch.softmax(model(unlabeled_inputs), dim=1)

    # Sharpen the averaged predictions
    sharpened_pred = sharpen(avg_pred, T)

    model.train()
    return unlabeled_inputs, sharpened_pred


def main():
    parser = argparse.ArgumentParser(description="MixMatch Training (Reference Implementation)")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--n-labeled", type=int, default=4000, help="Number of labeled data")
    parser.add_argument("--batch-size", default=64, type=int, help="Train batch size")
    parser.add_argument("--lr", default=0.002, type=float, help="Initial learning rate")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--total-steps", default=20000, type=int, help="Total training steps")
    parser.add_argument("--eval-step", default=1000, type=int, help="Evaluation frequency")
    parser.add_argument("--lambda-u", default=100, type=float, help="Unlabeled loss weight")
    parser.add_argument("--T", default=0.5, type=float, help="Temperature for sharpening")
    parser.add_argument("--alpha", default=0.75, type=float, help="Mixup parameter")
    parser.add_argument("--use-ema", action="store_true", default=True, help="Use EMA model")
    parser.add_argument("--ema-decay", default=0.999, type=float, help="EMA decay rate")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--data-path", default="./data", type=str, help="Data path")
    parser.add_argument("--save-path", default="./saved_models", type=str, help="Save path")
    parser.add_argument("--log-path", default="./logs", type=str, help="Log path")

    args = parser.parse_args()

    # Create directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("Loading CIFAR-10 dataset...")
    labeled_loader, unlabeled_loader, val_loader, test_loader = get_cifar10_dataloaders_mixmatch(
        args.data_path, args.n_labeled, args.batch_size, num_workers=4, seed=args.seed
    )

    # Create model
    print("Creating WideResNet-28-2 model...")
    model = create_wideresnet28_2().to(device)

    # Create optimizer like reference implementation
    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr, momentum=0.9, nesterov=True)

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, args.total_steps)

    # Create EMA model
    ema_model = None
    if args.use_ema:
        ema_model = ModelEMA(model, args.ema_decay, device)

    # Training
    print("Starting training...")
    print(f"Total steps: {args.total_steps}")
    print(f"Evaluation every {args.eval_step} steps")

    epochs = math.ceil(args.total_steps / args.eval_step)
    best_acc = 0

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model.train()

    # Create log file
    log_file = os.path.join(args.log_path, f"mixmatch_{args.n_labeled}labels_seed{args.seed}.log")

    with open(log_file, "w") as f:
        f.write(f"MixMatch Training Log\n")
        f.write(f"N-labeled: {args.n_labeled}\n")
        f.write(f"Total steps: {args.total_steps}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Lambda-u: {args.lambda_u}\n")
        f.write(f"Temperature: {args.T}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write("=" * 50 + "\n")

    for epoch in range(epochs):
        # Training for eval_step steps
        losses = []
        losses_x = []
        losses_u = []

        pbar = tqdm(range(args.eval_step), desc=f"Epoch {epoch+1}/{epochs}")

        for batch_idx in range(args.eval_step):
            # Get labeled batch
            try:
                inputs_x, targets_x = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                inputs_x, targets_x = next(labeled_iter)

            # Get unlabeled batch
            try:
                unlabeled_batch = next(unlabeled_iter)
                # For MixMatch, we expect (inputs_u1, inputs_u2), targets
                # where inputs_u1 and inputs_u2 are the two augmentations
                inputs_u, _ = unlabeled_batch
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
                inputs_u, _ = unlabeled_batch

            # Move to device
            inputs_x = inputs_x.to(device)
            targets_x = targets_x.to(device)

            # Convert targets to one-hot
            targets_x_onehot = torch.zeros(targets_x.size(0), 10).to(device)
            targets_x_onehot.scatter_(1, targets_x.unsqueeze(1), 1)

            # Generate pseudo labels for unlabeled data
            inputs_u_processed, pseudo_labels = generate_pseudo_labels(
                model, inputs_u, K=2, T=args.T, alpha=args.alpha, device=device
            )

            # Ensure pseudo labels are on the correct device
            inputs_u_processed = inputs_u_processed.to(device)
            pseudo_labels = pseudo_labels.to(device)

            # Compute MixMatch loss
            current_step = epoch * args.eval_step + batch_idx
            loss, loss_x, loss_u = mixmatch_loss(
                model,
                inputs_x,
                targets_x_onehot,
                inputs_u_processed,
                pseudo_labels,
                lambda_u=args.lambda_u,
                T=args.T,
                alpha=args.alpha,
                device=device,
                current_epoch=current_step / args.eval_step,  # Convert to epoch-like value
            )

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.use_ema:
                ema_model.update(model)

            model.zero_grad()

            # Record losses
            losses.append(loss.item())
            losses_x.append(loss_x.item())
            losses_u.append(loss_u.item())

            # Update progress bar
            pbar.set_postfix(
                {"Loss": f"{loss.item():.4f}", "Sup": f"{loss_x.item():.4f}", "Unsup": f"{loss_u.item():.4f}"}
            )
            pbar.update()

        pbar.close()

        # Evaluation
        current_step = (epoch + 1) * args.eval_step
        print(f"Evaluating at step {current_step}...")
        test_model = ema_model.ema if args.use_ema else model
        test_acc, test_loss = evaluate_model(test_model, test_loader, device)

        print(f"Step {current_step}: Test Acc: {test_acc:.2f}%")

        # Log results
        with open(log_file, "a") as f:
            f.write(f"Step {current_step}: Test Acc: {test_acc:.2f}%, Test Loss: {test_loss:.4f}\n")
            f.write(
                f"  Avg Loss: {np.mean(losses):.4f}, Sup Loss: {np.mean(losses_x):.4f}, Unsup Loss: {np.mean(losses_u):.4f}\n"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best model saved with test accuracy: {best_acc:.2f}%")

            # Save best model
            save_path = os.path.join(args.save_path, f"mixmatch_{args.n_labeled}labels_seed{args.seed}_best.pth")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.ema.state_dict() if args.use_ema else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "step": current_step,
                    "test_acc": test_acc,
                    "args": args,
                },
                save_path,
            )

        # Early stopping if we've reached total steps
        if current_step >= args.total_steps:
            break

    # Save final results
    results = {
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "n_labeled": args.n_labeled,
        "total_steps": args.total_steps,
    }

    results_file = os.path.join(args.log_path, f"mixmatch_{args.n_labeled}labels_seed{args.seed}_results.json")
    import json

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
