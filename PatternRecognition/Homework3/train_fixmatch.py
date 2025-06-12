"""
FixMatch training script following reference implementation exactly
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
from data_utils import get_cifar10_dataloaders_fixmatch
from utils import interleave, de_interleave, evaluate_model, get_cosine_schedule_with_warmup


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


def main():
    parser = argparse.ArgumentParser(description="FixMatch Training (Reference Implementation)")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--n-labeled", type=int, default=4000, help="Number of labeled data")
    parser.add_argument("--batch-size", default=64, type=int, help="Train batch size")
    parser.add_argument("--lr", default=0.03, type=float, help="Initial learning rate")
    parser.add_argument("--weight-decay", default=5e-4, type=float, help="Weight decay")
    parser.add_argument("--total-steps", default=20000, type=int, help="Total training steps")
    parser.add_argument("--eval-step", default=1000, type=int, help="Evaluation frequency")
    parser.add_argument("--mu", default=7, type=int, help="Unlabeled batch size multiplier")
    parser.add_argument("--lambda-u", default=1.0, type=float, help="Unlabeled loss weight")
    parser.add_argument("--T", default=1.0, type=float, help="Temperature for sharpening")
    parser.add_argument("--threshold", default=0.95, type=float, help="Confidence threshold")
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
    labeled_loader, unlabeled_loader, test_loader = get_cifar10_dataloaders_fixmatch(
        args.data_path,
        args.n_labeled,
        args.batch_size,
        num_workers=4,
        seed=args.seed,
        eval_step=args.eval_step,
        mu=args.mu,
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
    log_file = os.path.join(args.log_path, f"fixmatch_{args.n_labeled}labels_seed{args.seed}.log")

    with open(log_file, "w") as f:
        f.write(f"FixMatch Training Log\n")
        f.write(f"N-labeled: {args.n_labeled}\n")
        f.write(f"Total steps: {args.total_steps}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Lambda-u: {args.lambda_u}\n")
        f.write(f"Temperature: {args.T}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write("=" * 50 + "\n")

    for epoch in range(epochs):
        # Training for eval_step steps
        losses = []
        losses_x = []
        losses_u = []
        mask_probs = []

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
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            batch_size = inputs_x.shape[0]

            # Move to device
            inputs_x = inputs_x.to(device)
            targets_x = targets_x.to(device)
            inputs_u_w = inputs_u_w.to(device)
            inputs_u_s = inputs_u_s.to(device)

            # Concatenate and interleave like reference implementation
            inputs = torch.cat((inputs_x, inputs_u_w, inputs_u_s))
            inputs = interleave(inputs, 2 * args.mu + 1)

            # Forward pass
            logits = model(inputs)

            # De-interleave
            logits = de_interleave(logits, 2 * args.mu + 1)

            # Split logits
            logits_x = logits[:batch_size]
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)

            # Supervised loss
            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")

            # Generate pseudo labels
            pseudo_label = torch.softmax(logits_u_w.detach() / args.T, dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(args.threshold).float()

            # Unsupervised loss
            Lu = (F.cross_entropy(logits_u_s, targets_u, reduction="none") * mask).mean()

            # Total loss
            loss = Lx + args.lambda_u * Lu

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            if args.use_ema:
                ema_model.update(model)

            model.zero_grad()

            # Record losses
            losses.append(loss.item())
            losses_x.append(Lx.item())
            losses_u.append(Lu.item())
            mask_probs.append(mask.mean().item())

            # Update progress bar
            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Sup": f"{Lx.item():.4f}",
                    "Unsup": f"{Lu.item():.4f}",
                    "Mask": f"{mask.mean().item():.3f}",
                }
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
                f"  Avg Loss: {np.mean(losses):.4f}, Sup Loss: {np.mean(losses_x):.4f}, Unsup Loss: {np.mean(losses_u):.4f}, Mask Ratio: {np.mean(mask_probs):.4f}\n"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"New best model saved with test accuracy: {best_acc:.2f}%")

            # Save best model
            save_path = os.path.join(args.save_path, f"fixmatch_{args.n_labeled}labels_seed{args.seed}_best.pth")
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
        if (epoch + 1) * args.eval_step >= args.total_steps:
            break

    # Save final results
    results = {
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "n_labeled": args.n_labeled,
        "total_steps": args.total_steps,
    }

    results_file = os.path.join(args.log_path, f"fixmatch_{args.n_labeled}labels_seed{args.seed}_results.json")
    import json

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
