"""Train a brain-tumor MRI classifier.

Examples
--------
    python train.py --data_dir data                                   # sensible defaults (ResNet-18)
    python train.py --data_dir data --model efficientnet_b0 --epochs 25
    python train.py --data_dir data --model simple_cnn --no_pretrained --lr 1e-3

Training recipe
---------------
1. (Pre-trained models) Warm-up: the backbone is frozen and only the new classification
   head is trained for ``--freeze_epochs`` epochs.
2. Fine-tuning: everything is unfrozen and trained with AdamW + cosine learning-rate decay.
3. Class-weighted cross-entropy with label smoothing, gradient clipping, optional mixed
   precision on NVIDIA GPUs, early stopping on validation macro-F1.
4. The best checkpoint (highest validation macro-F1) is evaluated once on the test set.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import build_dataloaders
from evaluate import evaluate_model
from model import HEAD_ATTR, SUPPORTED, build_model, count_parameters
from utils import get_device, plot_history, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a brain tumor MRI classifier.")
    p.add_argument("--data_dir", default="data", help="Folder containing Training/ and Testing/ (see README).")
    p.add_argument("--output_dir", default="outputs", help="Where checkpoints, plots and metrics are saved.")
    p.add_argument("--model", default="resnet18", choices=SUPPORTED)
    p.add_argument("--no_pretrained", action="store_true", help="Train from random initialisation.")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None, help="Fine-tuning LR (default 3e-4; 1e-3 for simple_cnn).")
    p.add_argument("--head_lr", type=float, default=1e-3, help="LR while only the head is trained (warm-up).")
    p.add_argument("--freeze_epochs", type=int, default=2, help="Warm-up epochs with the backbone frozen.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=6, help="Early-stopping patience (epochs).")
    p.add_argument("--val_split", type=float, default=0.15, help="Fraction of Training/ used for validation.")
    p.add_argument("--test_split", type=float, default=0.15, help="Only used if the dataset has no Testing/ folder.")
    p.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision on CUDA.")
    return p.parse_args()


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    head_prefix = HEAD_ATTR[model_name] + "."
    for name, param in model.named_parameters():
        param.requires_grad = trainable or name.startswith(head_prefix)


def make_optimizer(model: nn.Module, lr: float, weight_decay: float, epochs: int):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    return optimizer, scheduler


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older PyTorch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, desc=""):
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    preds_all, labels_all = [], []

    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc=desc, leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            preds_all.extend(logits.argmax(1).cpu().tolist())
            labels_all.extend(y.cpu().tolist())

    acc = sum(int(p == t) for p, t in zip(preds_all, labels_all)) / max(n, 1)
    f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return total_loss / max(n, 1), acc, f1


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda" and not args.no_amp
    lr = args.lr if args.lr is not None else (1e-3 if args.model == "simple_cnn" else 3e-4)

    print(f"Device: {device} | model: {args.model} | mixed precision: {use_amp}")
    loaders, info = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.val_split, args.test_split, args.num_workers, args.seed
    )
    classes = info["classes"]
    print(f"Layout: {info['layout']} | classes: {classes}")
    for split in ("train", "val", "test"):
        print(f"  {split:<5}: {sum(info['counts'][split].values()):>6} images  {info['counts'][split]}")

    pretrained = not args.no_pretrained
    model = build_model(args.model, len(classes), pretrained=pretrained, dropout=args.dropout).to(device)
    total, _ = count_parameters(model)
    print(f"Parameters: {total / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(
        weight=info["class_weights"].to(device), label_smoothing=args.label_smoothing
    )
    scaler = make_scaler(use_amp)

    # Warm-up only makes sense when the backbone is pre-trained.
    freeze_epochs = args.freeze_epochs if (pretrained and args.model != "simple_cnn") else 0
    freeze_epochs = max(0, min(freeze_epochs, args.epochs - 1))
    if freeze_epochs > 0:
        set_backbone_trainable(model, args.model, False)
        optimizer, scheduler = make_optimizer(model, args.head_lr, args.weight_decay, freeze_epochs)
        print(f"Warm-up: training the head only for {freeze_epochs} epoch(s)")
    else:
        optimizer, scheduler = make_optimizer(model, lr, args.weight_decay, args.epochs)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr")}
    best_f1, best_loss, best_epoch, bad_epochs = -1.0, float("inf"), 0, 0
    ckpt_path = out_dir / "best_model.pt"
    saved_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    for epoch in range(1, args.epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            set_backbone_trainable(model, args.model, True)
            optimizer, scheduler = make_optimizer(model, lr, args.weight_decay, args.epochs - freeze_epochs)
            print("Fine-tuning: backbone unfrozen")

        t0 = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, use_amp,
                                       desc=f"Epoch {epoch}/{args.epochs} [train]")
        va_loss, va_acc, va_f1 = run_epoch(model, loaders["val"], criterion, device, use_amp=use_amp,
                                           desc=f"Epoch {epoch}/{args.epochs} [val]")
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, va_f1, current_lr)):
            history[k].append(v)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} macro-F1 {va_f1:.4f} | "
              f"lr {current_lr:.1e} | {time.time() - t0:.0f}s")

        improved = va_f1 > best_f1 or (va_f1 == best_f1 and va_loss < best_loss)
        if improved:
            best_f1, best_loss, best_epoch, bad_epochs = va_f1, va_loss, epoch, 0
            torch.save(
                {
                    "model_name": args.model,
                    "classes": classes,
                    "img_size": args.img_size,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_f1": float(va_f1),
                    "val_acc": float(va_acc),
                    "args": saved_args,
                },
                ckpt_path,
            )
            print(f"  -> new best model saved (val macro-F1 {va_f1:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience and epoch > freeze_epochs:
                print(f"Early stopping: no improvement for {args.patience} epochs.")
                break

    save_json(history, out_dir / "history.json")
    plot_history(history, out_dir / "training_curves.png")

    # ---- Final, one-time evaluation of the best checkpoint on the untouched test set ----
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val macro-F1 {best_f1:.4f})")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate_model(model, loaders["test"], device, classes, out_dir / "test", "test")
    save_json(
        {"best_epoch": best_epoch, "best_val_macro_f1": best_f1, "test": test_metrics, "args": saved_args},
        out_dir / "results.json",
    )
    print(f"\nDone. Best model: {ckpt_path}")
    print("Try it:  python predict.py --image path/to/mri.jpg   |   python app.py")


if __name__ == "__main__":
    main()
