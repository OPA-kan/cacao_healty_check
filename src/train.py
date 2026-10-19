"""Train and compare ResNet50, CBAM, and LGF-CBAM on ImageFolder data."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import ImageFile
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from .models import ModelName, build_model

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/compare"))
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["resnet50", "cbam", "lgf_cbam"],
        default=["resnet50", "cbam", "lgf_cbam"],
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_dataset(dataset: datasets.ImageFolder, limit: int | None, seed: int) -> Subset:
    if limit is None or limit >= len(dataset):
        return Subset(dataset, list(range(len(dataset))))
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(dataset), size=limit, replace=False).tolist()
    return Subset(dataset, indices)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    size = args.image_size
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_base = datasets.ImageFolder(args.data_root / "train", transform=train_transform)
    val_base = datasets.ImageFolder(args.data_root / "val", transform=eval_transform)
    test_base = datasets.ImageFolder(args.data_root / "test", transform=eval_transform)
    if train_base.class_to_idx != val_base.class_to_idx or train_base.class_to_idx != test_base.class_to_idx:
        raise ValueError("train/val/test class folders must match exactly")

    train_set = limit_dataset(train_base, args.max_train_samples, args.seed)
    val_set = limit_dataset(val_base, args.max_eval_samples, args.seed + 1)
    test_set = limit_dataset(test_base, args.max_eval_samples, args.seed + 2)
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(train_set, shuffle=True, **common),
        DataLoader(val_set, shuffle=False, **common),
        DataLoader(test_set, shuffle=False, **common),
        train_base.classes,
    )


def class_weights(loader: DataLoader, num_classes: int, device: torch.device) -> Tensor:
    subset = loader.dataset
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for index in subset.indices:  # type: ignore[attr-defined]
        target = subset.dataset.targets[index]  # type: ignore[attr-defined]
        counts[target] += 1
    weights = counts.sum() / (counts.clamp_min(1) * num_classes)
    return weights.to(device)


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable or name.startswith("attention") or name.startswith("classifier")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, targets in tqdm(loader, leave=False):
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        y_true.extend(targets.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(loader.dataset.dataset.classes))), zero_division=0  # type: ignore[attr-defined]
    )
    classes = loader.dataset.dataset.classes  # type: ignore[attr-defined]
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(classes)
        },
    }
    return total_loss / max(len(loader.dataset), 1), metrics


def benchmark_cpu(model: nn.Module, image_size: int, device: torch.device) -> float:
    model.eval()
    sample = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.inference_mode():
        for _ in range(3):
            model(sample)
        start = time.perf_counter()
        for _ in range(10):
            model(sample)
        elapsed = (time.perf_counter() - start) / 10
    return elapsed * 1000


def train_one(
    variant: ModelName,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    classes: list[str],
    device: torch.device,
) -> None:
    output = args.output_dir / variant
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(variant, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_loader, len(classes), device))
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.3)
    history: list[dict[str, object]] = []
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        set_backbone_trainable(model, epoch > args.freeze_epochs)
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(float(val_metrics["macro_f1"]))
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(f"{variant} epoch={epoch:03d} train_f1={train_metrics['macro_f1']:.4f} val_f1={val_metrics['macro_f1']:.4f}")
        if float(val_metrics["macro_f1"]) > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            torch.save({"model": model.state_dict(), "classes": classes, "variant": variant}, output / "best.pt")

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_loss, test_metrics = run_epoch(model, test_loader, criterion, device)
    metrics = {
        "variant": variant,
        "classes": classes,
        "test_loss": test_loss,
        "test": test_metrics,
        "cpu_inference_ms": benchmark_cpu(model, args.image_size, device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader, classes = make_loaders(args)
    print(f"device={device} classes={classes} train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    for variant in args.models:
        train_one(variant, args, train_loader, val_loader, test_loader, classes, device)


if __name__ == "__main__":
    main()

