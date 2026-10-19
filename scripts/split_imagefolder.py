"""Split class folders into train/val/test ImageFolder directories."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--move", action="store_true", help="move files instead of copying them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val-ratio + test-ratio must be less than 1")
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)

    classes = sorted(path for path in args.source.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"No class folders found under {args.source}")
    operation = shutil.move if args.move else shutil.copy2
    rng = random.Random(args.seed)

    for class_dir in classes:
        files = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        rng.shuffle(files)
        test_count = round(len(files) * args.test_ratio)
        val_count = round(len(files) * args.val_ratio)
        split_files = {
            "test": files[:test_count],
            "val": files[test_count:test_count + val_count],
            "train": files[test_count + val_count:],
        }
        for split, selected in split_files.items():
            target_dir = args.output / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(selected):
                target = target_dir / f"{source.stem}_{index}{source.suffix.lower()}"
                operation(source, target)
        print(f"{class_dir.name}: {len(files)} images")


if __name__ == "__main__":
    main()

