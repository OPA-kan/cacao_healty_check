"""Split object crops by source image into ImageFolder directories.

Grouping by source image prevents crops from the same field photo appearing
in both training and evaluation splits.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.val_ratio + args.test_ratio >= 1.0:
        raise ValueError("val-ratio + test-ratio must be less than 1")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    with args.manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"source_image", "target_class", "crop_path"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["source_image"]].append(row)

    group_items = list(groups.items())
    random.Random(args.seed).shuffle(group_items)
    test_groups = round(len(group_items) * args.test_ratio)
    val_groups = round(len(group_items) * args.val_ratio)
    assignments: dict[str, str] = {}
    for index, (source_image, _) in enumerate(group_items):
        if index < test_groups:
            assignments[source_image] = "test"
        elif index < test_groups + val_groups:
            assignments[source_image] = "val"
        else:
            assignments[source_image] = "train"

    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_group_counts: Counter[str] = Counter()
    for source_image, group_rows in groups.items():
        split = assignments[source_image]
        split_group_counts[split] += 1
        for row in group_rows:
            source = Path(row["crop_path"])
            if not source.is_file():
                raise FileNotFoundError(source)
            target_dir = args.output / split / row["target_class"]
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / source.name
            shutil.copy2(source, target)
            split_counts[split][row["target_class"]] += 1

    split_manifest = args.output / "split_manifest.csv"
    split_manifest.parent.mkdir(parents=True, exist_ok=True)
    with split_manifest.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(rows[0]) + ["split"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row["split"] = assignments[row["source_image"]]
            writer.writerow(output_row)

    for split in ("train", "val", "test"):
        print(f"{split}: groups={split_group_counts[split]} images={sum(split_counts[split].values())} classes={dict(sorted(split_counts[split].items()))}")
    print(f"manifest={split_manifest.resolve()}")


if __name__ == "__main__":
    main()
