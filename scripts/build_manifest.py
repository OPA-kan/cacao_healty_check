"""Build a validated, label-normalized image manifest from dataset folders."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train", "val", "test"}


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_dataset(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("dataset must use NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset,
        required=True,
        help="Dataset source in NAME=PATH form. Repeat for multiple sources.",
    )
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_class_dirs(root: Path) -> list[tuple[str, str, Path]]:
    split_dirs = [child for child in root.iterdir() if child.is_dir() and child.name.lower() in SPLIT_NAMES]
    if split_dirs:
        entries: list[tuple[str, str, Path]] = []
        for split_dir in sorted(split_dirs, key=lambda path: path.name.lower()):
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                entries.append((split_dir.name.lower(), class_dir.name, class_dir))
        return entries
    return [("", class_dir.name, class_dir) for class_dir in sorted(path for path in root.iterdir() if path.is_dir())]


def build_rows(dataset_name: str, root: Path, mapping: dict[str, str]) -> tuple[list[dict[str, str]], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    class_dirs = iter_class_dirs(root)
    if not class_dirs:
        raise ValueError(f"No class folders found under {root}")

    rows: list[dict[str, str]] = []
    errors: list[str] = []
    normalized_mapping = {normalize_label(key): value for key, value in mapping.items()}
    for split, source_class, class_dir in class_dirs:
        target_class = normalized_mapping.get(normalize_label(source_class))
        if target_class is None:
            errors.append(f"{dataset_name}: missing mapping for source class {source_class!r}")
            continue
        for path in sorted(class_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    image_mode = image.mode
                digest = sha256_file(path)
            except (OSError, UnidentifiedImageError) as exc:
                errors.append(f"{path}: invalid image ({exc})")
                continue
            rows.append(
                {
                    "dataset": dataset_name,
                    "path": str(path.resolve()),
                    "relative_path": str(path.relative_to(root)),
                    "source_class": source_class,
                    "target_class": target_class,
                    "source_split": split,
                    "width": str(width),
                    "height": str(height),
                    "mode": image_mode,
                    "sha256": digest,
                }
            )
    return rows, errors


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    rows: list[dict[str, str]] = []
    errors: list[str] = []
    for dataset_name, root in args.dataset:
        dataset_mapping = mapping.get(dataset_name)
        if not isinstance(dataset_mapping, dict):
            raise ValueError(f"No mapping object found for dataset {dataset_name!r}")
        dataset_rows, dataset_errors = build_rows(dataset_name, root, dataset_mapping)
        rows.extend(dataset_rows)
        errors.extend(dataset_errors)

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    if not rows:
        raise ValueError("No image files found")

    rows.sort(key=lambda row: (row["dataset"], row["relative_path"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    class_counts = Counter(row["target_class"] for row in rows)
    duplicate_hashes = len(rows) - len({row["sha256"] for row in rows})
    print(f"images={len(rows)} datasets={len({row['dataset'] for row in rows})}")
    print(f"target_classes={dict(sorted(class_counts.items()))}")
    print(f"duplicate_content_candidates={duplicate_hashes}")
    print(f"manifest={args.output.resolve()}")


if __name__ == "__main__":
    main()
