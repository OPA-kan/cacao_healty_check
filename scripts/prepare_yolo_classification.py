"""Convert YOLO-labeled pod images into labeled pod crops for classification."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.08)
    return parser.parse_args()


def load_categories(source: Path) -> dict[int, str]:
    payload = json.loads((source / "notes.json").read_text(encoding="utf-8"))
    return {int(item["id"]): str(item["name"]) for item in payload["categories"]}


def parse_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 YOLO fields")
        class_id, x_center, y_center, width, height = parts
        boxes.append((int(class_id), float(x_center), float(y_center), float(width), float(height)))
    return boxes


def crop_box(image: Image.Image, box: tuple[float, float, float, float], padding: float) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    x_center, y_center, box_width, box_height = box
    half_width = box_width * width / 2
    half_height = box_height * height / 2
    pad_x = half_width * padding
    pad_y = half_height * padding
    left = max(0, int((x_center * width - half_width - pad_x)))
    top = max(0, int((y_center * height - half_height - pad_y)))
    right = min(width, int((x_center * width + half_width + pad_x)))
    bottom = min(height, int((y_center * height + half_height + pad_y)))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop coordinates {(left, top, right, bottom)}")
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def main() -> None:
    args = parse_args()
    source = args.source
    images_dir = source / "images"
    labels_dir = source / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError("source must contain images/ and labels/")
    if args.padding < 0:
        raise ValueError("padding must be non-negative")

    mapping_payload = json.loads(args.mapping.read_text(encoding="utf-8"))
    source_mapping = mapping_payload.get(args.dataset)
    if not isinstance(source_mapping, dict):
        raise ValueError(f"No mapping found for dataset {args.dataset!r}")
    mapping = {normalize_label(key): value for key, value in source_mapping.items()}
    categories = load_categories(source)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {image_path.name}")
        boxes = parse_boxes(label_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for box_index, (class_id, x_center, y_center, box_width, box_height) in enumerate(boxes):
                source_class = categories.get(class_id)
                if source_class is None:
                    raise ValueError(f"Unknown category id {class_id} in {label_path}")
                target_class = mapping.get(normalize_label(source_class))
                if target_class is None:
                    raise ValueError(f"Missing mapping for source class {source_class!r}")
                crop, coordinates = crop_box(
                    image,
                    (x_center, y_center, box_width, box_height),
                    args.padding,
                )
                target_dir = output / target_class
                target_dir.mkdir(parents=True, exist_ok=True)
                crop_path = target_dir / f"{image_path.stem}__box{box_index}{image_path.suffix.lower()}"
                crop.save(crop_path, quality=95)
                rows.append(
                    {
                        "dataset": args.dataset,
                        "source_image": str(image_path.resolve()),
                        "source_label": str(label_path.resolve()),
                        "source_class": source_class,
                        "target_class": target_class,
                        "box_index": str(box_index),
                        "x1": str(coordinates[0]),
                        "y1": str(coordinates[1]),
                        "x2": str(coordinates[2]),
                        "y2": str(coordinates[3]),
                        "crop_path": str(crop_path.resolve()),
                    }
                )

    manifest_path = output / "crop_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"images={len(image_paths)} crops={len(rows)}")
    print(f"classes={sorted({row['target_class'] for row in rows})}")
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
