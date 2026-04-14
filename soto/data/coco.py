"""COCO captions dataset loader with auto-download.

Downloads COCO 2014 validation captions from the official COCO website and
extracts one caption per image for use as text-to-image generation prompts.

Source: http://images.cocodataset.org/annotations/annotations_trainval2014.zip
"""

import json
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

__all__ = ["load_coco_captions", "download_coco_data"]

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
)
_CAPTIONS_JSON = "annotations/captions_val2014.json"


def download_coco_data(dataset_dir: str = "datasets/coco") -> Path:
    """Auto-download COCO val2014 captions if not present.

    Downloads the official COCO 2014 annotations zip and extracts only the
    captions JSON (``captions_val2014.json``).

    Args:
        dataset_dir: Directory to cache the downloaded data.

    Returns:
        Path to the captions JSON file.

    Raises:
        RuntimeError: If download or extraction fails.
    """
    dataset_path = Path(dataset_dir)
    captions_file = dataset_path / "captions_val2014.json"

    if captions_file.exists():
        return captions_file

    dataset_path.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_path / "annotations_trainval2014.zip"

    # Download
    print(f"[COCO] Downloading annotations from {COCO_ANNOTATIONS_URL} ...")
    print("  (This is ~252 MB and only needs to happen once)")
    try:
        urllib.request.urlretrieve(COCO_ANNOTATIONS_URL, str(zip_path))
    except Exception as e:
        raise RuntimeError(f"[COCO] Failed to download annotations: {e}")

    # Extract only the captions JSON
    print(f"[COCO] Extracting {_CAPTIONS_JSON} ...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extract(_CAPTIONS_JSON, str(dataset_path))
        # Move from nested annotations/ to dataset root
        extracted = dataset_path / _CAPTIONS_JSON
        extracted.rename(captions_file)
        # Clean up the now-empty annotations dir and zip
        annotations_dir = dataset_path / "annotations"
        if annotations_dir.exists():
            annotations_dir.rmdir()
    except Exception as e:
        raise RuntimeError(f"[COCO] Failed to extract captions: {e}")
    finally:
        if zip_path.exists():
            zip_path.unlink()

    print(f"[COCO] ✓ Captions saved to {captions_file}")
    return captions_file


def load_coco_captions(
    dataset_dir: str = "datasets/coco",
    num_samples: Optional[int] = 300,
) -> List[str]:
    """Load COCO val2014 captions, auto-downloading if needed.

    Extracts one caption per image from the COCO 2014 validation set,
    sorted by image ID for reproducibility.

    Args:
        dataset_dir: Directory to cache the downloaded data.
        num_samples: Number of captions to load.  Defaults to 300.
            Pass ``None`` to load all available captions (~40 k images).

    Returns:
        List of caption strings.
    """
    captions_file = download_coco_data(dataset_dir)

    with open(captions_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Pick one caption per image (the first one encountered), sorted by image_id
    image_to_caption: dict = {}
    for ann in data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_to_caption:
            image_to_caption[img_id] = ann["caption"].strip()

    # Sort by image_id for deterministic ordering
    captions = [cap for _, cap in sorted(image_to_caption.items())]

    if num_samples is not None:
        captions = captions[:num_samples]

    print(f"[COCO] Loaded {len(captions)} captions")
    return captions
