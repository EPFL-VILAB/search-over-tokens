"""DreamBench++ dataset loader with auto-download.

DreamBench++ is a human-aligned benchmark for personalized image generation
containing 150 diverse reference images and 1350 prompts (9 per image).

Each sample provides:
  - A text prompt describing the desired generation
  - A reference image whose subject should be preserved

Paper: https://arxiv.org/abs/2406.16855
GitHub: https://github.com/yuangpeng/dreambench_plus
HuggingFace: https://huggingface.co/datasets/yuangpeng/dreambench_plus
Google Drive: https://drive.google.com/file/d/17HNVYU5yvuHDC6VhesJsWsXo1UWy_CSs

Raw data layout (Google Drive / HuggingFace release)::

    <data_root>/
        captions/
            live_subject/
                animal/00.txt   ← line 1: subject name; lines 2–10: prompts
                human/00.txt
            object/00.txt
            style/00.txt
        images/
            live_subject/animal/00.jpg
            live_subject/human/00.jpg
            object/00.jpg
            style/00.jpg
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict

logger = logging.getLogger("data.dreambench")

__all__ = ["load_dreambench_captions", "download_dreambench_data", "load_dreambench_samples_by_category"]

DREAMBENCH_HF_DATASET = "yuangpeng/dreambench_plus"
DREAMBENCH_GDRIVE_ID = "17HNVYU5yvuHDC6VhesJsWsXo1UWy_CSs"


# ---------------------------------------------------------------------------
# Helper: locate the data root inside a download directory
# ---------------------------------------------------------------------------

def _find_data_root(root: Path) -> Optional[Path]:
    """Walk *root* looking for a directory that contains both ``captions/``
    and ``images/`` sub-directories with the expected DreamBench++ layout."""
    for candidate in [root / "data", root]:
        if candidate.is_dir():
            subdirs = {d.name for d in candidate.iterdir() if d.is_dir()}
            if "captions" in subdirs and "images" in subdirs:
                return candidate
    # Deeper recursive search (max 4 levels)
    for dirpath, dirnames, _ in os.walk(root):
        depth = Path(dirpath).relative_to(root).parts
        if len(depth) > 4:
            dirnames.clear()
            continue
        subdirs = set(dirnames)
        if "captions" in subdirs and "images" in subdirs:
            return Path(dirpath)
    return None


# ---------------------------------------------------------------------------
# Download backends
# ---------------------------------------------------------------------------

def _download_from_huggingface(target_dir: Path) -> Path:
    """Download DreamBench++ from HuggingFace using ``huggingface_hub``.

    Returns:
        Path to the data root (containing ``captions/`` and ``images/``).

    Raises:
        ImportError: if ``huggingface_hub`` is not installed.
        RuntimeError: if the data root cannot be located after download.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to auto-download DreamBench++.\n"
            "Install it with:  pip install huggingface_hub"
        )

    hf_dir = target_dir / "hf_download"
    print(
        f"[DreamBench++] Downloading from HuggingFace "
        f"({DREAMBENCH_HF_DATASET}) → {hf_dir} ..."
    )
    snapshot_download(
        repo_id=DREAMBENCH_HF_DATASET,
        repo_type="dataset",
        local_dir=str(hf_dir),
    )

    data_root = _find_data_root(hf_dir)
    if data_root is None:
        raise RuntimeError(
            f"Downloaded HuggingFace dataset to {hf_dir} but could not "
            "locate a data root with captions/ and images/ directories."
        )
    return data_root


def _download_from_gdrive(target_dir: Path) -> Path:
    """Download DreamBench++ evaluation dataset zip from Google Drive.

    Returns:
        Path to the data root (containing ``captions/`` and ``images/``).

    Raises:
        ImportError: if ``gdown`` is not installed.
        RuntimeError: if download/extraction fails or data root not found.
    """
    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required to download DreamBench++ from Google Drive.\n"
            "Install it with:  pip install gdown"
        )

    zip_path = target_dir / "dreambench_plus.zip"
    extract_dir = target_dir / "gdrive_download"

    url = f"https://drive.google.com/uc?id={DREAMBENCH_GDRIVE_ID}"
    print(f"[DreamBench++] Downloading from Google Drive → {zip_path} ...")
    gdown.download(url, str(zip_path), quiet=False)

    if not zip_path.exists():
        raise RuntimeError(f"Google Drive download failed — {zip_path} not found.")

    print(f"[DreamBench++] Extracting {zip_path} → {extract_dir} ...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(str(extract_dir))
    zip_path.unlink()  # clean up zip

    data_root = _find_data_root(extract_dir)
    if data_root is None:
        raise RuntimeError(
            f"Extracted Google Drive zip to {extract_dir} but could not "
            "locate a data root with captions/ and images/ directories."
        )
    return data_root


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_dreambench_data(dataset_dir: str = "datasets/dreambench") -> Path:
    """Ensure DreamBench++ data is available locally.

    Downloads automatically from HuggingFace (primary) or Google Drive
    (fallback) when the data is not present.

    Args:
        dataset_dir: Directory to store / look for the dataset.

    Returns:
        Path to the data root (directory containing ``captions/`` and
        ``images/`` sub-directories).
    """
    dataset_path = Path(dataset_dir)

    # Already present?
    data_root = _find_data_root(dataset_path)
    if data_root is not None:
        n_images = sum(1 for _ in (data_root / "images").rglob("*.jpg"))
        if n_images > 0:
            return data_root

    dataset_path.mkdir(parents=True, exist_ok=True)
    errors: list = []

    # 1) Try HuggingFace
    try:
        data_root = _download_from_huggingface(dataset_path)
        print("[DreamBench++] ✓ Data ready (HuggingFace)")
        return data_root
    except Exception as e:
        errors.append(("HuggingFace", e))
        logger.warning(f"HuggingFace download failed: {e}")

    # 2) Fallback: Google Drive
    try:
        data_root = _download_from_gdrive(dataset_path)
        print("[DreamBench++] ✓ Data ready (Google Drive)")
        return data_root
    except Exception as e:
        errors.append(("Google Drive", e))
        logger.warning(f"Google Drive download failed: {e}")

    # Both failed
    msg = "\n".join(f"  - {name}: {err}" for name, err in errors)
    print(
        f"\n⚠ DreamBench++ auto-download failed:\n{msg}\n\n"
        f"To set up manually:\n"
        f"  1. Download from: https://huggingface.co/datasets/{DREAMBENCH_HF_DATASET}\n"
        f"     Or Google Drive: https://drive.google.com/file/d/{DREAMBENCH_GDRIVE_ID}\n"
        f"  2. Extract so the layout matches:\n"
        f"       {dataset_path}/*/captions/live_subject/animal/*.txt\n"
        f"       {dataset_path}/*/images/live_subject/animal/*.jpg\n"
    )
    raise RuntimeError(
        "DreamBench++ data could not be downloaded. See instructions above."
    )


def load_dreambench_captions(
    dataset_dir: str = "datasets/dreambench",
    num_samples: Optional[int] = None,
    images_dir: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Load DreamBench++ captions and reference image paths.

    Reads directly from the raw per-subject ``.txt`` caption files.
    Downloads automatically if data is not present locally.

    Args:
        dataset_dir: Directory containing (or to download) DreamBench++ data.
        num_samples: Number of samples to load (``None`` = all 1350).
        images_dir: Optional override for the images root directory.
            When ``None``, uses ``<data_root>/images/``.

    Returns:
        (captions, image_paths): Tuple of:
            - captions: list of text prompts
            - image_paths: list of absolute paths to reference images
    """
    data_root = download_dreambench_data(dataset_dir)
    captions_root = data_root / "captions"
    images_base = Path(images_dir) if images_dir else data_root / "images"

    captions: List[str] = []
    image_paths: List[str] = []

    for txt_file in sorted(captions_root.rglob("*.txt")):
        rel_category = txt_file.relative_to(captions_root).parent  # e.g. live_subject/animal
        subject_id = txt_file.stem                                  # e.g. 00
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        # Line 0: subject/style name (skip); lines 1+ are prompts
        prompts = [l.strip() for l in lines[1:] if l.strip()]
        abs_img = str((images_base / rel_category / f"{subject_id}.jpg").resolve())
        for prompt in prompts:
            captions.append(prompt)
            image_paths.append(abs_img)

    if num_samples is not None:
        captions = captions[:num_samples]
        image_paths = image_paths[:num_samples]

    missing = sum(1 for p in image_paths if not os.path.isfile(p))
    if missing > 0:
        logger.warning(
            f"{missing}/{len(image_paths)} reference images not found under {images_base}/"
        )
        print(
            f"⚠ Warning: {missing}/{len(image_paths)} DreamBench++ reference images not found.\n"
            f"  Expected at: {images_base}/\n"
            f"  See: https://huggingface.co/datasets/{DREAMBENCH_HF_DATASET}"
        )

    print(f"[DreamBench++] Loaded {len(captions)} prompts "
          f"({len(captions) - missing} with valid reference images)")
    return captions, image_paths


def load_dreambench_samples_by_category(
    dataset_dir: str = "datasets/dreambench",
    categories: Optional[List[str]] = None,
    images_dir: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Load one sample per DreamBench++ category for demos.

    Picks the first valid (caption, image_path) from each category.

    Args:
        dataset_dir: Directory containing (or to download) DreamBench++ data.
        categories: Subset of category keys to load. Default: all four
            ["live_subject/animal", "live_subject/human", "object", "style"].
        images_dir: Optional override for images root directory.

    Returns:
        List of (caption, image_path, category_label) per category.
        category_label is e.g. "Animal", "Human", "Object", "Style".
    """
    default_cats = ["live_subject/animal", "live_subject/human", "object", "style"]
    cats = categories or default_cats
    cat_labels = {
        "live_subject/animal": "Animal",
        "live_subject/human": "Human",
        "object": "Object",
        "style": "Style",
    }

    data_root = download_dreambench_data(dataset_dir)
    captions_root = data_root / "captions"
    images_base = Path(images_dir) if images_dir else data_root / "images"

    found: Dict[str, Tuple[str, str]] = {}

    for txt_file in sorted(captions_root.rglob("*.txt")):
        rel_category = str(txt_file.relative_to(captions_root).parent)
        if rel_category not in cats or rel_category in found:
            continue
        subject_id = txt_file.stem
        abs_img = str((images_base / rel_category / f"{subject_id}.jpg").resolve())
        if not os.path.isfile(abs_img):
            continue
        lines = txt_file.read_text(encoding="utf-8").splitlines()
        prompts = [l.strip() for l in lines[1:] if l.strip()]
        if prompts:
            found[rel_category] = (prompts[0], abs_img)
        if len(found) >= len(cats):
            break

    result = [
        (found[cat][0], found[cat][1], cat_labels.get(cat, cat))
        for cat in cats
        if cat in found
    ]
    print(
        f"[DreamBench++] Loaded {len(result)} samples by category: "
        f"{[cat_labels.get(c, c) for c in cats if c in found]}"
    )
    return result
