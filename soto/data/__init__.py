"""Dataset loaders for SoTo."""

from .geneval import load_geneval_captions, download_geneval_data
from .dreambench import load_dreambench_captions, download_dreambench_data
from .coco import load_coco_captions, download_coco_data

__all__ = [
    "load_geneval_captions",
    "download_geneval_data",
    "load_dreambench_captions",
    "download_dreambench_data",
    "load_coco_captions",
    "download_coco_data",
]
