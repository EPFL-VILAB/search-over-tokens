"""Dataset loaders for SoT."""

from .geneval import load_geneval_captions, download_geneval_data
from .dreambench import load_dreambench_captions, download_dreambench_data
from .coco import load_coco_captions, download_coco_data
from .dpg import load_dpg_captions, download_dpg_data

__all__ = [
    "load_geneval_captions",
    "download_geneval_data",
    "load_dreambench_captions",
    "download_dreambench_data",
    "load_coco_captions",
    "download_coco_data",
    "load_dpg_captions",
    "download_dpg_data",
]
