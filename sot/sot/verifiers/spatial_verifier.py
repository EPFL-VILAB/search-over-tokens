"""Spatial verifier using GroundingDINO + SAM for compositional evaluation.

Evaluates generated images for:
  - Object counting: Are the expected objects present in the correct quantities?
  - Color attribution: Do objects have the expected colors?
  - Spatial relations: Are objects positioned correctly relative to each other?

All models are loaded from HuggingFace Hub (no local checkpoint files required):
  - GroundingDINO:  ``IDEA-Research/grounding-dino-tiny``
  - SAM:            ``facebook/sam-vit-huge``
  - CLIP (color):   ``openai/clip-vit-large-patch14`` (via open_clip)

This verifier is designed for use with GenEval-style metadata that
specifies ``include`` clauses with ``class``, ``count``, optional
``color``, and optional ``position`` fields.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Set, Tuple
from PIL import Image

from sot.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("sot.grounded_sam")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    GDINO_AVAILABLE = True
except (ImportError, Exception):
    GDINO_AVAILABLE = False

try:
    from transformers import SamModel, SamProcessor
    SAM_AVAILABLE = True
except (ImportError, Exception):
    SAM_AVAILABLE = False

try:
    import open_clip
    from clip_benchmark.metrics import zeroshot_classification as zsc
    zsc.tqdm = lambda it, *args, **kwargs: it  # suppress tqdm in zsc
    OPENCLIP_AVAILABLE = True
except (ImportError, Exception):
    OPENCLIP_AVAILABLE = False

__all__ = ["SpatialVerifier"]

# ── Colours used by CLIP-based colour classification ──────────────────
COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]


# ── PSE (Projected Spatial Evaluation) helpers ────────────────────────
def _calculate_prod_cumsum(att1: torch.Tensor, att2: torch.Tensor, relation: str) -> torch.Tensor:
    """PSG loss for spatial relation checking."""
    if relation in ("left", "top"):
        x1, x2 = att1, att2
    else:
        x1, x2 = att2, att1
    x2_cm = torch.cumsum(x2, dim=0)
    return torch.sum(x1 * x2_cm)


def _get_rel_pair(rel: str):
    mapping = {
        "to the right of": ("right", "left"),
        "right of": ("right", "left"),
        "to the left of": ("left", "right"),
        "left of": ("left", "right"),
        "above": ("top", "bottom"),
        "below": ("bottom", "top"),
    }
    return mapping.get(rel, (None, None))


def _get_xy_projection(mask: torch.Tensor):
    """Project a 2-D mask to x and y marginals."""
    xmask = mask.sum(dim=0).float()
    xmask = xmask / (xmask.sum() + 1e-8)
    ymask = mask.sum(dim=1).float()
    ymask = ymask / (ymask.sum() + 1e-8)
    return xmask, ymask


def _check_binary_relation(mask1: torch.Tensor, mask2: torch.Tensor, relation: str) -> float:
    """Check a spatial relation using PSE (projected spatial evaluation)."""
    if mask1.sum() == 0 or mask2.sum() == 0:
        return 0.0
    xm1, ym1 = _get_xy_projection(mask1)
    xm2, ym2 = _get_xy_projection(mask2)
    primary, reverse = _get_rel_pair(relation)
    if primary is None:
        return 0.0
    if primary in ("left", "right"):
        loss = _calculate_prod_cumsum(xm1, xm2, primary)
        rev = _calculate_prod_cumsum(xm1, xm2, reverse)
    else:
        loss = _calculate_prod_cumsum(ym1, ym2, primary)
        rev = _calculate_prod_cumsum(ym1, ym2, reverse)
    return max((rev - loss).item(), 0.0)


# ── Image crop dataset for colour classification ─────────────────────
class _ImageCrops(torch.utils.data.Dataset):
    """Crops objects from an image using boxes and masks for colour classification."""

    def __init__(self, image: Image.Image, boxes, masks, transform):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color="#999")
        self._boxes = boxes
        self._masks = masks
        self.transform = transform

    def __len__(self):
        return len(self._boxes)

    def __getitem__(self, index):
        box = self._boxes[index]
        mask = self._masks[index] if self._masks is not None else None
        if mask is not None:
            mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
            mask_img = Image.fromarray(mask_np.astype(np.uint8) * 255, mode="L")
            image = Image.composite(self._image, self._blank, mask_img)
        else:
            image = self._image
        box_np = box.cpu().numpy() if isinstance(box, torch.Tensor) else np.array(box)
        image = image.crop(box_np[:4])
        return (self.transform(image), 0)


# ── Main verifier ────────────────────────────────────────────────────
@VerifierFactory.register("grounded_sam")
class SpatialVerifier(BaseVerifier):
    """Spatial composition verifier using GroundingDINO + SAM + CLIP.

    All models downloaded from HuggingFace Hub.

    Config parameters:
        - grounding_dino_model: HF model name
          (default: ``"IDEA-Research/grounding-dino-tiny"``)
        - sam_model: HF model name
          (default: ``"facebook/sam-vit-huge"``)
        - box_threshold: Detection confidence threshold (default: 0.35)
        - text_threshold: Text matching threshold (default: 0.25)
        - part_by_part: Return continuous partial score (default: True)
        - spatial_metric: ``"pse"`` or ``"geneval"`` (default: ``"pse"``)
        - batch_size: Batch size for scoring (default: 4)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        if not GDINO_AVAILABLE:
            raise ImportError(
                "GroundingDINO not available. Install transformers>=4.38:\n"
                "  pip install transformers>=4.38"
            )
        if not SAM_AVAILABLE:
            raise ImportError(
                "SAM not available. Install transformers>=4.38:\n"
                "  pip install transformers>=4.38"
            )

        super().__init__(config, device)

        # ── Load GroundingDINO ─────────────────────────────────────
        gdino_name = config.get("grounding_dino_model", "IDEA-Research/grounding-dino-tiny")
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_name)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_name
        ).to(device).eval()
        logger.debug(f"Loaded GroundingDINO: {gdino_name}")

        # ── Load SAM ──────────────────────────────────────────────
        sam_name = config.get("sam_model", "facebook/sam-vit-huge")
        self.sam_processor = SamProcessor.from_pretrained(sam_name)
        self.sam_model = SamModel.from_pretrained(sam_name).to(device).eval()
        logger.debug(f"Loaded SAM: {sam_name}")

        # ── Load CLIP for colour classification ───────────────────
        if OPENCLIP_AVAILABLE:
            clip_arch = "ViT-L-14"
            self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
                clip_arch, pretrained="openai", device=device,
            )
            self.clip_tokenizer = open_clip.get_tokenizer(clip_arch)
            self.clip_model.eval()
            logger.debug(f"Loaded OpenCLIP {clip_arch} for colour classification")
        else:
            self.clip_model = None
            logger.warning("open_clip not available — colour classification disabled")

        self._color_classifiers: Dict[str, torch.Tensor] = {}
        self.box_threshold = config.get("box_threshold", 0.35)
        self.text_threshold = config.get("text_threshold", 0.25)
        self.part_by_part = config.get("part_by_part", True)
        self.spatial_metric = config.get("spatial_metric", "pse")
        logger.info(
            f"Loaded Grounded SAM (GroundingDINO + SAM + CLIP, metric={self.spatial_metric})"
        )

    # ── Object detection ──────────────────────────────────────────
    def _detect_objects(
        self, image: Image.Image, class_name: str
    ) -> Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run GroundingDINO to detect objects of a given class.

        Returns:
            (count, boxes [N,4], masks [N,H,W], confidences [N])
        """
        text_prompt = f"all {class_name}s."

        inputs = self.gdino_processor(
            images=image, text=text_prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.gdino_model(**inputs)

        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],  # (H, W)
        )[0]

        boxes = results["boxes"]  # [N, 4] in xyxy format
        scores = results["scores"]  # [N]

        if len(boxes) == 0:
            return 0, None, None, None

        # ── Prune overlapping boxes ─────────────────────────────
        boxes, scores = self._prune_boxes(boxes, scores)

        # ── Run SAM for masks ───────────────────────────────────
        masks = self._segment(image, boxes)

        count = len(boxes)
        return count, boxes, masks, scores

    def _prune_boxes(
        self, boxes: torch.Tensor, scores: torch.Tensor, threshold: float = 0.7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove highly overlapping boxes, keeping the larger one."""
        if len(boxes) <= 1:
            return boxes, scores
        removes = torch.zeros(len(boxes), dtype=torch.bool)
        for i in range(len(boxes)):
            if removes[i]:
                continue
            for j in range(i + 1, len(boxes)):
                if removes[j]:
                    continue
                # IoF (intersection over foreground)
                b1, b2 = boxes[i], boxes[j]
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])
                if x2 <= x1 or y2 <= y1:
                    continue
                inter = (x2 - x1) * (y2 - y1)
                a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                io1 = inter / (a1 + 1e-8)
                io2 = inter / (a2 + 1e-8)
                if io1 > threshold or io2 > threshold:
                    if a1 > a2:
                        removes[i] = True
                        break
                    else:
                        removes[j] = True
        keep = ~removes
        return boxes[keep], scores[keep]

    def _segment(
        self, image: Image.Image, boxes: torch.Tensor
    ) -> torch.Tensor:
        """Run SAM to segment objects given bounding boxes.

        Returns masks tensor of shape ``[N, H, W]``.
        """
        inputs = self.sam_processor(
            image, input_boxes=[boxes.cpu().tolist()], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam_model(**inputs)

        masks = self.sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0]  # [N, num_masks, H, W]

        # Take the best mask for each box (highest IoU)
        if masks.dim() == 4:
            iou_scores = outputs.iou_scores.cpu()  # [1, N, num_masks]
            best_idx = iou_scores.squeeze(0).argmax(dim=-1)  # [N]
            masks = torch.stack([masks[i, best_idx[i]] for i in range(len(masks))])
        return masks.bool()

    # ── Colour classification ─────────────────────────────────────
    def _classify_colors(
        self, image: Image.Image, boxes: torch.Tensor, masks: Optional[torch.Tensor], class_name: str
    ) -> List[str]:
        """Classify the colour of each detected object using CLIP."""
        if self.clip_model is None or len(boxes) == 0:
            return []

        if class_name not in self._color_classifiers:
            self._color_classifiers[class_name] = zsc.zero_shot_classifier(
                self.clip_model, self.clip_tokenizer, COLORS,
                [
                    f"a photo of a {{c}} {class_name}",
                    f"a photo of a {{c}}-colored {class_name}",
                    f"a photo of a {{c}} object",
                ],
                self.device,
            )

        clf = self._color_classifiers[class_name]
        ds = _ImageCrops(image, boxes, masks, self.clip_transform)
        dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=0)

        with torch.no_grad():
            pred, _ = zsc.run_classification(self.clip_model, clf, dl, self.device)
        return [COLORS[i.item()] for i in pred.argmax(1)]

    # ── Spatial relation checking ─────────────────────────────────
    def _check_position(
        self,
        obj_boxes: torch.Tensor,
        obj_masks: torch.Tensor,
        target_boxes: torch.Tensor,
        target_masks: torch.Tensor,
        expected_rel: str,
    ) -> float:
        """Check spatial relation between detected objects."""
        scores_list: List[float] = []
        for i in range(len(obj_boxes)):
            for j in range(len(target_boxes)):
                if self.spatial_metric == "pse":
                    mask_a = obj_masks[i].to(self.device)
                    mask_b = target_masks[j].to(self.device)
                    score = _check_binary_relation(mask_a, mask_b, expected_rel)
                else:
                    # Geneval-style box comparison
                    score = 1.0 if self._box_rel(obj_boxes[i], target_boxes[j], expected_rel) else 0.0
                scores_list.append(score)
        return np.mean(scores_list) if scores_list else 0.0

    @staticmethod
    def _box_rel(box_a: torch.Tensor, box_b: torch.Tensor, expected_rel: str) -> bool:
        """Simple box-center relative position check."""
        ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
        cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
        dx, dy = ca[0] - cb[0], ca[1] - cb[1]
        rels: Set[str] = set()
        if dx < 0:
            rels.add("left of")
        if dx > 0:
            rels.add("right of")
        if dy < 0:
            rels.add("above")
        if dy > 0:
            rels.add("below")
        return expected_rel in rels

    # ── Single image evaluation ───────────────────────────────────
    def _evaluate_image(self, image: Image.Image, metadata: Dict) -> float:
        """Evaluate one image against GenEval metadata.

        Returns a continuous score in [0, 1] when ``part_by_part`` is True,
        or binary {0, 1} otherwise.
        """
        if metadata is None:
            return 0.0

        include = metadata.get("include", [])
        if not include:
            return 1.0  # nothing to check

        pil_image = image if isinstance(image, Image.Image) else Image.fromarray(np.array(image))

        # Detect all required objects
        object_list = [req["class"] for req in include]
        obj_data: Dict[str, Tuple] = {}
        for obj_name in object_list:
            if obj_name not in obj_data:
                count, boxes, masks, confs = self._detect_objects(pil_image, obj_name)
                obj_data[obj_name] = (count, boxes, masks, confs)

        # Evaluate part-by-part
        score = 0.0
        max_score = 0.0
        all_correct = True

        for req in include:
            class_name = req["class"]
            required_count = req["count"]
            count, boxes, masks, confs = obj_data[class_name]
            max_score += 1.0  # count

            # Trim to required count
            if count > required_count and boxes is not None:
                boxes = boxes[:required_count]
                masks = masks[:required_count] if masks is not None else None
                confs = confs[:required_count] if confs is not None else None
                count = required_count

            # Count check
            if count == required_count:
                score += 1.0
            else:
                all_correct = False
                if self.part_by_part:
                    # Partial credit for counting
                    diff = abs(count - required_count) / max(required_count, 1)
                    score += max(0.0, 1.0 - diff)

            # Colour check
            if "color" in req and count > 0 and boxes is not None:
                max_score += 1.0
                colors = self._classify_colors(pil_image, boxes, masks, class_name)
                matching = sum(1 for c in colors if c == req["color"])
                if matching >= required_count:
                    score += 1.0
                else:
                    all_correct = False

            # Position check
            if "position" in req and count > 0 and boxes is not None:
                max_score += 1.0
                expected_rel, target_group = req["position"]
                target_name = object_list[target_group]
                t_count, t_boxes, t_masks, _ = obj_data.get(target_name, (0, None, None, None))
                if t_count > 0 and t_boxes is not None:
                    pos_score = self._check_position(
                        boxes, masks, t_boxes, t_masks, expected_rel,
                    )
                    if pos_score > 0.5:
                        score += pos_score
                    else:
                        all_correct = False
                        if self.part_by_part:
                            score += pos_score
                else:
                    all_correct = False

        if self.part_by_part and max_score > 0:
            return score / max_score
        return 1.0 if all_correct else 0.0

    # ── Main scoring method ───────────────────────────────────────
    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Score images by compositional correctness.

        Requires ``metadata`` in kwargs (GenEval-style dict with
        ``include`` / ``exclude`` clauses).

        Args:
            images: List of PIL Images.
            prompts: Text prompts.
            **kwargs: Must contain ``metadata`` dict.

        Returns:
            Scores tensor of shape ``[B]``.
        """
        metadata = kwargs.get("metadata", None)

        all_scores: List[float] = []
        for img in images:
            s = self._evaluate_image(img, metadata)
            all_scores.append(s)

        return torch.tensor(all_scores, dtype=torch.float32)

