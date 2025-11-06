import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap


# Optional imports protected per backend
_TORCHVISION_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from torchvision import transforms
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        DeepLabV3_ResNet50_Weights,
    )
    _TORCHVISION_AVAILABLE = True
except Exception:
    pass

try:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    pass


@dataclass
class SegConfig:
    backend: str = "hf_segformer_b2_ade"  # or "torchvision_deeplabv3" or "hf_oneformer_ade_swinl"
    device: str = "cuda" if (hasattr(np, "__version__") and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    prob_threshold: float = 0.4
    min_area_pct: float = 0.002


class Segmenter:
    def __init__(self, config: SegConfig) -> None:
        self.config = config
        self.backend = config.backend

        if self.backend == "torchvision_deeplabv3":
            if not _TORCHVISION_AVAILABLE:
                raise RuntimeError("Torchvision not available for deeplabv3 backend.")
            self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=self.weights).eval()
            self.preprocess = self.weights.transforms()
            self.class_names = self._tv_class_names()
            self.id_to_name = {i: n for i, n in enumerate(self.class_names)}
            self.device = torch.device(self._torch_device())
            self.model.to(self.device)
        elif self.backend == "hf_segformer_b2_ade":
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available for segformer backend.")
            self.processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512"
            )
            self.model.eval()
            # Use model config label mapping
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                # ensure int keys
                self.id_to_name = {int(k): str(v) for k, v in id2label.items()}
            else:
                # fallback to a generic mapping
                self.id_to_name = {}
            self.device = self._torch_device()
            try:
                import torch  # local scope for mypy
                self.model.to(self.device)
            except Exception:
                pass
        elif self.backend == "hf_oneformer_ade_swinl":
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available for oneformer backend.")
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation  # type: ignore
            self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large"
            )
            self.model.eval()
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                self.id_to_name = {int(k): str(v) for k, v in id2label.items()}
            else:
                self.id_to_name = {}
            self.device = self._torch_device()
            try:
                import torch
                self.model.to(self.device)
            except Exception:
                pass
        import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap


# Optional imports protected per backend
_TORCHVISION_AVAILABLE = False
_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from torchvision import transforms
    from torchvision.models.segmentation import (
        deeplabv3_resnet50,
        DeepLabV3_ResNet50_Weights,
    )
    _TORCHVISION_AVAILABLE = True
except Exception:
    pass

try:
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    _TRANSFORMERS_AVAILABLE = True
except Exception:
    pass


@dataclass
class SegConfig:
    backend: str = "hf_segformer_b2_ade"  # or "torchvision_deeplabv3"
    device: str = "cuda" if (hasattr(np, "__version__") and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "") else "cpu"
    prob_threshold: float = 0.4
    min_area_pct: float = 0.002


class Segmenter:
    def __init__(self, config: SegConfig) -> None:
        self.config = config
        self.backend = config.backend

        if self.backend == "torchvision_deeplabv3":
            if not _TORCHVISION_AVAILABLE:
                raise RuntimeError("Torchvision not available for deeplabv3 backend.")
            self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
            self.model = deeplabv3_resnet50(weights=self.weights).eval()
            self.preprocess = self.weights.transforms()
            self.class_names = self._tv_class_names()
            self.id_to_name = {i: n for i, n in enumerate(self.class_names)}
            self.device = torch.device(self._torch_device())
            self.model.to(self.device)
        elif self.backend == "hf_segformer_b2_ade":
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available for segformer backend.")
            self.processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512"
            )
            self.model.eval()
            # Use model config label mapping
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                # ensure int keys
                self.id_to_name = {int(k): str(v) for k, v in id2label.items()}
            else:
                # fallback to a generic mapping
                self.id_to_name = {}
            self.device = self._torch_device()
            try:
                import torch  # local scope for mypy
                self.model.to(self.device)
            except Exception:
                pass
        elif self.backend == "hf_segformer_b5_ade":
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available for segformer backend.")
            self.processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b5-finetuned-ade-640-640"
            )
            self.model.eval()
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                self.id_to_name = {int(k): str(v) for k, v in id2label.items()}
            else:
                self.id_to_name = {}
            self.device = self._torch_device()
            try:
                import torch
                self.model.to(self.device)
            except Exception:
                pass
        elif self.backend == "hf_oneformer_ade_swinl":
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available for oneformer backend.")
            # Import here to avoid hard dependency if Transformers version is old
            from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation  # type: ignore
            self.processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                "shi-labs/oneformer_ade20k_swin_large"
            )
            self.model.eval()
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                self.id_to_name = {int(k): str(v) for k, v in id2label.items()}
            else:
                self.id_to_name = {}
            self.device = self._torch_device()
            try:
                import torch
                self.model.to(self.device)
            except Exception:
                pass
        elif self.backend == "hf_mask2former_ade_swinl":
            if not _TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Transformers not available for mask2former backend.")
            from transformers import Mask2FormerForUniversalSegmentation  # type: ignore
            self.processor = AutoImageProcessor.from_pretrained(
                "facebook/mask2former-swin-large-ade-semantic"
            )
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-ade-semantic"
            )
            self.model.eval()
            id2label = getattr(self.model.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                self.id_to_name = {int(k): str(v) for k, v in id2label.items()}
            else:
                self.id_to_name = {}
            self.device = self._torch_device()
            try:
                import torch
                self.model.to(self.device)
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown segmentation backend: {self.backend}")

        # Initialize VPR relevance lexicon
        self._init_vpr_relevance()

    def _torch_device(self) -> str:
        try:
            import torch
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

    # -------------------- VPR relevance utilities --------------------
    def _init_vpr_relevance(self) -> None:
        # Canonical VPR-relevant terms (structures and place-specific elements)
        canon: Set[str] = {
            "building",
            "skyscraper",
            "house",
            "apartment",
            "bridge",
            "tower",
            "church",
            "cathedral",
            "temple",
            "mosque",
            "castle",
            "monument",
            "statue",
            "fountain",
            "sign",
            "signboard",
            "billboard",
            "traffic light",
            "street light",
            "lamp",
            "lamppost",
            "storefront",
            "shop",
            "awning",
            "balcony",
            "window",
            "door",
            "gate",
            "arch",
            "dome",
            "pillar",
            "column",
            "stairs",
            "staircase",
            "fence",
            "railing",
            "wall",
            "brick wall",
            "road",
            "street",
            "sidewalk",
            "pavement",
            "crosswalk",
            "plaza",
            "square",
            "canal",
            "river",
            "canal bank",
            "river bank",
            "clock",
            "clock tower",
            "lighthouse",
            "station",
            "train station",
            "bus stop",
        }
        # Synonyms mapping to canonical keys
        self._vpr_synonyms: Dict[str, str] = {
            "streetlight": "street light",
            "lamp post": "street light",
            "lamppost": "street light",
            "street lamp": "street light",
            "traffic signal": "traffic light",
            "street sign": "sign",
            "traffic sign": "sign",
            "signboard": "sign",
            "store front": "storefront",
            "shopfront": "storefront",
            "store": "storefront",
            "shop": "storefront",
            "side walk": "sidewalk",
            "paving": "pavement",
            "stairs": "staircase",
            "stair": "staircase",
            "column": "column",
            "pillar": "column",
            "museum": "building",
            "skyscraper": "skyscraper",
        }
        self._vpr_canonical: Set[str] = canon

    @staticmethod
    def _normalize_text(text: str) -> str:
        return str(text).strip().lower()

    def _canonicalize_label(self, name: str) -> str:
        name_l = self._normalize_text(name)
        if name_l in self._vpr_synonyms:
            return self._vpr_synonyms[name_l]
        return name_l

    def is_vpr_relevant(self, name: str) -> bool:
        key = self._canonicalize_label(name)
        # quickly filter common non-relevant categories
        if key in {"person", "car", "bus", "truck", "bicycle", "motorbike", "dog", "cat", "sky", "grass", "tree"}:
            return False
        # allow if canonical or substrings in canonical terms
        if key in self._vpr_canonical:
            return True
        # allow if canonical term is substring of label or vice versa
        return any((key in c) or (c in key) for c in self._vpr_canonical)

    def extract_relevant_terms_from_text(self, description: str) -> Set[str]:
        desc = self._normalize_text(description)
        present: Set[str] = set()
        # phrase presence check for canonical terms and synonyms
        for term in self._vpr_canonical:
            if term and term in desc:
                present.add(term)
        for syn, can in self._vpr_synonyms.items():
            if syn in desc:
                present.add(can)
        return present

    def build_vpr_overlay(self, image_path: str, label_map: np.ndarray, description: str) -> Tuple[Image.Image, Set[str], Set[str], Set[str]]:
        """
        Returns overlay image and sets:
        - matched_relevant: relevant classes detected and mentioned in text
        - detected_not_in_text: relevant classes detected but not mentioned
        - text_not_detected: relevant terms mentioned in text but not detected in image
        """
        # Ensure label map matches image size
        img = Image.open(image_path).convert("RGB")
        label_map_img = self._resize_label_map_to_image(label_map, img.size)

        # Map ids to class names for present labels
        unique_ids = np.unique(label_map_img)
        cid_to_name: Dict[int, str] = {}
        for cid in unique_ids:
            if cid == 0:
                continue
            name = self.id_to_name.get(int(cid), f"class_{int(cid)}")
            cid_to_name[int(cid)] = name

        # Compute relevant mention set from text
        mentioned_terms = self.extract_relevant_terms_from_text(description)

        # Determine detected relevant names (canonicalized)
        detected_relevant_names: Set[str] = set()
        for cid, name in cid_to_name.items():
            if self.is_vpr_relevant(name):
                detected_relevant_names.add(self._canonicalize_label(name))

        # Matched and mismatched sets
        matched_relevant = {n for n in detected_relevant_names if any((n in t) or (t in n) for t in mentioned_terms)}
        detected_not_in_text = detected_relevant_names - matched_relevant
        text_not_detected = {t for t in mentioned_terms if not any((t in n) or (n in t) for n in detected_relevant_names)}

        # Build overlay: blue for matched, red for detected_not_in_text
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        arr = np.array(overlay)
        h, w = label_map_img.shape

        blue = (0, 102, 255, 110)
        red = (255, 64, 64, 110)

        for cid, name in cid_to_name.items():
            if not self.is_vpr_relevant(name):
                continue
            name_c = self._canonicalize_label(name)
            mask = (label_map_img == int(cid))
            if not mask.any():
                continue
            if name_c in matched_relevant:
                arr[mask] = blue
            else:
                # relevant but not mentioned
                arr[mask] = red

        overlay_rgba = Image.fromarray(arr, mode="RGBA")
        composed = Image.alpha_composite(img.convert("RGBA"), overlay_rgba)
        return composed.convert("RGB"), matched_relevant, detected_not_in_text, text_not_detected

    def annotate_overlay(self, overlay_img: Image.Image, description: str, blue_items: Set[str], red_items: Set[str]) -> Image.Image:
        img = overlay_img.convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Prepare text blocks
        blue_text = ", ".join(sorted(list(blue_items))) if blue_items else "(none)"
        red_text = ", ".join(sorted(list(red_items))) if red_items else "(none)"
        lines = []
        lines.append(f"Description: {description}")
        lines.append(f"Blue (mentioned & detected): {blue_text}")
        lines.append(f"Red (detected, not in text): {red_text}")

        # Wrap long lines
        wrapped: List[str] = []
        max_width = img.size[0] - 20
        for line in lines:
            # rough wrap by characters; refine using font metrics for width
            for seg in textwrap.wrap(line, width=100):
                wrapped.append(seg)

        # Estimate panel height
        try:
            line_height = font.getbbox("A")[3] + 6
        except Exception:
            # Fallback for older Pillow
            line_height = font.getsize("A")[1] + 6
        panel_height = 10 + line_height * len(wrapped) + 10

        # Draw semi-opaque white panel at bottom (compat with older Pillow: use paste+mask)
        W, H = img.size
        panel_top = H - panel_height
        panel = Image.new("RGBA", (W, panel_height), (255, 255, 255, 200))
        base_rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
        overlay.paste(panel, (0, panel_top), panel)
        img = Image.alpha_composite(base_rgba, overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw text lines
        y = panel_top + 10
        for t in wrapped:
            draw.text((10, y), t, font=font, fill=(0, 0, 0))
            y += line_height

        return img

    # -------------------- helpers --------------------
    @staticmethod
    def _resize_label_map_to_image(label_map: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """Resize integer label map to match image size using nearest-neighbor."""
        target_w, target_h = image_size
        h, w = label_map.shape
        if (w, h) == (target_w, target_h):
            return label_map.astype(np.int32, copy=False)
        # Use 32-bit int mode to preserve ids
        pil = Image.fromarray(label_map.astype(np.int32), mode="I")
        pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
        return np.array(pil, dtype=np.int32)

    @staticmethod
    def _tv_class_names() -> List[str]:
        # Standard Pascal VOC 21 classes used commonly with torchvision models
        return [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def segment_image(self, image_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Returns:
        - label_map: HxW int array with class indices
        - objects_detected: unique non-background class names
        """
        img = Image.open(image_path).convert("RGB")

        if self.backend == "torchvision_deeplabv3":
            import torch
            with torch.no_grad():
                x = self.preprocess(img).unsqueeze(0).to(self.device)
                out = self.model(x)["out"]  # [1, C, H, W]
                logits = out.squeeze(0).cpu().numpy()
            label_map = logits.argmax(axis=0).astype(np.int32)
            objects = self._postprocess_objects(label_map, logits)
            return label_map, objects

        if self.backend == "hf_segformer_b2_ade":
            import torch
            inputs = self.processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            logits = outputs.logits.squeeze(0).cpu().numpy()  # [C, h, w]
            label_map = logits.argmax(axis=0).astype(np.int32)
            objects = self._postprocess_objects(label_map, logits)
            return label_map, objects

        if self.backend == "hf_oneformer_ade_swinl":
            import torch
            inputs = self.processor(images=img, task_inputs=["semantic"], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            seg = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[img.size[::-1]]
            )[0]
            if hasattr(seg, "cpu"):
                seg = seg.cpu()
            label_map = np.array(seg, dtype=np.int32)
            objects = self._postprocess_objects(label_map, logits=None)  # type: ignore
            return label_map, objects

        if self.backend == "hf_segformer_b5_ade":
            import torch
            inputs = self.processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            logits = outputs.logits.squeeze(0).cpu().numpy()
            label_map = logits.argmax(axis=0).astype(np.int32)
            objects = self._postprocess_objects(label_map, logits)
            return label_map, objects

        if self.backend == "hf_oneformer_ade_swinl":
            import torch
            # OneFormer requires task_inputs=["semantic"] and post-processing
            inputs = self.processor(images=img, task_inputs=["semantic"], return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            seg = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[img.size[::-1]]
            )[0]
            if hasattr(seg, "cpu"):
                seg = seg.cpu()
            label_map = np.array(seg, dtype=np.int32)
            objects = self._postprocess_objects(label_map, logits=None)  # type: ignore
            return label_map, objects

        if self.backend == "hf_mask2former_ade_swinl":
            import torch
            inputs = self.processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            seg = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[img.size[::-1]]
            )[0]
            if hasattr(seg, "cpu"):
                seg = seg.cpu()
            label_map = np.array(seg, dtype=np.int32)
            objects = self._postprocess_objects(label_map, logits=None)  # type: ignore
            return label_map, objects

        else:
            raise ValueError(f"Unknown segmentation backend: {self.backend}")

        # Initialize VPR relevance lexicon
        self._init_vpr_relevance()

    def _postprocess_objects(self, label_map: np.ndarray, logits: np.ndarray) -> List[str]:
        h, w = label_map.shape
        total = float(h * w)
        unique_ids = np.unique(label_map)
        objects: List[str] = []
        for cid in unique_ids:
            if cid == 0:  # background for most models
                continue
            mask = label_map == cid
            area = mask.sum() / total
            if area < self.config.min_area_pct:
                continue
            if cid in self.id_to_name:
                name = str(self.id_to_name[cid])
            else:
                name = f"class_{int(cid)}"

            # probability filter via logits if available
            try:
                probs = self._softmax_channel(logits, cid, mask)
                if probs < self.config.prob_threshold:
                    continue
            except Exception:
                pass
            objects.append(name)
        return sorted(list(set(objects)))

    @staticmethod
    def _softmax_channel(logits: np.ndarray, cid: int, mask: np.ndarray) -> float:
        # logits: [C, H, W]
        c = logits.shape[0]
        flat = logits.reshape(c, -1)
        mask_flat = mask.reshape(-1)
        if mask_flat.sum() == 0:
            return 0.0
        region = flat[:, mask_flat]
        region = region - region.max(axis=0, keepdims=True)
        exp = np.exp(region)
        soft = exp / (exp.sum(axis=0, keepdims=True) + 1e-9)
        return float(soft[cid].mean())

    @staticmethod
    def overlay_mask(image_path: str, label_map: np.ndarray, class_palette: Dict[int, Tuple[int, int, int]]) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        # Resize label map to match the image size
        target_w, target_h = img.size
        h, w = label_map.shape
        if (w, h) != (target_w, target_h):
            pil = Image.fromarray(label_map.astype(np.int32), mode="I")
            pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
            label_map = np.array(pil, dtype=np.int32)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        arr = np.array(overlay)
        h, w = label_map.shape
        for cid, color in class_palette.items():
            if cid == 0:
                continue
            mask = (label_map == cid)
            if not mask.any():
                continue
            arr[mask] = (*color, 90)
        overlay = Image.fromarray(arr, mode="RGBA")
        composed = Image.alpha_composite(img.convert("RGBA"), overlay)
        return composed.convert("RGB")

    def default_palette(self) -> Dict[int, Tuple[int, int, int]]:
        rng = np.random.RandomState(42)
        palette: Dict[int, Tuple[int, int, int]] = {}
        for cid in range(1, 256):
            palette[cid] = tuple(int(x) for x in rng.randint(0, 255, size=3))
        return palette





    def _torch_device(self) -> str:
        try:
            import torch
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

    # -------------------- VPR relevance utilities --------------------
    def _init_vpr_relevance(self) -> None:
        # Canonical VPR-relevant terms (structures and place-specific elements)
        canon: Set[str] = {
            "building",
            "skyscraper",
            "house",
            "apartment",
            "bridge",
            "tower",
            "church",
            "cathedral",
            "temple",
            "mosque",
            "castle",
            "monument",
            "statue",
            "fountain",
            "sign",
            "signboard",
            "billboard",
            "traffic light",
            "street light",
            "lamp",
            "lamppost",
            "storefront",
            "shop",
            "awning",
            "balcony",
            "window",
            "door",
            "gate",
            "arch",
            "dome",
            "pillar",
            "column",
            "stairs",
            "staircase",
            "fence",
            "railing",
            "wall",
            "brick wall",
            "road",
            "street",
            "sidewalk",
            "pavement",
            "crosswalk",
            "plaza",
            "square",
            "canal",
            "river",
            "canal bank",
            "river bank",
            "clock",
            "clock tower",
            "lighthouse",
            "station",
            "train station",
            "bus stop",
        }
        # Synonyms mapping to canonical keys
        self._vpr_synonyms: Dict[str, str] = {
            "streetlight": "street light",
            "lamp post": "street light",
            "lamppost": "street light",
            "street lamp": "street light",
            "traffic signal": "traffic light",
            "street sign": "sign",
            "traffic sign": "sign",
            "signboard": "sign",
            "store front": "storefront",
            "shopfront": "storefront",
            "store": "storefront",
            "shop": "storefront",
            "side walk": "sidewalk",
            "paving": "pavement",
            "stairs": "staircase",
            "stair": "staircase",
            "column": "column",
            "pillar": "column",
            "museum": "building",
            "skyscraper": "skyscraper",
        }
        self._vpr_canonical: Set[str] = canon

    @staticmethod
    def _normalize_text(text: str) -> str:
        return str(text).strip().lower()

    def _canonicalize_label(self, name: str) -> str:
        name_l = self._normalize_text(name)
        if name_l in self._vpr_synonyms:
            return self._vpr_synonyms[name_l]
        return name_l

    def is_vpr_relevant(self, name: str) -> bool:
        key = self._canonicalize_label(name)
        # quickly filter common non-relevant categories
        if key in {"person", "car", "bus", "truck", "bicycle", "motorbike", "dog", "cat", "sky", "grass", "tree"}:
            return False
        # allow if canonical or substrings in canonical terms
        if key in self._vpr_canonical:
            return True
        # allow if canonical term is substring of label or vice versa
        return any((key in c) or (c in key) for c in self._vpr_canonical)

    def extract_relevant_terms_from_text(self, description: str) -> Set[str]:
        desc = self._normalize_text(description)
        present: Set[str] = set()
        # phrase presence check for canonical terms and synonyms
        for term in self._vpr_canonical:
            if term and term in desc:
                present.add(term)
        for syn, can in self._vpr_synonyms.items():
            if syn in desc:
                present.add(can)
        return present

    def build_vpr_overlay(self, image_path: str, label_map: np.ndarray, description: str) -> Tuple[Image.Image, Set[str], Set[str], Set[str]]:
        """
        Returns overlay image and sets:
        - matched_relevant: relevant classes detected and mentioned in text
        - detected_not_in_text: relevant classes detected but not mentioned
        - text_not_detected: relevant terms mentioned in text but not detected in image
        """
        # Ensure label map matches image size
        img = Image.open(image_path).convert("RGB")
        label_map_img = self._resize_label_map_to_image(label_map, img.size)

        # Map ids to class names for present labels
        unique_ids = np.unique(label_map_img)
        cid_to_name: Dict[int, str] = {}
        for cid in unique_ids:
            if cid == 0:
                continue
            name = self.id_to_name.get(int(cid), f"class_{int(cid)}")
            cid_to_name[int(cid)] = name

        # Compute relevant mention set from text
        mentioned_terms = self.extract_relevant_terms_from_text(description)

        # Determine detected relevant names (canonicalized)
        detected_relevant_names: Set[str] = set()
        for cid, name in cid_to_name.items():
            if self.is_vpr_relevant(name):
                detected_relevant_names.add(self._canonicalize_label(name))

        # Matched and mismatched sets
        matched_relevant = {n for n in detected_relevant_names if any((n in t) or (t in n) for t in mentioned_terms)}
        detected_not_in_text = detected_relevant_names - matched_relevant
        text_not_detected = {t for t in mentioned_terms if not any((t in n) or (n in t) for n in detected_relevant_names)}

        # Build overlay: blue for matched, red for detected_not_in_text
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        arr = np.array(overlay)
        h, w = label_map_img.shape

        blue = (0, 102, 255, 110)
        red = (255, 64, 64, 110)

        for cid, name in cid_to_name.items():
            if not self.is_vpr_relevant(name):
                continue
            name_c = self._canonicalize_label(name)
            mask = (label_map_img == int(cid))
            if not mask.any():
                continue
            if name_c in matched_relevant:
                arr[mask] = blue
            else:
                # relevant but not mentioned
                arr[mask] = red

        overlay_rgba = Image.fromarray(arr, mode="RGBA")
        composed = Image.alpha_composite(img.convert("RGBA"), overlay_rgba)
        return composed.convert("RGB"), matched_relevant, detected_not_in_text, text_not_detected

    def annotate_overlay(self, overlay_img: Image.Image, description: str, blue_items: Set[str], red_items: Set[str]) -> Image.Image:
        img = overlay_img.convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Prepare text blocks
        blue_text = ", ".join(sorted(list(blue_items))) if blue_items else "(none)"
        red_text = ", ".join(sorted(list(red_items))) if red_items else "(none)"
        lines = []
        lines.append(f"Description: {description}")
        lines.append(f"Blue (mentioned & detected): {blue_text}")
        lines.append(f"Red (detected, not in text): {red_text}")

        # Wrap long lines
        wrapped: List[str] = []
        max_width = img.size[0] - 20
        for line in lines:
            # rough wrap by characters; refine using font metrics for width
            for seg in textwrap.wrap(line, width=100):
                wrapped.append(seg)

        # Estimate panel height
        line_height = font.getbbox("A")[3] + 6
        panel_height = 10 + line_height * len(wrapped) + 10

        # Draw semi-opaque white panel at bottom (compat with older Pillow: use paste+mask)
        W, H = img.size
        panel_top = H - panel_height
        panel = Image.new("RGBA", (W, panel_height), (255, 255, 255, 200))
        base_rgba = img.convert("RGBA")
        overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
        overlay.paste(panel, (0, panel_top), panel)
        img = Image.alpha_composite(base_rgba, overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw text lines
        y = panel_top + 10
        for t in wrapped:
            draw.text((10, y), t, font=font, fill=(0, 0, 0))
            y += line_height

        return img

    # -------------------- helpers --------------------
    @staticmethod
    def _resize_label_map_to_image(label_map: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """Resize integer label map to match image size using nearest-neighbor."""
        target_w, target_h = image_size
        h, w = label_map.shape
        if (w, h) == (target_w, target_h):
            return label_map.astype(np.int32, copy=False)
        # Use 32-bit int mode to preserve ids
        pil = Image.fromarray(label_map.astype(np.int32), mode="I")
        pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
        return np.array(pil, dtype=np.int32)

    @staticmethod
    def _tv_class_names() -> List[str]:
        # Standard Pascal VOC 21 classes used commonly with torchvision models
        return [
            "background",
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]

    def segment_image(self, image_path: str) -> Tuple[np.ndarray, List[str]]:
        """
        Returns:
        - label_map: HxW int array with class indices
        - objects_detected: unique non-background class names
        """
        img = Image.open(image_path).convert("RGB")

        if self.backend == "torchvision_deeplabv3":
            import torch
            with torch.no_grad():
                x = self.preprocess(img).unsqueeze(0).to(self.device)
                out = self.model(x)["out"]  # [1, C, H, W]
                logits = out.squeeze(0).cpu().numpy()
            label_map = logits.argmax(axis=0).astype(np.int32)
            objects = self._postprocess_objects(label_map, logits)
            return label_map, objects

        if self.backend == "hf_segformer_b2_ade":
            import torch
            inputs = self.processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            logits = outputs.logits.squeeze(0).cpu().numpy()  # [C, h, w]
            label_map = logits.argmax(axis=0).astype(np.int32)
            objects = self._postprocess_objects(label_map, logits)
            return label_map, objects

        raise ValueError(f"Unsupported backend: {self.backend}")

    def _postprocess_objects(self, label_map: np.ndarray, logits: np.ndarray) -> List[str]:
        h, w = label_map.shape
        total = float(h * w)
        unique_ids = np.unique(label_map)
        objects: List[str] = []
        for cid in unique_ids:
            if cid == 0:  # background for most models
                continue
            mask = label_map == cid
            area = mask.sum() / total
            if area < self.config.min_area_pct:
                continue
            if cid in self.id_to_name:
                name = str(self.id_to_name[cid])
            else:
                name = f"class_{int(cid)}"

            # probability filter via logits if available
            try:
                probs = self._softmax_channel(logits, cid, mask)
                if probs < self.config.prob_threshold:
                    continue
            except Exception:
                pass
            objects.append(name)
        return sorted(list(set(objects)))

    @staticmethod
    def _softmax_channel(logits: np.ndarray, cid: int, mask: np.ndarray) -> float:
        # logits: [C, H, W]
        c = logits.shape[0]
        flat = logits.reshape(c, -1)
        mask_flat = mask.reshape(-1)
        if mask_flat.sum() == 0:
            return 0.0
        region = flat[:, mask_flat]
        region = region - region.max(axis=0, keepdims=True)
        exp = np.exp(region)
        soft = exp / (exp.sum(axis=0, keepdims=True) + 1e-9)
        return float(soft[cid].mean())

    @staticmethod
    def overlay_mask(image_path: str, label_map: np.ndarray, class_palette: Dict[int, Tuple[int, int, int]]) -> Image.Image:
        img = Image.open(image_path).convert("RGB")
        # Resize label map to match the image size
        target_w, target_h = img.size
        h, w = label_map.shape
        if (w, h) != (target_w, target_h):
            pil = Image.fromarray(label_map.astype(np.int32), mode="I")
            pil = pil.resize((target_w, target_h), resample=Image.NEAREST)
            label_map = np.array(pil, dtype=np.int32)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        arr = np.array(overlay)
        h, w = label_map.shape
        for cid, color in class_palette.items():
            if cid == 0:
                continue
            mask = (label_map == cid)
            if not mask.any():
                continue
            arr[mask] = (*color, 90)
        overlay = Image.fromarray(arr, mode="RGBA")
        composed = Image.alpha_composite(img.convert("RGBA"), overlay)
        return composed.convert("RGB")

    def default_palette(self) -> Dict[int, Tuple[int, int, int]]:
        rng = np.random.RandomState(42)
        palette: Dict[int, Tuple[int, int, int]] = {}
        for cid in range(1, 256):
            palette[cid] = tuple(int(x) for x in rng.randint(0, 255, size=3))
        return palette


