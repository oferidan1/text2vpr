import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


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
        else:
            raise ValueError(f"Unknown segmentation backend: {self.backend}")

    def _torch_device(self) -> str:
        try:
            import torch
            if self.config.device.startswith("cuda") and torch.cuda.is_available():
                return "cuda"
            return "cpu"
        except Exception:
            return "cpu"

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


