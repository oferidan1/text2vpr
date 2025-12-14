import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class JudgeConfig:
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens: int = 384
    temperature: float = 0.2
    top_p: float = 0.9
    device_map: str = "auto"
    dtype: Optional[str] = None  # e.g., "float16"
    device: Optional[int] = None  # GPU device index; -1 for CPU; None=auto


class HFJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        torch_dtype = None
        if config.dtype:
            if config.dtype == "float16":
                torch_dtype = torch.float16
            elif config.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
        # Prefer accelerate when available; otherwise load normally and place on a single device
        try:
            import accelerate  # noqa: F401
            has_accelerate = True
        except Exception:
            has_accelerate = False

        # Use accelerate only when both installed and an explicit device_map is desired
        use_accelerate = bool(has_accelerate and self.config.device_map)

        model_kwargs = {"torch_dtype": torch_dtype}
        if use_accelerate:
            model_kwargs["device_map"] = self.config.device_map

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, **model_kwargs)

        # If not using accelerate sharding, place the model on a single device
        if not use_accelerate:
            # Determine target device based on config.device
            target_device_str = "cpu"
            pipeline_device = -1
            if self.config.device is not None:
                if self.config.device >= 0 and torch.cuda.is_available():
                    target_device_str = f"cuda:{self.config.device}"
                    pipeline_device = int(self.config.device)
                elif self.config.device < 0:
                    target_device_str = "cpu"
                    pipeline_device = -1
                elif torch.backends.mps.is_available():  # macOS GPU fallback
                    target_device_str = "mps"
                    pipeline_device = -1
                else:
                    target_device_str = "cpu"
                    pipeline_device = -1
            else:
                # Auto
                if torch.cuda.is_available():
                    target_device_str = "cuda"
                    pipeline_device = 0
                elif torch.backends.mps.is_available():
                    target_device_str = "mps"
                    pipeline_device = -1
                else:
                    target_device_str = "cpu"
                    pipeline_device = -1
            model.to(target_device_str)

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Build pipeline; only set device when not using accelerate (sharded models manage their own devices)
        pipeline_kwargs: Dict[str, Any] = {
            "task": "text-generation",
            "model": model,
            "tokenizer": tokenizer,
            "return_full_text": False,
        }
        if not use_accelerate:
            # Use explicit device selection when provided; otherwise auto
            if self.config.device is not None:
                pipeline_kwargs["device"] = int(self.config.device) if int(self.config.device) >= 0 else -1
            else:
                pipeline_kwargs["device"] = 0 if torch.cuda.is_available() else -1

        self.generator = pipeline(**pipeline_kwargs)

    def judge(self, prompt: str, system_instructions: str = "") -> Dict[str, Any]:
        full_prompt = prompt if not system_instructions else f"{system_instructions}\n\n{prompt}"
        outputs = self.generator(
            full_prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0.0,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        text = outputs[0]["generated_text"] if outputs and outputs[0] else ""
        parsed = self._parse_json_output(text)
        # Include raw artifacts for optional persistence by callers
        parsed["_raw_text"] = text
        parsed["_full_prompt"] = full_prompt
        return parsed

    @staticmethod
    def _parse_json_output(text: str) -> Dict[str, Any]:
        # Try direct JSON
        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            pass

        # Try to locate JSON substring
        if obj is None:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    obj = json.loads(match.group(0))
                except Exception:
                    obj = None

        if obj is None:
            return HFJudge._fallback(text)
        return HFJudge._normalize(obj)

    @staticmethod
    def _to_list(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            # stringify all items
            return [str(v).strip() for v in value if str(v).strip()]
        # Try to split bullet-like text
        s = str(value)
        parts = re.split(r"\n|;|\,| • | - ", s)
        return [p.strip(" -•\t\r\n").strip() for p in parts if p.strip()]

    @staticmethod
    def _to_int_list(value: Any) -> List[int]:
        if value is None:
            return []
        if isinstance(value, list):
            out = []
            for v in value:
                try:
                    out.append(int(v))
                except Exception:
                    continue
            return out
        # try to parse from comma/space separated string
        s = str(value)
        ints: List[int] = []
        for tok in re.split(r"[^0-9]+", s):
            if tok.isdigit():
                try:
                    ints.append(int(tok))
                except Exception:
                    pass
        return ints

    @staticmethod
    def _fallback(text: str) -> Dict[str, Any]:
        # Heuristics: look for a 1-5 score and crude lists
        score_match = re.search(r"score\D+(\d)", text, flags=re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 3
        # Attempt to split sections
        overlap = []
        crit = []
        noncrit = []
        lower = text.lower()
        if "overlap" in lower:
            section = text[text.lower().find("overlap"):]
            overlap = HFJudge._to_list(section)
        if "critical" in lower:
            section = text[text.lower().find("critical"):]
            crit = HFJudge._to_list(section)
        if "non" in lower and "critical" in lower:
            idx = lower.find("non")
            section = text[idx:]
            noncrit = HFJudge._to_list(section)
        return {
            "overlap": overlap[:20],
            "critical_inconsistencies": crit[:20],
            "noncritical_inconsistencies": noncrit[:20],
            "score": max(1, min(5, score)),
            "rationale": text.strip()[:1000],
            "outlier_indices": [],
        }

    @staticmethod
    def _normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
        # Accept several key variants
        score_raw = obj.get("score", obj.get("pair_score", obj.get("rating", 3)))
        try:
            score = int(score_raw)
        except Exception:
            score = 3
        score = max(1, min(5, score))

        overlap = obj.get("overlap", obj.get("overlap_themes", obj.get("shared", obj.get("common", []))))
        crit = obj.get("critical_inconsistencies", obj.get("critical", []))
        noncrit = obj.get("noncritical_inconsistencies", obj.get("non_critical", obj.get("noncritical", [])))
        rationale = obj.get("rationale", obj.get("explanation", ""))
        outliers = obj.get("outlier_indices", obj.get("outliers", []))

        # Normalize outlier reasons into a map[int, str]
        reasons_map: Dict[int, str] = {}
        # Preferred: explicit map
        raw_map = obj.get("outlier_reasons_map", None)
        if isinstance(raw_map, dict):
            for k, v in raw_map.items():
                try:
                    ki = int(k)
                except Exception:
                    try:
                        ki = int(v.get("index")) if isinstance(v, dict) and "index" in v else None
                    except Exception:
                        ki = None
                if ki is not None:
                    reasons_map[ki] = str(v if not isinstance(v, dict) else v.get("reason", "")).strip()[:300]
        else:
            raw_reasons = obj.get("outlier_reasons", None)
            # If list aligned with outlier indices
            if isinstance(raw_reasons, list):
                ints = HFJudge._to_int_list(outliers)
                for i, reason in enumerate(raw_reasons):
                    if i < len(ints):
                        reasons_map[ints[i]] = str(reason).strip()[:300]
            # If dict form
            if isinstance(raw_reasons, dict):
                for k, v in raw_reasons.items():
                    try:
                        ki = int(k)
                        reasons_map[ki] = str(v).strip()[:300]
                    except Exception:
                        continue
            # If 'outliers' is list of objects with {index, reason}
            raw_outliers = obj.get("outliers", None)
            if isinstance(raw_outliers, list):
                for item in raw_outliers:
                    if isinstance(item, dict):
                        try:
                            ki = int(item.get("index"))
                            rv = str(item.get("reason", "")).strip()[:300]
                            if rv:
                                reasons_map[ki] = rv
                        except Exception:
                            continue

        return {
            "overlap": HFJudge._to_list(overlap)[:20],
            "critical_inconsistencies": HFJudge._to_list(crit)[:20],
            "noncritical_inconsistencies": HFJudge._to_list(noncrit)[:20],
            "score": score,
            "rationale": str(rationale).strip()[:1000],
            "outlier_indices": HFJudge._to_int_list(outliers)[:100],
            "outlier_reasons_map": reasons_map,
        }


