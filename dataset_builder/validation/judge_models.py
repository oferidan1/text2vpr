import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


@dataclass
class JudgeConfig:
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    device_map: str = "auto"
    dtype: Optional[str] = None  # e.g., "float16"


class HFJudge:
    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        torch_dtype = None
        if config.dtype:
            if config.dtype == "float16":
                torch_dtype = torch.float16
            elif config.dtype == "bfloat16":
                torch_dtype = torch.bfloat16
        self.generator = pipeline(
            "text-generation",
            model=AutoModelForCausalLM.from_pretrained(
                config.model_name, device_map=config.device_map, torch_dtype=torch_dtype
            ),
            tokenizer=AutoTokenizer.from_pretrained(config.model_name),
            return_full_text=False,
        )

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
        return self._parse_json_output(text)

    @staticmethod
    def _parse_json_output(text: str) -> Dict[str, Any]:
        # Try direct JSON
        try:
            obj = json.loads(text)
            return HFJudge._normalize(obj)
        except Exception:
            pass

        # Try to locate JSON substring
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                obj = json.loads(match.group(0))
                return HFJudge._normalize(obj)
            except Exception:
                pass

        # Fallback best-effort extraction
        score_match = re.search(r"score\D+(\d)", text, flags=re.IGNORECASE)
        score = int(score_match.group(1)) if score_match else 3
        explanation = text.strip()
        return {"score": max(1, min(5, score)), "explanation": explanation[:500]}

    @staticmethod
    def _normalize(obj: Dict[str, Any]) -> Dict[str, Any]:
        score = int(obj.get("score", 3))
        score = max(1, min(5, score))
        explanation = str(obj.get("explanation", "")).strip()
        return {"score": score, "explanation": explanation[:1000]}


