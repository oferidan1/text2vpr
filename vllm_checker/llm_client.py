from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging


@dataclass
class LLMConfig:
    """Configuration for the local text-only LLM backend.

    This backend mirrors the style used in `visual_checker.llm_client`
    and is meant to be easy to swap out for a vLLM / Phi-4-vision
    server if you prefer a true vision-language model.
    """

    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    device: Optional[int] = None  # GPU index; -1 for CPU; None=auto


class TextOnlyLLMClient:
    """Simple Hugging Face transformers client.

    NOTE: This implementation reasons from **text only** (the
    `description` column), not raw pixels. The checker still passes
    `image_path` so you can replace this class with a true
    vision-language backend that looks at the actual image.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()

        # Make transformers quiet: only errors, no progress bars.
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Configure generation defaults on the model so we don't have to
        # pass them on every call (avoids some HF warnings).
        try:
            gen_cfg = model.generation_config
            if gen_cfg is not None:
                if hasattr(gen_cfg, "temperature"):
                    gen_cfg.temperature = self.config.temperature
                if hasattr(gen_cfg, "top_p"):
                    gen_cfg.top_p = self.config.top_p
        except Exception:  # pragma: no cover - best-effort
            pass

        if self.config.device is not None:
            pipeline_device = int(self.config.device)
        else:
            pipeline_device = 0 if torch.cuda.is_available() else -1

        self._pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            device=pipeline_device,
        )

    def is_object_in_image(
        self,
        *,
        image_path: str,
        object_name: str,
        description: Optional[str] = None,
    ) -> bool:
        """Ask the model if an object is present.

        The default implementation uses only the textual description,
        ignoring actual pixels. You can swap this class for your own
        implementation (e.g. talking to a vLLM server that receives the
        image bytes) as long as it exposes the same method signature.
        """

        object_name = object_name.strip()
        if not object_name:
            return False

        # Fallback description to avoid empty prompts.
        desc_for_prompt = description or f"An image located at '{image_path}'."

        prompt = (
            "You are a strict classifier. Given an image description and an "
            "object name, answer ONLY with 'yes' or 'no' (lowercase) on "
            "whether the object clearly appears in the described image.\n\n"
            f"Description: {desc_for_prompt}\n"
            f"Object: {object_name}\n"
            "Answer (yes or no):"
        )

        outputs = self._pipe(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0.0,
            num_return_sequences=1,
        )

        if not outputs:
            return False

        # HF text-generation pipeline usually returns List[Dict[str, str]].
        raw = outputs[0].get("generated_text", "").strip().lower()
        if not raw:
            return False

        # Look at the first token/word only.
        first_token = raw.split()[0]
        return first_token.startswith("y")


def build_default_client() -> TextOnlyLLMClient:
    """Factory used by the CLI.

    You can modify this function to instead construct a client that
    talks to a vLLM HTTP server, OpenAI-compatible endpoint, etc.
    """

    return TextOnlyLLMClient()

