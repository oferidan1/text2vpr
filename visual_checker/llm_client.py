from dataclasses import dataclass
from typing import Optional
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import logging as hf_logging


@dataclass
class LLMConfig:
    model_name: str = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    device: Optional[int] = None  # GPU index; -1 for CPU; None=auto


class LLMClient:
    """
    Simple local Hugging Face transformers client for Phi-3.5-mini-instruct
    (or any compatible causal LM).
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

        # Make transformers as quiet as possible: only errors, no internal progress bars.
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Configure generation defaults on the model instead of passing them
        # every call, to avoid noisy "invalid generation flags" warnings.
        try:
            gen_cfg = model.generation_config
            if gen_cfg is not None:
                if hasattr(gen_cfg, "temperature"):
                    gen_cfg.temperature = self.config.temperature
                if hasattr(gen_cfg, "top_p"):
                    gen_cfg.top_p = self.config.top_p
        except Exception:
            # Best-effort; fall back to model defaults if anything goes wrong.
            pass

        # Decide device for the pipeline
        if self.config.device is not None:
            pipeline_device = int(self.config.device)
        else:
            pipeline_device = 0 if torch.cuda.is_available() else -1

        self.generator = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            device=pipeline_device,
        )

    def get_objects_from_caption(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """
        Call the local model with a fully formatted prompt and return the text response.
        
        Args:
            prompt: The fully formatted prompt to send to the model
            max_new_tokens: Optional override for max_new_tokens (if None, uses config default)
        """
        # Use provided max_new_tokens or fall back to config
        actual_max_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        
        outputs = self.generator(
            prompt,
            max_new_tokens=actual_max_tokens,
            # Deterministic when temperature == 0.0
            do_sample=self.config.temperature > 0.0,
            num_return_sequences=1,  # Ensure only one sequence is returned
        )

        if not outputs or "generated_text" not in outputs[0]:
            return ""

        generated_text = outputs[0]["generated_text"].strip()
        return self._postprocess_generated_text(generated_text)

    def get_objects_from_captions_batch(
        self,
        prompts: list[str],
        max_new_tokens: Optional[int] = None,
    ) -> list[str]:
        """
        Batched variant of get_objects_from_caption.

        Args:
            prompts: List of fully formatted prompts.
            max_new_tokens: Optional override for max_new_tokens for the whole batch.
                            If None, uses the config default.

        Returns:
            List of cleaned object/region strings, one per prompt.
        """
        if not prompts:
            return []

        actual_max_tokens = (
            max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        )

        # Hugging Face text-generation pipelines support batched prompts by passing
        # a list of strings. The output is typically List[List[dict]] when
        # num_return_sequences==1.
        outputs = self.generator(
            prompts,
            max_new_tokens=actual_max_tokens,
            do_sample=self.config.temperature > 0.0,
            num_return_sequences=1,
        )

        results: list[str] = []
        for item in outputs:
            if not item or "generated_text" not in item[0]:
                results.append("")
                continue

            generated_text = item[0]["generated_text"].strip()
            results.append(self._postprocess_generated_text(generated_text))

        return results

    def _postprocess_generated_text(self, generated_text: str) -> str:
        """
        Clean up and normalize raw model output into a single dot-separated list.
        """
        # Clean up the response: remove structured response markers
        # Common patterns: "- Support:", "- response:", "Support:", "response:", etc.
        lines = generated_text.split("\n")
        cleaned_lines = []

        # List of prefixes to remove
        prefixes_to_remove = [
            "- Support:",
            "- support:",
            "Support:",
            "support:",
            "- Response:",
            "- response:",
            "Response:",
            "response:",
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove any of the known prefixes
            for prefix in prefixes_to_remove:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break

            # Also remove leading "- " if present (but not if it's part of a word)
            if line.startswith("- ") and len(line) > 2:
                line = line[2:].strip()

            if line:
                cleaned_lines.append(line)

        # Join all cleaned lines with space
        full_text = " ".join(cleaned_lines)

        # If the prompt asked for an explicit end-of-list marker, truncate at it.
        # Be robust to the model emitting only part of the marker by checking
        # every possible prefix of the full marker and cutting at the earliest
        # occurrence of any such prefix.
        full_marker = "### END_OF_LIST ###"
        prefixes = [full_marker[:i] for i in range(1, len(full_marker) + 1)]

        cut_idx = -1
        for marker in prefixes:
            idx = full_text.find(marker)
            if idx != -1:
                if cut_idx == -1 or idx < cut_idx:
                    cut_idx = idx

        if cut_idx != -1:
            full_text = full_text[:cut_idx].strip()

        # Try to extract the first clean list of objects/regions
        # The model might generate the list multiple times, so we want the first complete one
        if ". " in full_text:
            # Split by ". " to get individual objects
            parts = [p.strip() for p in full_text.split(". ") if p.strip()]
        elif ", " in full_text:
            # Handle comma-separated format
            parts = [p.strip() for p in full_text.split(", ") if p.strip()]
        else:
            # Single line, might already be formatted
            parts = [full_text] if full_text else []

        # Remove duplicate objects (case-insensitive) while preserving order
        seen = set()
        unique_parts = []
        for part in parts:
            # Skip parts that look like metadata or incomplete
            part_clean = part.strip()
            if not part_clean or len(part_clean) < 2:
                continue

            part_lower = part_clean.lower()
            # Skip if this part is a duplicate
            if part_lower not in seen:
                seen.add(part_lower)
                unique_parts.append(part_clean)

        # Detect if the list repeats itself and truncate at the repetition point
        # Look for patterns where the beginning of the list repeats
        if len(unique_parts) > 6:
            # Check if first 3 items repeat later in the list
            first_three = tuple(unique_parts[:3])
            for i in range(3, len(unique_parts) - 2):
                if tuple(unique_parts[i:i+3]) == first_three:
                    # Found repetition, keep only up to this point
                    unique_parts = unique_parts[:i]
                    break

        # Join with ". " as specified in the prompt
        result = ". ".join(unique_parts)

        return result
