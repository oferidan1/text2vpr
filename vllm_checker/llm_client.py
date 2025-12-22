from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import time
from typing import Optional, Protocol
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

import csv
import os
from datetime import datetime
from pathlib import Path

# NOTE: We import heavyweight / optional deps lazily inside the client init so
# the CLI can fail with a clear message instead of a cryptic ModuleNotFoundError.


@dataclass
class LLMConfig:
    """Configuration for the local vision-language LLM backend."""

    # Default to a strong open vision-language model.
    model_name: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_new_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    device: Optional[int] = None  # GPU index; -1 for CPU; None=auto


@dataclass
class OpenAICompatConfig:
    """Configuration for an OpenAI-compatible (HTTP) chat-completions backend.

    This works with:
      - vLLM started in OpenAI-compatible mode
      - Other OpenAI-compatible servers that support image_url inputs
    """

    base_url: str = "http://localhost:8000"
    api_key: str = "EMPTY"
    model: str = "Qwen/Qwen2-VL-7B-Instruct"
    max_tokens: int = 8
    temperature: float = 0.0
    top_p: float = 0.9
    timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0


class LLMClient(Protocol):
    def is_object_in_image(
        self,
        *,
        image_path: str,
        object_name: str,
        description: Optional[str] = None,
    ) -> bool: ...

    def is_object_in_image_batch(
        self,
        queries: list[dict[str, Optional[str]]],
    ) -> list[bool]: ...


_UNEXPECTED_ANSWERS_CSV = (
    Path(__file__).resolve().parent / "outs" / "llm_unexpected_answers.csv"
)

_LLM_IO_CSV = Path(__file__).resolve().parent / "outs" / "llm_io_log.csv"


def _progress_enabled() -> bool:
    """Enable lightweight stage prints for long-running init/inference.

    Controlled via `VLLM_PROGRESS=1`. Kept separate from `VLLM_DEBUG` so users can
    get coarse progress without verbose logging.
    """
    return os.environ.get("VLLM_PROGRESS", "0") == "1"


def _p(msg: str) -> None:
    """Best-effort progress print."""
    if not _progress_enabled():
        return
    try:
        print(msg, flush=True)
    except Exception:
        return


def _log_llm_io(
    *,
    backend: str,
    model_name: str,
    image_path: str,
    object_name: str,
    description: str,
    prompt: str,
    raw_output: str,
) -> None:
    """Optional debug logging of prompts + raw outputs (best-effort).

    Enabled by setting `VLLM_LOG_IO=1`.
    """
    if os.environ.get("VLLM_LOG_IO", "0") != "1":
        return
    try:
        _LLM_IO_CSV.parent.mkdir(parents=True, exist_ok=True)
        file_exists = _LLM_IO_CSV.is_file()
        fieldnames = [
            "timestamp",
            "backend",
            "model_name",
            "image_path",
            "object_name",
            "description",
            "prompt",
            "raw_output",
        ]
        with _LLM_IO_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "backend": backend,
                    "model_name": model_name,
                    "image_path": image_path,
                    "object_name": object_name,
                    "description": description,
                    "prompt": prompt[:4000],
                    "raw_output": (raw_output or "")[:4000],
                }
            )
    except Exception:
        return


def _log_unexpected_answer(
    *,
    image_path: str,
    object_name: str,
    description: str,
    raw_output: str,
    first_token: str,
    model_name: str,
) -> None:
    """Append a row to a CSV with unexpected LLM answers."""

    try:
        out_dir = _UNEXPECTED_ANSWERS_CSV.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        file_exists = _UNEXPECTED_ANSWERS_CSV.is_file()
        fieldnames = [
            "timestamp",
            "model_name",
            "image_path",
            "object_name",
            "description",
            "raw_output",
            "first_token",
        ]

        with _UNEXPECTED_ANSWERS_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "model_name": model_name,
                    "image_path": image_path,
                    "object_name": object_name,
                    "description": description,
                    "raw_output": raw_output,
                    "first_token": first_token,
                }
            )
    except Exception:
        # Best-effort logging; never break main flow.
        return


class TextOnlyLLMClient:
    """Vision-language client backed by a Hugging Face VLM (e.g. Qwen2-VL).

    Despite the legacy name, this client now looks at **both** the image
    and text when answering yes/no questions about object presence.
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        _p("[llm] Stage: importing torch...")
        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            # Provide a clear error message for common binary/runtime failures.
            msg = (
                "PyTorch failed to import, so the local HuggingFace VLM backend cannot run.\n"
                f"Original error: {e}\n\n"
                "Common fixes:\n"
                "  - If you see `undefined symbol: iJIT_NotifyEvent`, install Intel ITT runtime:\n"
                "      conda install -y -c conda-forge ittapi\n"
                "  - Or reinstall PyTorch cleanly into the conda env (recommended):\n"
                "      conda install -y -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1\n"
                "  - Avoid mixing ~/.local site-packages with conda by running with:\n"
                "      PYTHONNOUSERSITE=1 ...\n"
            )
            raise ImportError(msg) from e

        # Store torch on the instance to avoid relying on a global import.
        self._torch = torch

        _p("[llm] Stage: importing transformers...")
        try:
            # NOTE: Qwen2-VL (and other VLMs) are typically *not* CausalLMs.
            # They are usually loaded via AutoModelForImageTextToText (newer HF),
            # or AutoModelForVision2Seq (older HF). We import what is available
            # and choose the right class at runtime.
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
            try:
                from transformers import AutoModelForImageTextToText  # type: ignore
            except Exception:  # pragma: no cover
                AutoModelForImageTextToText = None  # type: ignore
            try:
                from transformers import AutoModelForVision2Seq  # type: ignore
            except Exception:  # pragma: no cover
                AutoModelForVision2Seq = None  # type: ignore
            from transformers.utils import logging as hf_logging  # type: ignore
        except Exception as e:  # pragma: no cover
            import traceback

            tb = traceback.format_exc()
            local_site_hint = ""
            torchvision_hint = ""
            # A common gotcha: a user-site `packaging` (or other dep) shadows the conda env
            # and breaks transformers' dependency checks (often via a TypeError in
            # packaging.version.parse when _torch_version is None / unexpected).
            if "/.local/lib/python" in tb and "site-packages/packaging/version.py" in tb:
                local_site_hint = (
                    "\n\nIt looks like Python is importing dependencies from your user site-packages "
                    "(~/.local), which can conflict with your conda environment.\n"
                    "Quick fixes:\n"
                    "  - Run with user-site disabled:\n"
                    "      python -s vllm_checker/main.py ...\n"
                    "    (or: PYTHONNOUSERSITE=1 python vllm_checker/main.py ...)\n"
                    "  - Or remove/upgrade the user-site `packaging` so it doesn't shadow conda:\n"
                    "      python -m pip uninstall packaging\n"
                    "      conda install -y packaging\n"
                )

            # Another common issue: torch/torchvision binary mismatch causes torchvision C++ ops
            # (like `torchvision::nms`) to be missing at runtime, which breaks transformers' image utils.
            # Symptom: `RuntimeError: operator torchvision::nms does not exist`
            if "operator torchvision::nms does not exist" in tb:
                torchvision_hint = (
                    "\n\nIt looks like `torchvision` is installed but its compiled operators did not load "
                    "(common when torch/torchvision are mismatched or mixed across conda/pip channels).\n"
                    "Fix (recommended): reinstall a matching torch+torchvision pair from the same channel:\n"
                    "  - conda remove -y torchvision torchaudio pytorch pytorch-cuda\n"
                    "  - conda install -y -c pytorch -c nvidia pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.1\n"
                    "If you are running with `-s` / PYTHONNOUSERSITE=1, ensure required deps are installed into the conda env.\n"
                )
            msg = (
                "HuggingFace `transformers` (and its deps) are required for the local VLM backend.\n"
                f"Original error: {e}\n\n"
                "Fix (install into your active environment):\n"
                "  - conda install -y -c conda-forge transformers huggingface_hub\n"
                "    (or: python -m pip install -U transformers huggingface_hub)\n"
                "If you're running with PYTHONNOUSERSITE=1, ~/.local packages are ignored, so the deps\n"
                "must be installed inside the conda env.\n"
                f"{local_site_hint}"
                f"{torchvision_hint}"
            )
            raise ImportError(msg) from e

        _p("[llm] Stage: importing Pillow (PIL)...")
        try:
            from PIL import Image  # type: ignore
        except Exception as e:  # pragma: no cover
            msg = (
                "Pillow (`PIL`) is required to load images for the local VLM backend.\n"
                f"Original error: {e}\n\n"
                "Fix:\n"
                "  - conda install -y -c conda-forge pillow\n"
                "    (or: python -m pip install -U pillow)\n"
            )
            raise ImportError(msg) from e

        # Store lazy imports on the instance.
        self._Image = Image
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoModelForImageTextToText = AutoModelForImageTextToText
        self._AutoModelForVision2Seq = AutoModelForVision2Seq
        self._AutoProcessor = AutoProcessor
        self._hf_logging = hf_logging

        self.config = config or LLMConfig()
        self._debug_mode = os.environ.get("VLLM_DEBUG", "0") == "1"
        self._device = self._torch.device("cpu")

        # Make transformers quiet: only errors, no progress bars.
        self._hf_logging.set_verbosity_error()
        self._hf_logging.disable_progress_bar()
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        if self._debug_mode:
            print(f"      Loading VLM model '{self.config.model_name}'...")
        try:
            torch_cuda_ver = getattr(self._torch.version, "cuda", None)
        except Exception:
            torch_cuda_ver = None
        _p(f"[llm] Stage: torch={getattr(self._torch, '__version__', 'unknown')}, torch.version.cuda={torch_cuda_ver}")
        _p(f"[llm] Stage: selecting device (cuda available={self._torch.cuda.is_available()})...")

        # Decide device: explicit device index if provided, otherwise auto CUDA/CPU.
        if self.config.device is not None:
            if self.config.device >= 0 and self._torch.cuda.is_available():
                device_str = f"cuda:{self.config.device}"
            else:
                device_str = "cpu"
        else:
            device_str = "cuda" if self._torch.cuda.is_available() else "cpu"

        self._device = self._torch.device(device_str)
        _p(f"[llm] Stage: device selected -> {self._device}")
        if self._device.type != "cuda" and self._torch.cuda.is_available() is False:
            _p("[llm] Hint: CUDA is not available in this Python env (often a CPU-only torch build or wrong env).")

        # Use bfloat16 on GPU to save memory; float32 on CPU.
        torch_dtype = (
            self._torch.bfloat16 if self._device.type == "cuda" else self._torch.float32
        )

        _p("[llm] Stage: loading model weights (may download on first run)...")
        # Choose the correct AutoModel class:
        # - Prefer ImageTextToText when available (newer HF)
        # - Else prefer Vision2Seq (older HF; now deprecated)
        # - Fall back to CausalLM for text-only models / older transformers versions
        model_cls = self._AutoModelForCausalLM
        if self._AutoModelForVision2Seq is not None:
            model_cls = self._AutoModelForVision2Seq
        if self._AutoModelForImageTextToText is not None:
            model_cls = self._AutoModelForImageTextToText
        model = model_cls.from_pretrained(  # type: ignore[attr-defined]
            self.config.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        ).to(self._device)

        if self._debug_mode:
            dev_label = "GPU" if self._device.type == "cuda" else "CPU"
            print(f"      Model loaded on {dev_label} ({self._device})")
        _p("[llm] Stage: model loaded")

        if self._debug_mode:
            print(f"      Loading processor...")
        _p("[llm] Stage: loading processor/tokenizer...")
        self._processor = self._AutoProcessor.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self._debug_mode:
            print(f"      Processor ready!")
        _p("[llm] Stage: processor ready")

        self._model = model
        _p("[llm] Stage: client ready")

    def is_object_in_image(
        self,
        *,
        image_path: str,
        object_name: str,
        description: Optional[str] = None,
    ) -> bool:
        """Ask the model if an object is present in the given image."""

        object_name = object_name.strip()
        if not object_name:
            return False

        # Load image.
        try:
            image = self._Image.open(image_path).convert("RGB")
        except Exception:
            # If we cannot open the image, be conservative and say "not present".
            return False

        # Build a strict yes/no prompt.
        #
        # IMPORTANT: The VLM should receive ONLY the image + the yes/no question.
        # We keep `description` for logging/analysis, but we do NOT feed it to the model.
        desc_for_prompt = description.strip() if description else ""
        base_question = (
            f"Answer strictly with 'yes' or 'no'. "
            f"Is there a '{object_name}' clearly visible in this image?"
        )
        # Add a minimal in-prompt example to anchor the strict output format.
        prompt = (
            "Example:\n"
            "Input: Answer strictly with 'yes' or 'no'. "
            "Is there a 'tree canopies' clearly visible in this image?\n"
            "Expected output: yes.\n\n"
            f"Input: {base_question}\n"
            "Expected output:"
        )

        if self._debug_mode:
            print(f"        Running VLM inference...", end=" ", flush=True)

        # Some VLM processors (e.g. Qwen2-VL) require chat templating to produce the
        # correct model input format. When available, we use it; otherwise we fall
        # back to passing raw text+image.
        if hasattr(self._processor, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            try:
                templated = self._processor.apply_chat_template(  # type: ignore[attr-defined]
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                proc_inputs = self._processor(
                    text=[templated],
                    images=[image],
                    return_tensors="pt",
                    padding=True,
                )
            except Exception:
                proc_inputs = self._processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                )
        else:
            proc_inputs = self._processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )

        inputs = proc_inputs.to(self._device)

        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
        )

        if self._debug_mode:
            print("done!", flush=True)

        # Decode only the generated continuation when possible (otherwise some
        # models include the prompt in the decoded text).
        #
        # NOTE: `inputs` is often a HF `BatchEncoding` (dict-like but not always
        # `isinstance(..., dict)`), so be robust when extracting `input_ids`.
        input_len = 0
        try:
            input_ids = None
            if hasattr(inputs, "get"):
                # BatchEncoding / Mapping-like
                input_ids = inputs.get("input_ids")  # type: ignore[attr-defined]
            if input_ids is None and hasattr(inputs, "input_ids"):
                input_ids = getattr(inputs, "input_ids")
            if input_ids is not None and hasattr(input_ids, "shape"):
                input_len = int(input_ids.shape[1])
        except Exception:
            input_len = 0

        # If we couldn't infer input length, fall back to decoding everything.
        to_decode = generated_ids[:, input_len:] if input_len > 0 else generated_ids
        raw = self._processor.batch_decode(to_decode, skip_special_tokens=True)[0]
        _log_llm_io(
            backend="local_hf",
            model_name=self.config.model_name,
            image_path=image_path,
            object_name=object_name,
            description=desc_for_prompt,
            prompt=prompt,
            raw_output=raw,
        )
        return self._classify_raw_answer(
            image_path=image_path,
            object_name=object_name,
            description=desc_for_prompt,
            raw_output=raw,
        )

    def is_object_in_image_batch(
        self,
        queries: list[dict[str, Optional[str]]],
    ) -> list[bool]:
        """Batched variant of `is_object_in_image`.

        Each query dict must contain:
          - "image_path"
          - "object_name"
          - "description" (optional; may be None/empty)
        """
        if not queries:
            return []

        # Simple implementation: loop over queries and reuse the single
        # example path. This keeps the code straightforward while still
        # benefiting from the VLM.
        results: list[bool] = []
        for q in queries:
            image_path = (q.get("image_path") or "").strip()
            object_name = (q.get("object_name") or "").strip()
            description = q.get("description") or None
            if not object_name:
                results.append(False)
                continue

            present = self.is_object_in_image(
                image_path=image_path,
                object_name=object_name,
                description=description,
            )
            results.append(present)

        return results

    def _classify_raw_answer(
        self,
        *,
        image_path: str,
        object_name: str,
        description: str,
        raw_output: str,
    ) -> bool:
        """Shared helper to interpret raw LLM output as yes/no."""
        raw = raw_output.strip().lower()
        if not raw:
            return False

        # Look at the first token/word only.
        first_token = raw.split()[0]

        if first_token.startswith("y"):
            return True
        if first_token.startswith("n"):
            return False

        # If the answer is neither clear "yes" nor "no", log it for inspection
        # and fall back to treating it as "no" (object not present).
        _log_unexpected_answer(
            image_path=image_path,
            object_name=object_name,
            description=description,
            raw_output=raw,
            first_token=first_token,
            model_name=self.config.model_name,
        )
        return False


def _guess_mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    # Fallback that is commonly accepted.
    return "image/jpeg"


def _image_to_data_url(image_path: str) -> str:
    mime = _guess_mime_type(image_path)
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class OpenAICompatVLMClient:
    """VLM client that calls an OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, config: Optional[OpenAICompatConfig] = None) -> None:
        self.config = config or OpenAICompatConfig()
        self._debug_mode = os.environ.get("VLLM_DEBUG", "0") == "1"

        # Normalize base_url (no trailing slash).
        self._base_url = self.config.base_url.rstrip("/")
        self._endpoint = f"{self._base_url}/v1/chat/completions"

    def is_object_in_image(
        self,
        *,
        image_path: str,
        object_name: str,
        description: Optional[str] = None,
    ) -> bool:
        object_name = object_name.strip()
        if not object_name:
            return False

        try:
            img_url = _image_to_data_url(image_path)
        except Exception:
            return False

        # IMPORTANT: The VLM should receive ONLY the image + the yes/no question.
        # We keep `description` for logging/analysis, but we do NOT feed it to the model.
        desc_for_prompt = description.strip() if description else ""
        base_question = (
            "Answer strictly with 'yes' or 'no'. "
            f"Is there a '{object_name}' clearly visible in this image?"
        )
        # Add a minimal in-prompt example to anchor the strict output format.
        prompt = (
            "Example:\n"
            "Input: Answer strictly with 'yes' or 'no'. "
            "Is there a 'tree canopies' clearly visible in this image?\n"
            "Expected output: yes.\n\n"
            f"Input: {base_question}\n"
            "Expected output:"
        )

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_url}},
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
        }

        try:
            raw = self._post_chat_completions(payload)
        except Exception as e:
            # Log prompt even on failure so we can debug server-side issues.
            _log_llm_io(
                backend="openai_compat_http",
                model_name=self.config.model,
                image_path=image_path,
                object_name=object_name,
                description=desc_for_prompt,
                prompt=prompt,
                raw_output=f"[ERROR] {type(e).__name__}: {e}",
            )
            raise

        _log_llm_io(
            backend="openai_compat_http",
            model_name=self.config.model,
            image_path=image_path,
            object_name=object_name,
            description=desc_for_prompt,
            prompt=prompt,
            raw_output=raw,
        )
        return _classify_raw_answer_static(
            image_path=image_path,
            object_name=object_name,
            description=desc_for_prompt,
            raw_output=raw,
            model_name=self.config.model,
        )

    def is_object_in_image_batch(
        self,
        queries: list[dict[str, Optional[str]]],
    ) -> list[bool]:
        if not queries:
            return []
        # Simple, robust default: call one-by-one.
        out: list[bool] = []
        for q in queries:
            out.append(
                self.is_object_in_image(
                    image_path=(q.get("image_path") or "").strip(),
                    object_name=(q.get("object_name") or "").strip(),
                    description=q.get("description") or None,
                )
            )
        return out

    def _post_chat_completions(self, payload: dict) -> str:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = urllib_request.Request(
            self._endpoint, data=body, headers=headers, method="POST"
        )
        if self._debug_mode:
            print(f"        POST {self._endpoint} (model={self.config.model})")
        transient_http = {408, 409, 425, 429, 500, 502, 503, 504}
        last_exc: Optional[Exception] = None
        attempts = max(0, int(self.config.max_retries)) + 1
        for attempt in range(attempts):
            try:
                with urllib_request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    data = resp.read().decode("utf-8")
                last_exc = None
                break
            except HTTPError as e:
                last_exc = e
                try:
                    details = e.read().decode("utf-8")
                except Exception:
                    details = ""
                if int(getattr(e, "code", 0) or 0) in transient_http and attempt < attempts - 1:
                    sleep_s = float(self.config.retry_backoff_s) * (2**attempt)
                    if self._debug_mode:
                        print(f"        HTTP {e.code}; retrying in {sleep_s:.1f}s...")
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(
                    f"OpenAI-compatible request failed (HTTP {e.code}) to {self._endpoint}.\n"
                    f"Response body: {details}"
                ) from e
            except (URLError, TimeoutError) as e:
                last_exc = e
                if attempt < attempts - 1:
                    sleep_s = float(self.config.retry_backoff_s) * (2**attempt)
                    if self._debug_mode:
                        print(f"        Network error; retrying in {sleep_s:.1f}s... ({e})")
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(
                    f"Failed to reach OpenAI-compatible endpoint at {self._endpoint}.\n"
                    f"Reason: {e}"
                ) from e

        if last_exc is not None:  # pragma: no cover
            raise RuntimeError(
                f"Failed to reach OpenAI-compatible endpoint at {self._endpoint}.\n"
                f"Reason: {last_exc}"
            ) from last_exc

        try:
            parsed = json.loads(data)
            # OpenAI format: choices[0].message.content
            return (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or ""
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse OpenAI-compatible response as JSON.\nRaw: {data[:2000]}"
            ) from e


def _classify_raw_answer_static(
    *,
    image_path: str,
    object_name: str,
    description: str,
    raw_output: str,
    model_name: str,
) -> bool:
    raw = (raw_output or "").strip().lower()
    if not raw:
        return False

    first_token = raw.split()[0]
    if first_token.startswith("y"):
        return True
    if first_token.startswith("n"):
        return False

    _log_unexpected_answer(
        image_path=image_path,
        object_name=object_name,
        description=description,
        raw_output=raw,
        first_token=first_token,
        model_name=model_name,
    )
    return False


def build_default_client() -> LLMClient:
    """Best-effort default client selection.

    - If `OPENAI_BASE_URL` (or `VLLM_OPENAI_BASE_URL`) is set, prefer HTTP backend
      to avoid requiring local PyTorch/transformers.
    - Otherwise, use the local HuggingFace backend.
    """

    base_url = os.environ.get("VLLM_OPENAI_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
        model = (
            os.environ.get("OPENAI_MODEL")
            or os.environ.get("VLLM_OPENAI_MODEL")
            or OpenAICompatConfig.model
        )
        timeout_s = float(
            os.environ.get("VLLM_OPENAI_TIMEOUT_S")
            or os.environ.get("OPENAI_TIMEOUT_S")
            or OpenAICompatConfig.timeout_s
        )
        max_retries = int(
            os.environ.get("VLLM_OPENAI_MAX_RETRIES")
            or os.environ.get("OPENAI_MAX_RETRIES")
            or OpenAICompatConfig.max_retries
        )
        retry_backoff_s = float(
            os.environ.get("VLLM_OPENAI_RETRY_BACKOFF_S")
            or os.environ.get("OPENAI_RETRY_BACKOFF_S")
            or OpenAICompatConfig.retry_backoff_s
        )
        return OpenAICompatVLMClient(
            OpenAICompatConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                timeout_s=timeout_s,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )
        )

    # No explicit HTTP config: prefer local backend *if* PyTorch is available
    # and sufficiently new for the installed `transformers`. If torch is
    # missing (or too old), try a sensible HTTP fallback before failing.
    try:
        import torch  # type: ignore

        def _torch_is_modern_enough() -> bool:
            # `transformers` increasingly assumes PyTorch >= 2.1. Some setups
            # (e.g. torch 2.0.x) will import but be treated as "disabled" by
            # transformers, leading to confusing runtime ImportErrors.
            v = getattr(torch, "__version__", "") or ""
            # Extract leading X.Y (ignore +cu121 / .post / etc.)
            parts: list[int] = []
            for chunk in v.split("+", 1)[0].split(".", 3):
                num = ""
                for ch in chunk:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                if num:
                    parts.append(int(num))
                if len(parts) >= 2:
                    break
            if len(parts) < 2:
                return True  # Can't parse: assume ok and let client init decide.
            major, minor = parts[0], parts[1]
            return (major, minor) >= (2, 1)

        torch_ok = _torch_is_modern_enough()
    except Exception as e:  # pragma: no cover
        torch_ok = False
        # Optional fallback: many users run vLLM in OpenAI-compatible mode locally.
        fallback_base_url = os.environ.get("VLLM_OPENAI_FALLBACK_URL") or "http://localhost:8000"

        def _probe_openai_compat(url: str) -> bool:
            url = url.rstrip("/")
            endpoint = f"{url}/v1/models"
            req = urllib_request.Request(endpoint, method="GET")
            try:
                with urllib_request.urlopen(req, timeout=2.0) as resp:
                    return 200 <= int(getattr(resp, "status", 200)) < 300
            except Exception:
                return False

        if _probe_openai_compat(fallback_base_url):
            api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
            model = (
                os.environ.get("OPENAI_MODEL")
                or os.environ.get("VLLM_OPENAI_MODEL")
                or OpenAICompatConfig.model
            )
            if os.environ.get("VLLM_DEBUG", "0") == "1":
                print(
                    "      PyTorch is not available; falling back to OpenAI-compatible HTTP backend at "
                    f"{fallback_base_url} (model={model})."
                )
            return OpenAICompatVLMClient(
                OpenAICompatConfig(base_url=fallback_base_url, api_key=api_key, model=model)
            )

        msg = (
            "No LLM backend is available.\n\n"
            "Tried local HuggingFace VLM backend, but PyTorch failed to import:\n"
            f"  {e}\n\n"
            "Fix options:\n"
            "  - Install PyTorch in your active environment (enables local backend), OR\n"
            "  - Run an OpenAI-compatible server (e.g. vLLM) and set:\n"
            "      OPENAI_BASE_URL=http://<host>:<port>\n"
            "    (or VLLM_OPENAI_BASE_URL)\n"
            "    Optionally also set OPENAI_MODEL / OPENAI_API_KEY.\n\n"
            f"Note: I also tried the default fallback URL {fallback_base_url} but it was not reachable.\n"
        )
        raise ImportError(msg) from e

    if not torch_ok:  # pragma: no cover
        fallback_base_url = os.environ.get("VLLM_OPENAI_FALLBACK_URL") or "http://localhost:8000"
        if os.environ.get("VLLM_DEBUG", "0") == "1":
            try:
                import torch as _t  # type: ignore

                print(
                    f"      Detected torch={getattr(_t, '__version__', 'unknown')} which is too old for this setup; "
                    f"trying OpenAI-compatible HTTP backend at {fallback_base_url}."
                )
            except Exception:
                pass

        def _probe_openai_compat(url: str) -> bool:
            url = url.rstrip("/")
            endpoint = f"{url}/v1/models"
            req = urllib_request.Request(endpoint, method="GET")
            try:
                with urllib_request.urlopen(req, timeout=2.0) as resp:
                    return 200 <= int(getattr(resp, "status", 200)) < 300
            except Exception:
                return False

        if _probe_openai_compat(fallback_base_url):
            api_key = os.environ.get("OPENAI_API_KEY") or "EMPTY"
            model = (
                os.environ.get("OPENAI_MODEL")
                or os.environ.get("VLLM_OPENAI_MODEL")
                or OpenAICompatConfig.model
            )
            return OpenAICompatVLMClient(
                OpenAICompatConfig(base_url=fallback_base_url, api_key=api_key, model=model)
            )

        raise ImportError(
            "Local HuggingFace VLM backend is not usable because your PyTorch is too old for this `transformers` "
            "installation.\n\n"
            "Fix options:\n"
            "  - Upgrade PyTorch to >= 2.1 in your active environment, OR\n"
            "  - Run an OpenAI-compatible server (e.g. vLLM) and set:\n"
            "      OPENAI_BASE_URL=http://<host>:<port>\n"
            "    (or VLLM_OPENAI_BASE_URL)\n"
            f"\nI also tried the default fallback URL {fallback_base_url} but it was not reachable.\n"
        )

    # Local HF backend: allow overriding the model via env var so users can pick
    # a smaller model when memory is limited.
    local_model = (
        os.environ.get("VLLM_HF_MODEL")
        or os.environ.get("LOCAL_HF_MODEL")
        or os.environ.get("HF_MODEL")
        or LLMConfig.model_name
    )
    return TextOnlyLLMClient(LLMConfig(model_name=str(local_model)))

