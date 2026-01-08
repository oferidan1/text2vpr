from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

# Support both:
# - `python -m vllm_checker.anlysis.llm_validate_objects_by_description ...` (package execution)
# - `python vllm_checker/anlysis/llm_validate_objects_by_description.py ...` (direct script execution)
if __package__ in (None, ""):  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vllm_checker.object_utils import join_objects, parse_objects_field
else:
    from ..object_utils import join_objects, parse_objects_field


# -----------------------------------------------------------------------------
# Text-only QA: validate that `objects_vllm_said_no` are actually implied by the row's `description`.
#
# For each row:
#   - input: description (caption-like text) + objects list
#   - output: per-object yes/no:
#       YES => the description supports that object being present / a real object mention
#       NO  => likely a mistake / not supported / not an object in this context
#
# Few-shot examples:
#   - default small examples are included
#   - can be overridden via --prompt_examples_path
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Optional: hardcoded few-shot examples (edit this block directly).
#
# If non-empty, this text is inserted verbatim before the instructions, and is
# used when you do NOT pass --prompt_examples_path.
#
# Priority order:
#   1) --prompt_examples_path (if readable)
#   2) HARD_CODED_FEW_SHOT_EXAMPLES (if non-empty)
#   3) built-in defaults (unless --prompt_disable_default_examples)
# -----------------------------------------------------------------------------
HARD_CODED_FEW_SHOT_EXAMPLES = r"""
description: Green satellite dish, white building facade, potted plants, tall utility pole with dense black wires, green metal fence, red mailbox, two-story peach/pink colored building facade, dark brown horizontal slatted overhang, corrugated roof, multiple satellite dishes on the roof, more green metal fence.
Objects:
dish
facade
red
two-story
Output:
dish	Answer: no
facade	Answer: no
red	Answer: no
two-story	Answer: no

""".strip()


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class OpenAICompatTextConfig:
    base_url: str = "http://localhost:8000"
    api_key: str = "EMPTY"
    model: str = "microsoft/phi-3.5-mini-instruct"
    temperature: float = 0.0
    top_p: float = 0.9
    max_tokens: int = 2048
    timeout_s: float = 120.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0


class OpenAICompatTextClient:
    """Minimal OpenAI-compatible chat client for text-only prompts."""

    def __init__(self, cfg: OpenAICompatTextConfig) -> None:
        self.cfg = cfg
        self._base_url = (cfg.base_url or "").rstrip("/")
        self._endpoint = f"{self._base_url}/v1/chat/completions"

    def chat(self, *, system: str, user: str) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": float(self.cfg.temperature),
            "top_p": float(self.cfg.top_p),
            "max_tokens": int(self.cfg.max_tokens),
        }
        return self._post_chat_completions(payload)

    def _post_chat_completions(self, payload: dict) -> str:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.cfg.api_key}",
        }
        req = urllib_request.Request(self._endpoint, data=body, headers=headers, method="POST")
        transient_http = {408, 409, 425, 429, 500, 502, 503, 504}

        attempts = max(0, int(self.cfg.max_retries)) + 1
        last_exc: Optional[Exception] = None
        data = ""
        for attempt in range(attempts):
            try:
                with urllib_request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
                    data = resp.read().decode("utf-8")
                last_exc = None
                break
            except HTTPError as e:
                last_exc = e
                try:
                    details = e.read().decode("utf-8")
                except Exception:
                    details = ""
                code = int(getattr(e, "code", 0) or 0)
                if code in transient_http and attempt < attempts - 1:
                    time.sleep(float(self.cfg.retry_backoff_s) * (2**attempt))
                    continue
                raise RuntimeError(
                    f"OpenAI-compatible request failed (HTTP {code}) to {self._endpoint}.\n"
                    f"Response body: {details[:4000]}"
                ) from e
            except (URLError, TimeoutError) as e:
                last_exc = e
                if attempt < attempts - 1:
                    time.sleep(float(self.cfg.retry_backoff_s) * (2**attempt))
                    continue
                raise RuntimeError(
                    f"Failed to reach OpenAI-compatible endpoint at {self._endpoint}.\nReason: {e}"
                ) from e

        if last_exc is not None:  # pragma: no cover
            raise RuntimeError(
                f"Failed to reach OpenAI-compatible endpoint at {self._endpoint}.\nReason: {last_exc}"
            ) from last_exc

        try:
            parsed = json.loads(data)
            return (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ) or ""
        except Exception as e:
            raise RuntimeError(
                f"Failed to parse OpenAI-compatible response as JSON.\nRaw: {data[:2000]}"
            ) from e


@dataclass(frozen=True)
class HFTextConfig:
    model: str = "microsoft/Phi-3.5-mini-instruct"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    device: str = "auto"  # auto|cpu|cuda|cuda:0...
    dtype: str = "auto"  # auto|float16|bfloat16|float32


class HFTextClient:
    """Local HuggingFace Transformers text-only chat client."""

    def __init__(self, cfg: HFTextConfig) -> None:
        self.cfg = cfg
        self._debug = os.environ.get("VLLM_DEBUG", "0") == "1"

        try:
            import torch  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "PyTorch is required for --backend hf.\n"
                "Fix: install torch in your environment (conda/pip), or use --backend openai_compat.\n"
                f"Original error: {e}"
            ) from e

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "HuggingFace transformers is required for --backend hf.\n"
                "Fix: pip/conda install transformers.\n"
                f"Original error: {e}"
            ) from e

        self._torch = torch
        self._AutoModelForCausalLM = AutoModelForCausalLM
        self._AutoTokenizer = AutoTokenizer

        if self._debug:
            print(f"[{_ts()}] [hf] loading tokenizer: {self.cfg.model}", flush=True)
        self._tokenizer = self._AutoTokenizer.from_pretrained(
            self.cfg.model,
            trust_remote_code=True,
            use_fast=True,
        )

        if self._debug:
            print(f"[{_ts()}] [hf] loading model: {self.cfg.model}", flush=True)

        dtype = self._resolve_dtype(self.cfg.dtype)
        device = self._resolve_device(self.cfg.device)

        model = None
        if device.startswith("cuda"):
            try:
                model = self._from_pretrained_with_dtype(
                    device_map="auto",
                    dtype=dtype,
                )
            except Exception:
                model = None

        if model is None:
            model = self._from_pretrained_with_dtype(dtype=dtype)
            try:
                model = model.to(device)
            except Exception:
                pass

        self._model = model.eval()
        self._device = device

        if self._debug:
            print(f"[{_ts()}] [hf] ready (device={self._device}, dtype={self.cfg.dtype})", flush=True)

    def _resolve_device(self, dev: str) -> str:
        d = (dev or "auto").strip().lower()
        if d == "auto":
            return "cuda" if self._torch.cuda.is_available() else "cpu"
        return dev

    def _resolve_dtype(self, dtype: str):
        d = (dtype or "auto").strip().lower()
        if d == "auto":
            return self._torch.float16 if self._torch.cuda.is_available() else self._torch.float32
        if d in {"float16", "fp16"}:
            return self._torch.float16
        if d in {"bfloat16", "bf16"}:
            return self._torch.bfloat16
        if d in {"float32", "fp32"}:
            return self._torch.float32
        return self._torch.float16 if self._torch.cuda.is_available() else self._torch.float32

    def _from_pretrained_with_dtype(self, *, dtype, device_map: Optional[str] = None):
        """Load model with dtype in a transformers-version-compatible way."""
        kwargs = {"trust_remote_code": True}
        if device_map is not None:
            kwargs["device_map"] = device_map
        try:
            return self._AutoModelForCausalLM.from_pretrained(self.cfg.model, dtype=dtype, **kwargs)
        except TypeError:
            return self._AutoModelForCausalLM.from_pretrained(self.cfg.model, torch_dtype=dtype, **kwargs)

    def chat(self, *, system: str, user: str) -> str:
        prompt = self._build_chat_prompt(system=system, user=user)
        tok = self._tokenizer

        inputs = tok(prompt, return_tensors="pt")
        try:
            inputs = inputs.to(self._device)  # type: ignore[assignment]
        except Exception:
            pass

        do_sample = float(self.cfg.temperature) > 0.0
        gen = self._model.generate(
            **inputs,
            max_new_tokens=int(self.cfg.max_new_tokens),
            use_cache=False,
            do_sample=bool(do_sample),
            temperature=float(self.cfg.temperature) if do_sample else None,
            top_p=float(self.cfg.top_p) if do_sample else None,
            pad_token_id=getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None),
            eos_token_id=getattr(tok, "eos_token_id", None),
        )

        in_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        out_ids = gen[0][in_len:] if in_len > 0 else gen[0]
        return tok.decode(out_ids, skip_special_tokens=True).strip()

    def _build_chat_prompt(self, *, system: str, user: str) -> str:
        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template"):
            try:
                return tok.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


_ANSWER_RE = re.compile(r"\b(answer\s*:\s*)?(yes|no)\b", re.IGNORECASE)


def _load_examples_text(path: Optional[str]) -> str:
    if not path:
        return ""
    try:
        text = Path(path).expanduser().read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""
    if len(text) > 8000:
        text = text[:8000].rstrip() + "\n[...truncated examples...]"
    return text


def _default_few_shot_examples() -> str:
    # Format-focused (short) few-shot examples; users can override via --prompt_examples_path.
    return (
        "Few-shot examples (copy the output format exactly):\n"
        "Description: A narrow street with parked cars on both sides, a red bus in the distance, and pedestrians on the sidewalk.\n"
        "Objects:\n"
        "car\n"
        "bus\n"
        "banana\n"
        "happiness\n"
        "Output:\n"
        "car\tAnswer: yes\n"
        "bus\tAnswer: yes\n"
        "banana\tAnswer: no\n"
        "happiness\tAnswer: no\n"
        "\n"
        "Description: A building facade with a large billboard. Utility poles and overhead wires cross the street.\n"
        "Objects:\n"
        "billboard\n"
        "wire\n"
        "tree\n"
        "Output:\n"
        "billboard\tAnswer: yes\n"
        "wire\tAnswer: yes\n"
        "tree\tAnswer: no\n"
    )


def _resolve_examples_text(
    *,
    prompt_examples_path: Optional[str],
    disable_default_examples: bool,
) -> str:
    """
    Resolve few-shot examples text according to the priority order documented above.
    """
    # 1) Explicit file path
    file_text = _load_examples_text(prompt_examples_path)
    if file_text:
        return file_text

    # 2) Hardcoded block (user-editable)
    hardcoded = (HARD_CODED_FEW_SHOT_EXAMPLES or "").strip()
    if hardcoded:
        if len(hardcoded) > 8000:
            hardcoded = hardcoded[:8000].rstrip() + "\n[...truncated examples...]"
        return hardcoded

    # 3) Built-in defaults
    if not disable_default_examples:
        return _default_few_shot_examples()

    return ""


def _build_prompt(
    *,
    description: str,
    objects: list[str],
    examples_text: str,
    disable_default_examples: bool,
) -> tuple[str, str]:
    system = (
        "You are a strict QA assistant for image captions.\n"
        "Given a textual description of an image and a list of candidate object words/phrases, "
        "decide for each object whether it is supported by the description.\n"
        "\n"
        "Answer YES when the object is explicitly mentioned or clearly implied by the description.\n"
        "Answer NO when the object is not supported, is abstract/non-object, or looks like a syntactic error.\n"
        "If uncertain, answer NO.\n"
        "\n"
        "You must follow the output format exactly."
    )
    ex = examples_text.strip()
    if not ex and not disable_default_examples:
        ex = _default_few_shot_examples()

    obj_lines = "\n".join(o for o in objects)
    user = (
        (ex + "\n\n") if ex else ""
        + "Now do the same for this input.\n"
        "Rules:\n"
        "- Output EXACTLY one line per input object, same order.\n"
        "- Output format per line: <object>\\tAnswer: yes|no\n"
        "- Only 'yes' or 'no' (lowercase).\n"
        "- No extra text.\n"
        "\n"
        f"Description: {description.strip()}\n"
        "Objects:\n"
        f"{obj_lines}\n"
        "\n"
        "Output:\n"
    )
    return system, user


def _parse_answers_in_order(requested_objects: list[str], model_text: str) -> Optional[list[bool]]:
    lines = [ln.strip() for ln in (model_text or "").splitlines() if ln.strip()]
    if len(lines) != len(requested_objects):
        return None
    out: list[bool] = []
    for obj, ln in zip(requested_objects, lines):
        if str(obj).strip().lower() not in ln.lower():
            # allow mismatch, as long as yes/no exists
            pass
        m = _ANSWER_RE.search(ln)
        if not m:
            return None
        out.append(m.group(2).lower() == "yes")
    return out


def _chunks(items: list[str], n: int) -> Iterable[list[str]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _norm_obj(s: str) -> str:
    return str(s or "").strip().lower()


@dataclass
class _ObjStats:
    display_name: str
    total_rows: int = 0
    yes_rows: int = 0
    no_rows: int = 0
    image_paths_no: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.image_paths_no is None:
            self.image_paths_no = set()


def run_csv_description_validation(
    *,
    input_csv: Path,
    output_csv: Path,
    summary_csv: Optional[Path],
    objects_col: str,
    description_col: str,
    image_col: str,
    answers_col: str,
    rejected_col: str,
    client: object,
    model_label: str,
    objects_batch_size: int,
    max_rows: Optional[int],
    encoding: str,
    prompt_examples_path: Optional[str],
    prompt_disable_default_examples: bool,
    resume: bool = False,
) -> None:
    examples_text = _resolve_examples_text(
        prompt_examples_path=prompt_examples_path,
        disable_default_examples=bool(prompt_disable_default_examples),
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary_csv is not None:
        summary_csv.parent.mkdir(parents=True, exist_ok=True)

    obj_stats: dict[str, _ObjStats] = {}
    processed_image_paths: set[str] = set()

    # Resume support:
    # - append to existing output_csv
    # - skip rows whose image_path already exists in output
    # - reconstruct per-object stats from existing answers_col JSON
    out_mode = "w"
    if resume and output_csv.is_file() and output_csv.stat().st_size > 0:
        try:
            with output_csv.open("r", encoding="utf-8", newline="") as f_prev:
                r_prev = csv.DictReader(f_prev)
                prev_fields = r_prev.fieldnames or []
                if not prev_fields:
                    raise ValueError("output_csv has no header")
                if image_col not in prev_fields:
                    raise ValueError(f"output_csv missing image column '{image_col}'")
                # We require answers_col to rebuild summary; if missing, we can still resume output,
                # but summary will only reflect newly processed rows.
                can_rebuild = answers_col in prev_fields

                for prev_row in r_prev:
                    img = (prev_row.get(image_col) or "").strip()
                    if img:
                        processed_image_paths.add(img)
                    if can_rebuild:
                        raw_ans = (prev_row.get(answers_col) or "").strip()
                        if raw_ans:
                            try:
                                ans_dict = json.loads(raw_ans)
                            except Exception:
                                ans_dict = {}
                            if isinstance(ans_dict, dict):
                                for o, a in ans_dict.items():
                                    o_str = str(o or "").strip()
                                    if not o_str:
                                        continue
                                    a_str = str(a or "").strip().lower()
                                    if a_str not in {"yes", "no"}:
                                        continue
                                    k = _norm_obj(o_str)
                                    st = obj_stats.get(k)
                                    if st is None:
                                        st = _ObjStats(display_name=o_str)
                                        obj_stats[k] = st
                                    st.total_rows += 1
                                    if a_str == "yes":
                                        st.yes_rows += 1
                                    else:
                                        st.no_rows += 1
                                        if img:
                                            st.image_paths_no.add(img)
            out_mode = "a"
            if processed_image_paths:
                print(
                    f"[{_ts()}] [resume] output already has {len(processed_image_paths)} unique {image_col} values; "
                    "will append only new rows.",
                    flush=True,
                )
        except Exception as e:
            raise ValueError(
                f"Failed to --resume from existing output_csv: {output_csv}\n"
                f"Reason: {type(e).__name__}: {e}"
            ) from e

    with input_csv.open("r", encoding=str(encoding), newline="") as f_in, output_csv.open(
        out_mode, encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        in_fields = reader.fieldnames or []
        for req in (objects_col, description_col, image_col):
            if req not in in_fields:
                raise ValueError(f"Column not found: {req}. Available columns: {in_fields}")

        out_fields = list(in_fields)
        if answers_col not in out_fields:
            out_fields.append(answers_col)
        if rejected_col not in out_fields:
            out_fields.append(rejected_col)
        writer = csv.DictWriter(f_out, fieldnames=out_fields, dialect=reader.dialect)
        if out_mode == "w":
            writer.writeheader()

        n_rows = 0
        for row in reader:
            if not row:
                continue
            n_rows += 1
            if max_rows is not None and int(max_rows) > 0 and n_rows > int(max_rows):
                break

            desc = (row.get(description_col) or "").strip()
            img = (row.get(image_col) or "").strip()
            if processed_image_paths and img and img in processed_image_paths:
                continue
            raw_objs = (row.get(objects_col) or "").strip()
            objs = parse_objects_field(raw_objs) if raw_objs else []
            objs = [o.strip() for o in objs if o and o.strip()]

            answers: dict[str, str] = {}
            rejected: list[str] = []

            if desc and objs:
                # Chunk objects for this row to keep prompts small.
                for batch in _chunks(objs, int(objects_batch_size)):
                    system, user = _build_prompt(
                        description=desc,
                        objects=batch,
                        examples_text=examples_text,
                        disable_default_examples=bool(prompt_disable_default_examples),
                    )
                    resp = getattr(client, "chat")(system=system, user=user)
                    flags = _parse_answers_in_order(batch, resp)
                    if flags is None:
                        # Fallback: ask per-object if response was malformed.
                        flags = []
                        for o in batch:
                            sys1, user1 = _build_prompt(
                                description=desc,
                                objects=[o],
                                examples_text=examples_text,
                                disable_default_examples=bool(prompt_disable_default_examples),
                            )
                            resp1 = getattr(client, "chat")(system=sys1, user=user1)
                            parsed = _parse_answers_in_order([o], resp1)
                            flags.append(bool(parsed[0]) if parsed else False)

                    for o, ok in zip(batch, flags):
                        ans = "yes" if ok else "no"
                        answers[o] = ans
                        if not ok:
                            rejected.append(o)

                        k = _norm_obj(o)
                        st = obj_stats.get(k)
                        if st is None:
                            st = _ObjStats(display_name=o)
                            obj_stats[k] = st
                        st.total_rows += 1
                        if ok:
                            st.yes_rows += 1
                        else:
                            st.no_rows += 1
                            if img:
                                st.image_paths_no.add(img)

            # Write augmented row
            row_out = dict(row)
            row_out[answers_col] = json.dumps(answers, ensure_ascii=False)
            row_out[rejected_col] = join_objects(rejected)
            writer.writerow(row_out)

            if n_rows % 500 == 0:
                print(f"[{_ts()}] processed {n_rows} rows...", flush=True)

    print(f"[{_ts()}] wrote: {output_csv}", flush=True)

    if summary_csv is not None:
        with summary_csv.open("w", encoding="utf-8", newline="") as f_sum:
            writer = csv.DictWriter(
                f_sum,
                fieldnames=[
                    "timestamp",
                    "input_csv",
                    "model",
                    "object",
                    "total_rows",
                    "yes_rows",
                    "no_rows",
                    "image_paths_no",
                ],
            )
            writer.writeheader()
            ts = _ts()
            # Sort by most rejected first
            for k in sorted(obj_stats.keys(), key=lambda kk: (-obj_stats[kk].no_rows, obj_stats[kk].display_name.lower())):
                st = obj_stats[k]
                writer.writerow(
                    {
                        "timestamp": ts,
                        "input_csv": str(input_csv),
                        "model": str(model_label),
                        "object": st.display_name,
                        "total_rows": int(st.total_rows),
                        "yes_rows": int(st.yes_rows),
                        "no_rows": int(st.no_rows),
                        "image_paths_no": json.dumps(sorted(st.image_paths_no), ensure_ascii=False),
                    }
                )
        print(f"[{_ts()}] wrote: {summary_csv}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Validate objects_vllm_said_no against the row description using a text-only LLM.\n\n"
            "Writes:\n"
            "- an augmented CSV with per-object yes/no answers (JSON dict)\n"
            "- an optional per-object summary CSV (counts + image list for rejected cases)\n"
        )
    )
    p.add_argument("--input_csv", required=True, help="Path to the input CSV.")
    p.add_argument("--output_dir", default=None, help="Optional output dir (default: next to input).")
    p.add_argument("--output_csv", default=None, help="Optional explicit output CSV path.")
    p.add_argument("--summary_csv", default=None, help="Optional explicit summary CSV path.")

    p.add_argument("--objects_col", default="objects_vllm_said_no", help="Objects column (default: objects_vllm_said_no).")
    p.add_argument("--description_col", default="description", help="Description column (default: description).")
    p.add_argument("--image_col", default="image_path", help="Image path column (default: image_path).")
    p.add_argument(
        "--answers_col",
        default="objects_vllm_said_no_desc_answers",
        help="Name of the new column holding JSON dict of object->yes|no.",
    )
    p.add_argument(
        "--rejected_col",
        default="objects_vllm_said_no_desc_rejected",
        help="Name of the new column holding objects answered NO (dot-joined).",
    )

    p.add_argument("--backend", default="hf", choices=["hf", "openai_compat"], help="Text LLM backend.")
    p.add_argument("--batch_size", type=int, default=32, help="Max objects per request within a row (default: 32).")
    p.add_argument("--max_rows", type=int, default=None, help="Optional: process only first N rows.")
    p.add_argument("--csv_encoding", default="utf-8-sig", help="Input CSV encoding (default: utf-8-sig).")
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If output_csv already exists, append to it and skip rows whose image_path "
            "already appears in the existing output. Summary CSV is rewritten at the end."
        ),
    )

    p.add_argument(
        "--prompt_examples_path",
        default=None,
        help="Optional path to a text file with few-shot examples to include verbatim before the instructions.",
    )
    p.add_argument(
        "--prompt_disable_default_examples",
        action="store_true",
        help="Disable the built-in few-shot examples.",
    )

    # HF backend
    p.add_argument("--hf_model", default=None, help="HF model id for --backend hf.")
    p.add_argument("--hf_device", default="auto", help="HF device: auto|cpu|cuda|cuda:0...")
    p.add_argument("--hf_dtype", default="auto", help="HF dtype: auto|float16|bfloat16|float32")
    p.add_argument("--hf_max_new_tokens", type=int, default=2048, help="HF max_new_tokens per request.")

    # OpenAI-compatible backend
    p.add_argument("--api_key", default=None, help="API key (default: env OPENAI_API_KEY or 'EMPTY').")
    p.add_argument("--openai_base_url", default=None, help="Base URL (default: env OPENAI_BASE_URL or http://localhost:8000).")
    p.add_argument("--openai_model", default=None, help="Model name (default: env OPENAI_MODEL or microsoft/phi-3.5-mini-instruct).")
    p.add_argument("--timeout_s", type=float, default=None, help="HTTP timeout seconds (default: 120).")
    p.add_argument("--max_retries", type=int, default=None, help="Max retries for transient failures (default: 2).")
    p.add_argument("--retry_backoff_s", type=float, default=None, help="Retry backoff base seconds (default: 1.0).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = input_csv.stem
    output_csv = Path(args.output_csv).resolve() if args.output_csv else (out_dir / f"{stem}__desc_validated.csv")
    summary_csv = Path(args.summary_csv).resolve() if args.summary_csv else (out_dir / f"{stem}__desc_validated_summary.csv")

    backend = str(args.backend or "hf").strip().lower()
    if backend == "hf":
        hf_model = (
            args.hf_model
            or os.environ.get("HF_MODEL")
            or os.environ.get("VLLM_HF_MODEL")
            or HFTextConfig.model
        )
        hf_cfg = HFTextConfig(
            model=str(hf_model),
            max_new_tokens=int(args.hf_max_new_tokens),
            device=str(args.hf_device),
            dtype=str(args.hf_dtype),
        )
        client = HFTextClient(hf_cfg)
        model_label = f"hf:{hf_cfg.model} device={hf_cfg.device} dtype={hf_cfg.dtype}"
    else:
        base_url = (
            args.openai_base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("VLLM_OPENAI_BASE_URL")
            or "http://localhost:8000"
        )
        model = args.openai_model or os.environ.get("OPENAI_MODEL") or OpenAICompatTextConfig.model
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("VLLM_API_KEY") or "EMPTY"
        cfg = OpenAICompatTextConfig(
            base_url=str(base_url),
            api_key=str(api_key),
            model=str(model),
            timeout_s=float(args.timeout_s) if args.timeout_s is not None else OpenAICompatTextConfig.timeout_s,
            max_retries=int(args.max_retries) if args.max_retries is not None else OpenAICompatTextConfig.max_retries,
            retry_backoff_s=float(args.retry_backoff_s) if args.retry_backoff_s is not None else OpenAICompatTextConfig.retry_backoff_s,
        )
        client = OpenAICompatTextClient(cfg)
        model_label = f"openai_compat:{cfg.model} base_url={cfg.base_url}"

    run_csv_description_validation(
        input_csv=input_csv,
        output_csv=output_csv,
        summary_csv=summary_csv,
        objects_col=str(args.objects_col),
        description_col=str(args.description_col),
        image_col=str(args.image_col),
        answers_col=str(args.answers_col),
        rejected_col=str(args.rejected_col),
        client=client,
        model_label=model_label,
        objects_batch_size=int(args.batch_size),
        max_rows=(int(args.max_rows) if args.max_rows is not None else None),
        encoding=str(args.csv_encoding),
        prompt_examples_path=(str(args.prompt_examples_path) if args.prompt_examples_path else None),
        prompt_disable_default_examples=bool(args.prompt_disable_default_examples),
        resume=bool(args.resume),
    )


if __name__ == "__main__":  # pragma: no cover
    main()


