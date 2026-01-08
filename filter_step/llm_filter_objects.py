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
from typing import Dict, Iterable, Optional
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError


# -----------------------------------------------------------------------------
# Text-only LLM filter: "Could an open-set object detector detect this category?"
#
# Supported modes:
#
# (1) TXT mode (legacy):
#   Input file format (one per line):
#     wall<TAB>256
#     pillar<TAB>200
#     wires<TAB>173
#     ...
#
#   Output:
#    - <stem>_detectability_checked.txt  (all objects with yes/no)
#    - <stem>_not_detectable.txt         (only "no" objects, includes counts)
#    - <stem>_detectability_summary.txt  (totals + percentage)
#
# (2) CSV mode:
#   Input: a pipeline CSV that includes at least:
#     - image_path
#     - objects_vllm_said_no
#
#   Output:
#    - <stem>__objects_vllm_said_no_detectability_summary.csv
#        one row per unique object in objects_vllm_said_no:
#        object, count, image_paths, llm_filter_answer
#    - <stem>__with_objects_vllm_said_no_detectability.csv
#        a copy of the input CSV with one extra column containing the per-object yes/no answers.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectCount:
    name: str
    count: int


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

        if last_exc is not None:
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


_SEP_RE = re.compile(r"[,;\t ]+")


def _parse_object_counts(text: str) -> list[ObjectCount]:
    out: list[ObjectCount] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in _SEP_RE.split(line) if p]
        if len(parts) < 2:
            continue
        name = " ".join(parts[:-1]).strip()
        try:
            count = int(float(parts[-1]))
        except Exception:
            continue
        if not name:
            continue
        out.append(ObjectCount(name=name, count=count))
    return out


def _chunks(items: list[ObjectCount], n: int) -> Iterable[list[ObjectCount]]:
    n = max(1, int(n))
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _few_shot_examples() -> str:
    # Kept short & format-focused. These are "few-shot in prompt" examples.
    return (
        "Few-shot examples (format to copy exactly):\n"
        "Input objects:\n"
        "wall\t256\n"
        "sky\t100\n"
        "happiness\t8\n"
        "text\t12\n"
        "rust\t30\n"
        "\n"
        "Output:\n"
        "wall\tAnswer: yes\n"
        "sky\tAnswer: yes\n"
        "happiness\tAnswer: no\n"
        "text\tAnswer: no\n"
        "rust\tAnswer: no\n"
    )


def _build_prompt(objects: list[ObjectCount]) -> tuple[str, str]:
    system = (
        "You are a computer vision expert. Your job: decide whether a category word "
        "could be detected (in principle) by an open-vocabulary / open-set object detector "
        "from a single RGB street-level image.\n"
        "\n"
        "Answer YES when the word is a visually observable physical object / stuff category "
        "that could be localized or segmented (e.g., car, fence, wall, sky, road, wires).\n"
        "Answer NO when the word is abstract, a non-visual concept, a property/material/condition "
        "(e.g., happiness, cleanliness, wetness, rust), a verb/action, or something that is not "
        "a stable visual category.\n"
        "\n"
        "You must follow the output format exactly."
    )
    lines = "\n".join(f"{o.name}\t{o.count}" for o in objects)
    user = (
        f"{_few_shot_examples()}\n\n"
        "Now do the same for these input objects.\n"
        "Rules:\n"
        "- Output EXACTLY one line per input line, same order.\n"
        "- Output format per line: <object>\\tAnswer: yes|no\n"
        "- Only 'yes' or 'no' (lowercase).\n"
        "- No extra text.\n"
        "\n"
        "Input objects:\n"
        f"{lines}\n"
        "\n"
        "Output:\n"
    )
    return system, user


_ANSWER_RE = re.compile(r"\b(answer\s*:\s*)?(yes|no)\b", re.IGNORECASE)


def _parse_answers_in_order(
    requested: list[ObjectCount], model_text: str
) -> Optional[list[bool]]:
    lines = [ln.strip() for ln in (model_text or "").splitlines() if ln.strip()]
    if len(lines) != len(requested):
        return None
    out: list[bool] = []
    for obj, ln in zip(requested, lines):
        # Prefer matching object prefix, but don't be overly strict.
        # Expected: "<obj>\tAnswer: yes"
        if obj.name.lower() not in ln.lower():
            # Still allow, as long as there is a yes/no token.
            pass
        m = _ANSWER_RE.search(ln)
        if not m:
            return None
        out.append(m.group(2).lower() == "yes")
    return out


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


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

        # Try device_map=auto when on CUDA; fall back to explicit .to(device).
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
        """Load model with dtype in a transformers-version-compatible way.

        Newer transformers deprecates `torch_dtype` in favor of `dtype`.
        """
        kwargs = {"trust_remote_code": True}
        if device_map is not None:
            kwargs["device_map"] = device_map
        try:
            # transformers >= 4.47-ish
            return self._AutoModelForCausalLM.from_pretrained(self.cfg.model, dtype=dtype, **kwargs)
        except TypeError:
            # Older transformers
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
            # Phi-3(.5) remote code can be incompatible with some transformers cache implementations.
            # Disabling KV-cache avoids `past_key_values.seen_tokens` issues.
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


def _detectability_flags_for_objects(
    *,
    objects: list[ObjectCount],
    client: object,
    batch_size: int,
) -> list[bool]:
    """Return one boolean per ObjectCount in the same order (True=yes detectable)."""
    if not objects:
        return []

    flags_all: list[bool] = []
    num_batches = (len(objects) + max(1, int(batch_size)) - 1) // max(1, int(batch_size))
    processed = 0

    for batch_idx, batch in enumerate(_chunks(objects, batch_size), start=1):
        print(
            f"[{_ts()}] batch {batch_idx}/{num_batches} (processing {len(batch)} items; processed {processed}/{len(objects)})",
            flush=True,
        )
        system, user = _build_prompt(batch)
        resp = getattr(client, "chat")(system=system, user=user)
        flags = _parse_answers_in_order(batch, resp)
        if flags is None:
            # Fallback: ask per-object if the batch response was malformed.
            flags = []
            for o in batch:
                sys1, user1 = _build_prompt([o])
                resp1 = getattr(client, "chat")(system=sys1, user=user1)
                parsed = _parse_answers_in_order([o], resp1)
                flags.append(bool(parsed[0]) if parsed else False)

        flags_all.extend(bool(x) for x in flags)
        processed += len(batch)

    if len(flags_all) != len(objects):
        raise RuntimeError(f"Internal error: flags_all length {len(flags_all)} != objects length {len(objects)}")
    return flags_all


def _norm_obj(s: str) -> str:
    return str(s or "").strip().lower()


@dataclass
class _ObjAgg:
    display_name: str
    count_rows: int = 0
    image_paths: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.image_paths is None:
            self.image_paths = set()


def _read_objects_vllm_said_no_from_csv(
    *,
    input_csv: Path,
    objects_col: str,
    image_col: str,
    encoding: str,
) -> Dict[str, _ObjAgg]:
    """
    Parse CSV and aggregate unique objects (normalized to lowercase) from objects_col.
    Aggregates:
      - count_rows: number of CSV rows in which the object appears (per-row de-duped)
      - image_paths: set of image paths for rows aligned to this object
    """
    # Support both package execution and direct script execution.
    if __package__ in (None, ""):  # pragma: no cover
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vllm_checker.object_utils import parse_objects_field  # type: ignore
    else:
        from ..object_utils import parse_objects_field  # type: ignore

    agg: Dict[str, _ObjAgg] = {}

    with input_csv.open("r", encoding=str(encoding), newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if objects_col not in headers:
            raise ValueError(f"Column not found: {objects_col}. Available columns: {headers}")
        if image_col not in headers:
            raise ValueError(f"Column not found: {image_col}. Available columns: {headers}")

        for row in reader:
            if not row:
                continue
            img = (row.get(image_col) or "").strip()
            raw = (row.get(objects_col) or "").strip()
            if not raw:
                continue
            objs = parse_objects_field(raw)
            if not objs:
                continue
            # Per-row de-dupe on normalized key (so "door. door" counts once for that row).
            uniq_keys: dict[str, str] = {}
            for o in objs:
                disp = str(o or "").strip()
                if not disp:
                    continue
                k = _norm_obj(disp)
                if not k:
                    continue
                if k not in uniq_keys:
                    uniq_keys[k] = disp

            for k, disp in uniq_keys.items():
                a = agg.get(k)
                if a is None:
                    a = _ObjAgg(display_name=disp)
                    agg[k] = a
                a.count_rows += 1
                if img:
                    a.image_paths.add(img)

    return agg


def run_detectability_on_csv(
    *,
    input_csv: Path,
    output_summary_csv: Path,
    output_augmented_csv: Path,
    objects_col: str,
    image_col: str,
    output_answers_col: str,
    client: object,
    model_label: str,
    batch_size: int,
    encoding: str,
    resume: bool = False,
) -> None:
    """
    Writes:
      - summary CSV: one row per unique object in objects_col with count + image_paths + yes/no answer
      - augmented CSV: copy of input with an extra column containing per-object yes/no answers
    """
    obj_agg = _read_objects_vllm_said_no_from_csv(
        input_csv=input_csv,
        objects_col=str(objects_col),
        image_col=str(image_col),
        encoding=str(encoding),
    )

    # Stable ordering: by descending count then name
    keys_sorted = sorted(obj_agg.keys(), key=lambda k: (-int(obj_agg[k].count_rows), obj_agg[k].display_name.lower()))
    objects = [ObjectCount(name=obj_agg[k].display_name, count=int(obj_agg[k].count_rows)) for k in keys_sorted]

    print(f"[{_ts()}] unique objects in {objects_col}: {len(objects)}", flush=True)

    # Resume support:
    # - If a previous summary CSV exists, reuse object->yes/no answers to avoid re-querying.
    # - Only query the LLM for objects that are new in the current input.
    key_to_yes: dict[str, bool] = {}
    if resume and output_summary_csv.is_file():
        try:
            with output_summary_csv.open("r", encoding="utf-8", newline="") as f_prev:
                r_prev = csv.DictReader(f_prev)
                for row in r_prev:
                    obj = str((row.get("object") or "")).strip()
                    ans = str((row.get("llm_filter_answer") or "")).strip().lower()
                    if not obj or ans not in {"yes", "no"}:
                        continue
                    key_to_yes[_norm_obj(obj)] = (ans == "yes")
            if key_to_yes:
                print(
                    f"[{_ts()}] [resume] loaded {len(key_to_yes)} object answers from: {output_summary_csv}",
                    flush=True,
                )
        except Exception as e:
            print(
                f"[{_ts()}] [resume] WARN: failed reading prior summary CSV; will re-query all objects. "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            key_to_yes = {}

    missing_keys = [k for k in keys_sorted if k not in key_to_yes]
    if missing_keys:
        missing_objects = [
            ObjectCount(name=obj_agg[k].display_name, count=int(obj_agg[k].count_rows))
            for k in missing_keys
        ]
        print(
            f"[{_ts()}] querying detectability for {len(missing_objects)} new objects "
            f"(batch_size={int(batch_size)})",
            flush=True,
        )
        missing_flags = _detectability_flags_for_objects(
            objects=missing_objects, client=client, batch_size=int(batch_size)
        )
        for k, flg in zip(missing_keys, missing_flags):
            key_to_yes[k] = bool(flg)

    # Summary CSV
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_summary_csv.open("w", encoding="utf-8", newline="") as f_sum:
        writer = csv.DictWriter(
            f_sum,
            fieldnames=[
                "timestamp",
                "input_csv",
                "model",
                "object",
                "count",
                "image_paths",
                "llm_filter_answer",
            ],
        )
        writer.writeheader()
        ts = _ts()
        for k in keys_sorted:
            a = obj_agg[k]
            ans = "yes" if key_to_yes.get(k, False) else "no"
            writer.writerow(
                {
                    "timestamp": ts,
                    "input_csv": str(input_csv),
                    "model": str(model_label),
                    "object": a.display_name,
                    "count": int(a.count_rows),
                    "image_paths": json.dumps(sorted(a.image_paths), ensure_ascii=False),
                    "llm_filter_answer": ans,
                }
            )

    # Augmented CSV: stream input -> output, adding a JSON dict mapping object->yes/no for that row.
    if __package__ in (None, ""):  # pragma: no cover
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from vllm_checker.object_utils import parse_objects_field  # type: ignore
    else:
        from ..object_utils import parse_objects_field  # type: ignore

    processed_image_paths: set[str] = set()
    out_mode = "w"
    if resume and output_augmented_csv.is_file() and output_augmented_csv.stat().st_size > 0:
        try:
            with output_augmented_csv.open("r", encoding="utf-8", newline="") as f_prev_out:
                r_prev_out = csv.DictReader(f_prev_out)
                if r_prev_out.fieldnames is None:
                    raise ValueError("output_augmented_csv has no header")
                if image_col not in (r_prev_out.fieldnames or []):
                    raise ValueError(f"output_augmented_csv missing image column '{image_col}'")
                for prev_row in r_prev_out:
                    img = (prev_row.get(image_col) or "").strip()
                    if img:
                        processed_image_paths.add(img)
            out_mode = "a"
            if processed_image_paths:
                print(
                    f"[{_ts()}] [resume] output already has {len(processed_image_paths)} unique {image_col} values; "
                    "will append only new rows.",
                    flush=True,
                )
        except Exception as e:
            raise ValueError(
                f"Failed to --resume from existing output_augmented_csv: {output_augmented_csv}\n"
                f"Reason: {type(e).__name__}: {e}"
            ) from e

    output_augmented_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding=str(encoding), newline="") as f_in, output_augmented_csv.open(
        out_mode, encoding="utf-8", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        in_fields = reader.fieldnames or []
        if objects_col not in in_fields:
            raise ValueError(f"Column not found: {objects_col}. Available columns: {in_fields}")
        out_fields = list(in_fields)
        if output_answers_col not in out_fields:
            out_fields.append(output_answers_col)

        writer = csv.DictWriter(f_out, fieldnames=out_fields, dialect=reader.dialect)
        if out_mode == "w":
            writer.writeheader()

        n = 0
        for row in reader:
            if not row:
                continue
            if processed_image_paths:
                img = (row.get(image_col) or "").strip()
                if img and img in processed_image_paths:
                    continue
            raw = (row.get(objects_col) or "").strip()
            answers: dict[str, str] = {}
            if raw:
                objs = parse_objects_field(raw)
                for o in objs:
                    disp = str(o or "").strip()
                    if not disp:
                        continue
                    k = _norm_obj(disp)
                    if not k:
                        continue
                    ans = "yes" if key_to_yes.get(k, False) else "no"
                    # Keep row's original object token as the key in the JSON output.
                    answers[disp] = ans

            row_out = dict(row)
            row_out[output_answers_col] = json.dumps(answers, ensure_ascii=False)
            writer.writerow(row_out)
            n += 1
            if n % 2000 == 0:
                print(f"[{_ts()}] wrote {n} augmented rows...", flush=True)

    print(f"[{_ts()}] wrote: {output_summary_csv}")
    print(f"[{_ts()}] wrote: {output_augmented_csv}")


def run_detectability(
    *,
    input_txt: Path,
    output_checked_txt: Path,
    output_not_detectable_txt: Path,
    output_summary_txt: Path,
    client: object,
    model_label: str,
    batch_size: int,
) -> None:
    raw = input_txt.read_text(encoding="utf-8", errors="replace")
    objects = _parse_object_counts(raw)
    if not objects:
        raise ValueError(f"No valid lines parsed from input file: {input_txt}")

    total_sum = sum(o.count for o in objects)
    not_detectable_sum = 0
    checked_lines: list[str] = []
    not_detectable_lines: list[str] = []

    # Create output files early so users can see them immediately, even if the first
    # LLM call takes a while (e.g., model warmup / long generation).
    output_checked_txt.parent.mkdir(parents=True, exist_ok=True)
    output_checked_txt.write_text("", encoding="utf-8")
    output_not_detectable_txt.write_text("", encoding="utf-8")

    num_batches = (len(objects) + max(1, int(batch_size)) - 1) // max(1, int(batch_size))
    processed = 0

    for batch_idx, batch in enumerate(_chunks(objects, batch_size), start=1):
        print(
            f"[{_ts()}] batch {batch_idx}/{num_batches} (processing {len(batch)} items; processed {processed}/{len(objects)})",
            flush=True,
        )
        system, user = _build_prompt(batch)
        resp = getattr(client, "chat")(system=system, user=user)
        flags = _parse_answers_in_order(batch, resp)
        if flags is None:
            # Fallback: ask per-object if the batch response was malformed.
            flags = []
            for o in batch:
                sys1, user1 = _build_prompt([o])
                resp1 = getattr(client, "chat")(system=sys1, user=user1)
                parsed = _parse_answers_in_order([o], resp1)
                flags.append(bool(parsed[0]) if parsed else False)

        for o, detectable in zip(batch, flags):
            yn = "yes" if detectable else "no"
            checked_line = f"{o.name}\t{o.count}\tAnswer: {yn}"
            checked_lines.append(checked_line)
            if not detectable:
                not_line = f"{o.name}\t{o.count}"
                not_detectable_lines.append(not_line)
                not_detectable_sum += int(o.count)
            processed += 1

        # Incremental writes so the user sees progress on disk.
        output_checked_txt.write_text("\n".join(checked_lines).rstrip() + "\n", encoding="utf-8")
        output_not_detectable_txt.write_text(
            "\n".join(not_detectable_lines).rstrip() + ("\n" if not_detectable_lines else ""),
            encoding="utf-8",
        )

    not_pct = (not_detectable_sum / total_sum * 100.0) if total_sum else 0.0

    output_summary_txt.write_text(
        (
            f"timestamp: {_ts()}\n"
            f"input_txt: {input_txt}\n"
            f"model: {model_label}\n"
            "\n"
            f"num_objects: {len(objects)}\n"
            f"total_sum_counts: {total_sum}\n"
            f"not_detectable_sum_counts: {not_detectable_sum}\n"
            f"not_detectable_percent: {not_pct:.6f}\n"
        ),
        encoding="utf-8",
    )

    print(f"[{_ts()}] wrote: {output_checked_txt}")
    print(f"[{_ts()}] wrote: {output_not_detectable_txt}")
    print(f"[{_ts()}] wrote: {output_summary_txt}")
    print(f"[{_ts()}] not_detectable_sum={not_detectable_sum} / total_sum={total_sum} => {not_pct:.4f}%")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Ask a Phi (or other) text LLM whether each category word could be detected "
            "by an open-set object detector.\n\n"
            "Supports two modes:\n"
            "- TXT mode: --input_txt (legacy)\n"
            "- CSV mode: --input_csv (writes summary CSV + augmented CSV)\n"
        )
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_txt", default=None, help="Path to input TXT (word<TAB>count per line).")
    g.add_argument("--input_csv", default=None, help="Path to input CSV (must include objects_vllm_said_no).")
    p.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to input file's directory.",
    )
    p.add_argument(
        "--csv_objects_col",
        default="objects_vllm_said_no",
        help="CSV column containing objects to filter (default: objects_vllm_said_no).",
    )
    p.add_argument(
        "--csv_image_col",
        default="image_path",
        help="CSV column containing image path (default: image_path).",
    )
    p.add_argument(
        "--output_summary_csv",
        default=None,
        help="CSV-mode: output path for per-object summary CSV (default: derived from input CSV).",
    )
    p.add_argument(
        "--output_augmented_csv",
        default=None,
        help="CSV-mode: output path for augmented CSV (default: derived from input CSV).",
    )
    p.add_argument(
        "--output_answers_col",
        default="objects_vllm_said_no_detectability",
        help=(
            "CSV-mode: name of the additional column written to the augmented CSV. "
            "The value is a JSON dict mapping object->yes|no (default: objects_vllm_said_no_detectability)."
        ),
    )
    p.add_argument(
        "--csv_encoding",
        default="utf-8-sig",
        help="CSV-mode: encoding for reading input CSV (default: utf-8-sig).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "CSV-mode: if the output files already exist, reuse existing object answers "
            "from the summary CSV and append only new rows to the augmented CSV."
        ),
    )
    p.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "openai_compat"],
        help="Backend: hf (local transformers) or openai_compat (HTTP /v1/chat/completions).",
    )
    p.add_argument("--batch_size", type=int, default=64, help="How many objects per LLM request.")
    p.add_argument(
        "--hf_model",
        default=None,
        help="HF model id for --backend hf (default: microsoft/Phi-3.5-mini-instruct).",
    )
    p.add_argument("--hf_device", default="auto", help="HF device: auto|cpu|cuda|cuda:0...")
    p.add_argument("--hf_dtype", default="auto", help="HF dtype: auto|float16|bfloat16|float32")
    p.add_argument(
        "--hf_max_new_tokens",
        type=int,
        default=2048,
        help="HF max_new_tokens per request (raise if batches are large).",
    )
    p.add_argument("--api_key", default=None, help="API key for OpenAI-compatible server (default: env OPENAI_API_KEY or 'EMPTY').")
    p.add_argument("--openai_base_url", default=None, help="OpenAI-compatible base URL (default: env OPENAI_BASE_URL or http://localhost:8000).")
    p.add_argument("--openai_model", default=None, help="Model name (default: env OPENAI_MODEL or microsoft/phi-3.5-mini-instruct).")
    p.add_argument("--timeout_s", type=float, default=None, help="HTTP timeout seconds (default: 120).")
    p.add_argument("--max_retries", type=int, default=None, help="Max retries for transient failures (default: 2).")
    p.add_argument("--retry_backoff_s", type=float, default=None, help="Retry backoff base seconds (default: 1.0).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    input_txt = Path(args.input_txt).resolve() if args.input_txt else None
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None

    if input_txt is not None and not input_txt.is_file():
        raise FileNotFoundError(f"input_txt not found: {input_txt}")
    if input_csv is not None and not input_csv.is_file():
        raise FileNotFoundError(f"input_csv not found: {input_csv}")

    base_path = input_txt if input_txt is not None else input_csv
    if base_path is None:
        raise RuntimeError("No input provided (expected --input_txt or --input_csv).")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else base_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = base_path.stem

    backend = str(args.backend or "hf").strip().lower()
    if backend == "hf":
        hf_model = (
            args.hf_model
            or os.environ.get("HF_MODEL")
            or os.environ.get("VLLM_HF_MODEL")
            or "microsoft/Phi-3.5-mini-instruct"
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
        model = args.openai_model or os.environ.get("OPENAI_MODEL") or "microsoft/phi-3.5-mini-instruct"
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

    if input_txt is not None:
        checked_path = out_dir / f"{stem}_detectability_checked.txt"
        not_path = out_dir / f"{stem}_not_detectable.txt"
        summary_path = out_dir / f"{stem}_detectability_summary.txt"

        run_detectability(
            input_txt=input_txt,
            output_checked_txt=checked_path,
            output_not_detectable_txt=not_path,
            output_summary_txt=summary_path,
            client=client,
            model_label=model_label,
            batch_size=int(args.batch_size),
        )
        return

    # CSV mode
    assert input_csv is not None
    summary_csv = (
        Path(args.output_summary_csv).resolve()
        if args.output_summary_csv
        else (out_dir / f"{stem}__objects_vllm_said_no_detectability_summary.csv")
    )
    augmented_csv = (
        Path(args.output_augmented_csv).resolve()
        if args.output_augmented_csv
        else (out_dir / f"{stem}__with_objects_vllm_said_no_detectability.csv")
    )

    run_detectability_on_csv(
        input_csv=input_csv,
        output_summary_csv=summary_csv,
        output_augmented_csv=augmented_csv,
        objects_col=str(args.csv_objects_col),
        image_col=str(args.csv_image_col),
        output_answers_col=str(args.output_answers_col),
        client=client,
        model_label=model_label,
        batch_size=int(args.batch_size),
        encoding=str(args.csv_encoding),
        resume=bool(args.resume),
    )


if __name__ == "__main__":  # pragma: no cover
    main()


