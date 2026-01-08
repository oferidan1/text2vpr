#!/usr/bin/env bash
set -euo pipefail

# Pipeline:
# 1) visual_checker/main.py produces: WashingtonDC_objects.csv (written incrementally)
# 2) visual_checker/sam3_checker.py watches that CSV and produces: sam3_realtime_progress.csv (written incrementally)
# 3) vllm_checker/main.py watches sam3_realtime_progress.csv and produces: sam3_realtime_progress_vllm_checked.csv
# 4) filter_step/llm_filter_objects_detectability_and_description.py watches vllm_checked.csv and produces:
#    sam3_realtime_progress_vllm_checked_llm_filter.csv
# (legacy, not used by this pipeline anymore)
# - filter_step/llm_filter_objects.py (detectability-only, per-object)
# - filter_step/llm_validate_objects_by_description.py (description-only, per-row)
#
# This script runs step 1, then runs steps 2, 3 (and step 4) in --watch mode so they keep
# re-reading the growing CSVs until they go idle.

usage() {
  cat <<'EOF'
Usage:
  run_full_quality_pipeline.sh [options]

Options (general):
  --city <CityName>   (default: WashingtonDC)
  --images_dir <path/to/images>  (default: /mnt/d/data/gsv_cities/Images/<CITY>)
  --input_csv <path>  Headered CSV with columns: image_path,description (image_path can be relative to the CSV dir)
  --images_root <path>  Root dir to resolve relative image_path values (default: dirname(--input_csv) when --input_csv is set)
  --input_csv_wait_seconds <seconds>  (default: 900)
  --watch_poll_minutes <minutes>  Poll interval for watcher steps 2/3/4 (default: WATCH_POLL_SEC/60)
  --resume            Resume step 1 output CSV/timings if they already exist

Options (Step 2: SAM3 checker):
  --sam3_box_threshold <float>   (default: 0.2) Confidence threshold for SAM3 predictions
  --sam3_checkpoint_path <path>  Use a local SAM3 checkpoint (.pt) to avoid HuggingFace download

Options (Step 3: vllm_checker prompt):
  --vllm_prompt_style <strict_yn|describe_then_yesno>
  --vllm_prompt_examples_path <path/to/examples.txt>
  --vllm_prompt_disable_default_examples

Options (Step 4: merged text-LLM filtering on vllm_checked CSV) [runs by default]:
  --filter_backend <hf|openai_compat>    Backend for text-only LLM steps (default: hf)
  --filter_model <model_id>             Text model id/name for step4 (default: microsoft/Phi-3.5-mini-instruct)
  --filter_device <auto|cpu|cuda|cuda:0> Device for step4 when using --filter_backend hf (default: cpu)
  --filter_batch_size <int>   Max objects per request within a row (default: 64)
  --filter_objects_col <name> Column to validate/filter (default: objects_vllm_said_no)
  --filter_prompt_examples_path <path>   Few-shot examples for the merged filter step
  --filter_prompt_disable_default_examples  Disable built-in examples for the merged filter step

You can also set equivalent env vars:
  IMAGES_DIR
  INPUT_CSV
  IMAGES_ROOT
  VLLM_PROMPT_STYLE
  VLLM_PROMPT_EXAMPLES_PATH
  VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES=1
  FILTER_BACKEND=hf|openai_compat
  FILTER_MODEL=microsoft/Phi-3.5-mini-instruct
  FILTER_DEVICE=cpu
  FILTER_BATCH_SIZE=64
  FILTER_OBJECTS_COL=objects_vllm_said_no
  FILTER_PROMPT_EXAMPLES_PATH=/path/to/examples.txt
  FILTER_PROMPT_DISABLE_DEFAULT_EXAMPLES=1
EOF
}

REPO_ROOT="/home/dsi/danbadu/git_projects/text2vpr"

# Defaults (can be overridden via CLI args below or env vars)
CITY="${CITY:-WashingtonDC}"
CAPTIONS_CSV="${CAPTIONS_CSV:-/home/dsi/oferidan/data/gsv_cities/gsv_cities_predictions_nan_fix.csv}"
INPUT_CSV="${INPUT_CSV:-}"
IMAGES_ROOT="${IMAGES_ROOT:-}"
CITY_SET="${CITY_SET:-0}"

# Track whether the user explicitly set filter device (via env or CLI), so we can
# safely "auto-upgrade" it to GPU when doing 2-GPU split routing.
FILTER_DEVICE_USER_SET="${FILTER_DEVICE_USER_SET:-0}"
if [[ -n "${FILTER_DEVICE-}" ]]; then
  FILTER_DEVICE_USER_SET="1"
fi

# Watch settings
WATCH_POLL_SEC="${WATCH_POLL_SEC:-60}"
WATCH_IDLE_MIN="${WATCH_IDLE_MIN:-20}"
INPUT_CSV_WAIT_SECONDS="${INPUT_CSV_WAIT_SECONDS:-900}"

# Step 2 settings (defaults match sam3_checkr.py)
SAM3_BOX_THRESHOLD="${SAM3_BOX_THRESHOLD:-0.2}"
SAM3_CHECKPOINT_PATH="${SAM3_CHECKPOINT_PATH:-}"

LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
DETACH="${DETACH:-0}" # if 1, start all steps and exit immediately (like running in 3 terminals)
RESUME="${RESUME:-0}"

# Step 3 prompt settings (defaults preserve legacy behavior)
VLLM_PROMPT_STYLE="${VLLM_PROMPT_STYLE:-strict_yn}"
VLLM_PROMPT_EXAMPLES_PATH="${VLLM_PROMPT_EXAMPLES_PATH:-}"
VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES="${VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES:-0}"

# Step 4 (merged filter) [runs by default]
FILTER_BACKEND="${FILTER_BACKEND:-hf}"
FILTER_MODEL="${FILTER_MODEL:-microsoft/Phi-3.5-mini-instruct}"
FILTER_DEVICE="${FILTER_DEVICE:-cpu}"
FILTER_BATCH_SIZE="${FILTER_BATCH_SIZE:-64}"
FILTER_OBJECTS_COL="${FILTER_OBJECTS_COL:-objects_vllm_said_no}"
FILTER_PROMPT_EXAMPLES_PATH="${FILTER_PROMPT_EXAMPLES_PATH:-}"
FILTER_PROMPT_DISABLE_DEFAULT_EXAMPLES="${FILTER_PROMPT_DISABLE_DEFAULT_EXAMPLES:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --city)
      CITY="${2:-}"
      CITY_SET="1"
      shift 2
      ;;
    --images_dir)
      IMAGES_DIR="${2:-}"
      shift 2
      ;;
    --input_csv)
      INPUT_CSV="${2:-}"
      shift 2
      ;;
    --images_root)
      IMAGES_ROOT="${2:-}"
      shift 2
      ;;
    --watch_poll_minutes)
      # Convenience: sampling interval in minutes (internally we track seconds).
      WATCH_POLL_SEC="$(( ${2:-0} * 60 ))"
      shift 2
      ;;
    --input_csv_wait_seconds)
      INPUT_CSV_WAIT_SECONDS="${2:-}"
      shift 2
      ;;
    --resume)
      RESUME="1"
      shift 1
      ;;
    --sam3_box_threshold)
      SAM3_BOX_THRESHOLD="${2:-}"
      shift 2
      ;;
    --sam3_checkpoint_path)
      SAM3_CHECKPOINT_PATH="${2:-}"
      shift 2
      ;;
    --vllm_prompt_style)
      VLLM_PROMPT_STYLE="${2:-}"
      shift 2
      ;;
    --vllm_prompt_examples_path)
      VLLM_PROMPT_EXAMPLES_PATH="${2:-}"
      shift 2
      ;;
    --vllm_prompt_disable_default_examples)
      VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES="1"
      shift 1
      ;;
    --filter_backend)
      FILTER_BACKEND="${2:-}"
      shift 2
      ;;
    --filter_model)
      FILTER_MODEL="${2:-}"
      shift 2
      ;;
    --filter_device)
      FILTER_DEVICE="${2:-}"
      FILTER_DEVICE_USER_SET="1"
      shift 2
      ;;
    --filter_batch_size)
      FILTER_BATCH_SIZE="${2:-}"
      shift 2
      ;;
    --filter_objects_col)
      FILTER_OBJECTS_COL="${2:-}"
      shift 2
      ;;
    --filter_prompt_examples_path)
      FILTER_PROMPT_EXAMPLES_PATH="${2:-}"
      shift 2
      ;;
    --filter_prompt_disable_default_examples)
      FILTER_PROMPT_DISABLE_DEFAULT_EXAMPLES="1"
      shift 1
      ;;
    *)
      echo "[pipeline] ERROR: unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CITY}" ]]; then
  echo "[pipeline] ERROR: --city is empty" >&2
  exit 2
fi

# If user provided --input_csv but didn't explicitly set --city, derive a sane default
# city label from the CSV filename (so outputs don't go to WashingtonDC by accident).
if [[ -n "${INPUT_CSV}" && "${CITY_SET}" != "1" && "${CITY}" == "WashingtonDC" ]]; then
  base="$(basename "${INPUT_CSV}")"
  CITY="${base%.*}"
fi

# City-derived paths (allow env overrides if user set them explicitly)
IMAGES_DIR="${IMAGES_DIR:-/home/dsi/oferidan/data/gsv_cities/Images/${CITY}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/debug_yesno_low_sam_conf/${CITY}}"

if [[ -z "${IMAGES_DIR}" ]]; then
  echo "[pipeline] ERROR: --images_dir is empty" >&2
  exit 2
fi

# If using --input_csv, resolve an images_root for relative image paths.
if [[ -n "${INPUT_CSV}" ]]; then
  # Resolve to absolute for stable dirname behavior.
  INPUT_CSV_ABS="$(python3 -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "${INPUT_CSV}")"
  if [[ -z "${IMAGES_ROOT}" ]]; then
    IMAGES_ROOT="$(dirname "${INPUT_CSV_ABS}")"
  fi
fi

OBJECTS_CSV="${OBJECTS_CSV:-${OUT_DIR}/${CITY}_objects.csv}"
TIMINGS_CSV="${TIMINGS_CSV:-${OUT_DIR}/out_timings.csv}"
SAM3_PROGRESS_CSV="${SAM3_PROGRESS_CSV:-${OUT_DIR}/sam3_realtime_progress.csv}"
VLLM_OUT_CSV="${VLLM_OUT_CSV:-${OUT_DIR}/sam3_realtime_progress_vllm_checked.csv}"
FILTERED_CSV="${FILTERED_CSV:-${OUT_DIR}/sam3_realtime_progress_vllm_checked_llm_filter.csv}"
FILTERED_SUMMARY_CSV="${FILTERED_SUMMARY_CSV:-${OUT_DIR}/sam3_realtime_progress_vllm_checked_llm_filter_summary.csv}"

# Logs
COMMANDS_LOG="${COMMANDS_LOG:-${OUT_DIR}/pipeline_commands.log}"
STEP1_LOG="${STEP1_LOG:-${OUT_DIR}/step1_visual_checker.log}"
STEP2_LOG="${STEP2_LOG:-${OUT_DIR}/step2_sam3_checker.log}"
STEP3_LOG="${STEP3_LOG:-${OUT_DIR}/step3_vllm_checker.log}"
STEP4_LOG="${STEP4_LOG:-${OUT_DIR}/step4_detect_desc_filter.log}"

mkdir -p "${OUT_DIR}"

ts() { date -Iseconds; }

csv_data_row_count() {
  # Usage: csv_data_row_count /path/to/file.csv
  # Prints number of non-header rows (best-effort, counts lines minus header).
  # NOTE: This is intentionally simple/fast; it assumes no embedded newlines in CSV cells.
  local p="${1:-}"
  if [[ -z "${p}" || ! -f "${p}" ]]; then
    echo 0
    return 0
  fi
  # wc -l is fast and good enough for our pipeline CSVs.
  local lines
  lines="$(wc -l < "${p}" 2>/dev/null || echo 0)"
  lines="${lines// /}"
  if [[ -z "${lines}" || "${lines}" -le 0 ]]; then
    echo 0
    return 0
  fi
  # subtract header line
  if [[ "${lines}" -ge 1 ]]; then
    echo "$((lines - 1))"
  else
    echo 0
  fi
}

should_skip_step_by_length() {
  # Usage: should_skip_step_by_length "label" /path/to/output.csv expected_rows
  # Returns 0 (true) if RESUME=1 and output exists and has >= expected_rows data rows.
  local label="$1"
  local out_csv="$2"
  local expected="${3:-0}"

  if [[ "${RESUME}" != "1" ]]; then
    return 1
  fi
  if [[ -z "${out_csv}" || ! -f "${out_csv}" ]]; then
    return 1
  fi
  if [[ "${expected}" -le 0 ]]; then
    return 1
  fi

  local out_rows
  out_rows="$(csv_data_row_count "${out_csv}")"
  if [[ "${out_rows}" -ge "${expected}" ]]; then
    echo "[pipeline] RESUME: skipping ${label} (output already complete: ${out_rows}/${expected} rows): ${out_csv}"
    log_note "resume_skip_${label}" \
      "expected_rows=${expected}" \
      "output_rows=${out_rows}" \
      "output_csv=${out_csv}"
    return 0
  fi
  return 1
}

split_csv_first_two() {
  # Usage: split_csv_first_two "a,b,c"  -> prints:
  #   first line: a
  #   second line: b   (may be empty if missing)
  local s="${1:-}"
  s="${s// /}"
  local first=""
  local second=""
  if [[ -n "${s}" ]]; then
    IFS=',' read -r first second _rest <<<"${s}"
  fi
  printf '%s\n%s\n' "${first}" "${second}"
}

count_csv_items() {
  # Usage: count_csv_items "a,b,c" -> prints 3
  local s="${1:-}"
  s="${s// /}"
  if [[ -z "${s}" ]]; then
    echo 0
    return 0
  fi
  local -a items
  IFS=',' read -r -a items <<<"${s}"
  echo "${#items[@]}"
}

detect_visible_gpu_count() {
  # Returns "how many GPUs are visible to this job", best-effort.
  # Priority:
  #  - If CUDA_VISIBLE_DEVICES is set: count its entries
  #  - Else: use nvidia-smi -L line count
  #  - Else: 0
  if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    count_csv_items "${CUDA_VISIBLE_DEVICES}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
    n="${n// /}"
    if [[ -n "${n}" ]]; then
      echo "${n}"
      return 0
    fi
  fi
  echo 0
}

start_step_bg() {
  # Usage:
  #   start_step_bg PID_VAR "label" "cuda_visible" "log_path" "resume(0/1)" "detach(0/1)" cmd...
  #
  # IMPORTANT: Do NOT call this via command substitution $(...) because that would run
  # the function in a subshell and the background PID would not be a child of the
  # caller shell (breaking `wait`).
  #
  # Sets PID_VAR in the caller shell.
  local pid_var="$1"
  local label="$2"
  local cuda_visible="$3"
  local log_path="$4"
  local resume="$5"
  local detach="$6"
  shift 6

  local -a cmd=( "$@" )
  local -a full_cmd=( "${cmd[@]}" )
  if [[ -n "${cuda_visible}" ]]; then
    full_cmd=( env CUDA_VISIBLE_DEVICES="${cuda_visible}" "${cmd[@]}" )
  fi

  log_cmd "${label}" "${full_cmd[@]}"
  append_log_banner_if_resuming "${log_path}" "${label}" "${full_cmd[@]}"

  if [[ "${detach}" == "1" ]]; then
    if [[ "${resume}" == "1" ]]; then
      nohup "${full_cmd[@]}" >> "${log_path}" 2>&1 &
    else
      nohup "${full_cmd[@]}" > "${log_path}" 2>&1 &
    fi
  else
    if [[ "${resume}" == "1" ]]; then
      "${full_cmd[@]}" >> "${log_path}" 2>&1 &
    else
      "${full_cmd[@]}" > "${log_path}" 2>&1 &
    fi
  fi

  local pid="$!"
  printf -v "${pid_var}" '%s' "${pid}"
}

prepare_log_path() {
  # Some Windows/WSL mounts can intermittently fail creating certain filenames.
  # Ensure each log path is writable; if not, fall back to /tmp.
  local label="$1"
  local path="$2"
  local fallback_dir="/tmp/text2vpr_pipeline_logs/${CITY}"
  mkdir -p "${fallback_dir}" 2>/dev/null || true

  # IMPORTANT: do not truncate logs here. This function is also used in --resume mode
  # where the user expects logs to be appended to, not overwritten.
  if ( : >> "${path}" ) 2>/dev/null; then
    echo "${path}"
    return 0
  fi

  local base
  base="$(basename "${path}")"
  local fallback_path="${fallback_dir}/${base}"
  if ( : >> "${fallback_path}" ) 2>/dev/null; then
    echo "[pipeline] WARN: cannot write ${label} log at: ${path}" >&2
    echo "[pipeline] WARN: using fallback log at: ${fallback_path}" >&2
    echo "${fallback_path}"
    return 0
  fi

  echo "[pipeline] ERROR: cannot write ${label} log at: ${path} (and fallback failed: ${fallback_path})" >&2
  echo "[pipeline] ERROR: Check permissions/mount health for ${OUT_DIR} (WSL drvfs can be flaky); try moving outputs to the Linux filesystem." >&2
  return 1
}
log_cmd() {
  # Usage: log_cmd "label" cmd arg1 arg2...
  local label="$1"
  shift
  {
    echo "=== [$(ts)] ${label} ==="
    printf '%q ' "$@"
    echo
    echo
  } >> "${COMMANDS_LOG}"
}

log_note() {
  # Usage: log_note "label" "line1" "line2" ...
  local label="$1"
  shift
  {
    echo "=== [$(ts)] ${label} ==="
    for line in "$@"; do
      echo "${line}"
    done
    echo
  } >> "${COMMANDS_LOG}"
}

append_log_banner_if_resuming() {
  # Usage: append_log_banner_if_resuming /path/to/log "label" cmd...
  local log_path="$1"
  local label="$2"
  shift 2
  if [[ "${RESUME}" != "1" ]]; then
    return 0
  fi
  {
    echo "=== [$(ts)] RESUME ${label} ==="
    printf '%q ' "$@"
    echo
    echo
  } >> "${log_path}"
}

log_device_probe() {
  # Best-effort probe of whether this run *can* use GPU (CUDA) vs CPU.
  # Writes to pipeline_commands.log and never fails the pipeline.
  local label="${1:-device_probe}"
  local py="${2:-python3}"

  local py_path
  py_path="$(command -v "${py}" 2>/dev/null || true)"

  local py_ver
  py_ver="$("${py}" --version 2>&1 || true)"

  local cuda_visible="${CUDA_VISIBLE_DEVICES-<unset>}"

  local nvsmi_line="<nvidia-smi not found>"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvsmi_line="$(nvidia-smi -L 2>/dev/null | head -n 1 || true)"
    if [[ -z "${nvsmi_line}" ]]; then
      nvsmi_line="<nvidia-smi present but no GPUs reported>"
    fi
  fi

  local torch_line
  torch_line="$("${py}" - <<'PY' 2>/dev/null || true
import json
try:
    import torch  # type: ignore
    info = {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_is_available": bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False,
        "cuda_device_count": int(torch.cuda.device_count()) if hasattr(torch, "cuda") else 0,
        "device": "GPU" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "CPU",
    }
    if info["cuda_is_available"] and info["cuda_device_count"] > 0:
        try:
            info["cuda_device_0_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    print("torch: " + json.dumps(info, sort_keys=True))
except Exception as e:
    print(f"torch: <not importable> ({type(e).__name__}: {e})")
PY
)"
  if [[ -z "${torch_line}" ]]; then
    torch_line="torch: <probe produced no output>"
  fi

  log_note "${label}" \
    "python: ${py} (${py_path:-<not found>})" \
    "python_version: ${py_ver}" \
    "CUDA_VISIBLE_DEVICES: ${cuda_visible}" \
    "nvidia_smi: ${nvsmi_line}" \
    "${torch_line}"
}

echo "[pipeline] repo: ${REPO_ROOT}"
echo "[pipeline] city: ${CITY}"
echo "[pipeline] images_dir: ${IMAGES_DIR}"
if [[ -n "${INPUT_CSV}" ]]; then
  echo "[pipeline] input_csv: ${INPUT_CSV}"
fi
if [[ -n "${IMAGES_ROOT}" ]]; then
  echo "[pipeline] images_root: ${IMAGES_ROOT}"
fi
echo "[pipeline] out:  ${OUT_DIR}"
echo "[pipeline] watch_poll_sec=${WATCH_POLL_SEC}, watch_idle_minutes=${WATCH_IDLE_MIN}"
echo "[pipeline] input_csv_wait_seconds=${INPUT_CSV_WAIT_SECONDS}"
echo "[pipeline] sam3_box_threshold=${SAM3_BOX_THRESHOLD}"
if [[ -n "${SAM3_CHECKPOINT_PATH}" ]]; then
  echo "[pipeline] sam3_checkpoint_path=${SAM3_CHECKPOINT_PATH}"
fi
echo "[pipeline] resume=${RESUME}"
echo "[pipeline] vllm_prompt_style=${VLLM_PROMPT_STYLE}"
if [[ -n "${VLLM_PROMPT_EXAMPLES_PATH}" ]]; then
  echo "[pipeline] vllm_prompt_examples_path=${VLLM_PROMPT_EXAMPLES_PATH}"
fi
if [[ "${VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES}" == "1" ]]; then
  echo "[pipeline] vllm_prompt_disable_default_examples=1"
fi
echo "[pipeline] step4_filter=on (backend=${FILTER_BACKEND}, batch_size=${FILTER_BATCH_SIZE}, objects_col=${FILTER_OBJECTS_COL})"
echo "[pipeline] filter_model=${FILTER_MODEL}"
echo "[pipeline] filter_device=${FILTER_DEVICE}"
if [[ -n "${FILTER_PROMPT_EXAMPLES_PATH}" ]]; then
  echo "[pipeline] filter_prompt_examples_path=${FILTER_PROMPT_EXAMPLES_PATH}"
fi
if [[ "${FILTER_PROMPT_DISABLE_DEFAULT_EXAMPLES}" == "1" ]]; then
  echo "[pipeline] filter_prompt_disable_default_examples=1"
fi
echo ""

COMMANDS_LOG="$(prepare_log_path "commands" "${COMMANDS_LOG}")"
STEP1_LOG="$(prepare_log_path "step1" "${STEP1_LOG}")"
STEP2_LOG="$(prepare_log_path "step2" "${STEP2_LOG}")"
STEP3_LOG="$(prepare_log_path "step3" "${STEP3_LOG}")"
STEP4_LOG="$(prepare_log_path "step4" "${STEP4_LOG}")"

echo "[pipeline] Writing command log to: ${COMMANDS_LOG}"
log_cmd "pipeline_boot" echo "logs:" "${STEP1_LOG}" "${STEP2_LOG}" "${STEP3_LOG}"
log_device_probe "pipeline_device_probe" "python3"

EXPECTED_ROWS="0"
if [[ -n "${INPUT_CSV-}" ]]; then
  EXPECTED_ROWS="$(csv_data_row_count "${INPUT_CSV_ABS}")"
  echo "[pipeline] resume_length_check: input_rows=${EXPECTED_ROWS} (from ${INPUT_CSV_ABS})"
fi

VISIBLE_GPU_COUNT="$(detect_visible_gpu_count)"
USE_GPU_SPLIT="0"
GPU0_VISIBLE=""
GPU1_VISIBLE=""

if [[ "${VISIBLE_GPU_COUNT}" -ge 2 ]]; then
  USE_GPU_SPLIT="1"
  if [[ -n "${CUDA_VISIBLE_DEVICES-}" ]]; then
    read -r GPU0_VISIBLE GPU1_VISIBLE < <(split_csv_first_two "${CUDA_VISIBLE_DEVICES}")
  else
    GPU0_VISIBLE="0"
    GPU1_VISIBLE="1"
  fi
fi

if [[ "${USE_GPU_SPLIT}" == "1" ]]; then
  echo "[pipeline] GPU routing (2+ GPUs):"
  echo "  - step1 -> CUDA_VISIBLE_DEVICES=${GPU0_VISIBLE}"
  echo "  - step2 -> CUDA_VISIBLE_DEVICES=${GPU1_VISIBLE}"
  echo "  - step3 -> CUDA_VISIBLE_DEVICES=${GPU0_VISIBLE} (starts after step1 finishes)"
  echo "  - step4 -> CUDA_VISIBLE_DEVICES=${GPU1_VISIBLE} (starts after step2 finishes)"
  log_note "gpu_routing" \
    "visible_gpu_count=${VISIBLE_GPU_COUNT}" \
    "gpu0_visible=${GPU0_VISIBLE}" \
    "gpu1_visible=${GPU1_VISIBLE}" \
    "note: per-step CUDA_VISIBLE_DEVICES overrides are used only when 2+ GPUs are visible"

  # Step 4 defaults to CPU; if the user didn't explicitly set a device, switch it to GPU
  # so step 4 actually runs on the routed device (cuda:0 within that process).
  if [[ "${FILTER_BACKEND}" == "hf" && "${FILTER_DEVICE_USER_SET}" != "1" && "${FILTER_DEVICE}" == "cpu" ]]; then
    FILTER_DEVICE="cuda"
    echo "[pipeline] Step 4: auto-setting --filter_device to '${FILTER_DEVICE}' (2-GPU routing enabled; user did not override filter device)"
  fi
else
  echo "[pipeline] GPU routing: disabled (visible_gpu_count=${VISIBLE_GPU_COUNT}); steps will use default CUDA_VISIBLE_DEVICES"
fi

echo "[pipeline] Step 1/4: visual_checker (writes objects CSV incrementally)"
if [[ -n "${INPUT_CSV}" ]]; then
  CMD_STEP1=(
    python3 "${REPO_ROOT}/visual_checker/main.py"
    --input_csv "${INPUT_CSV_ABS}"
    --output_dir "${OUT_DIR}"
    --output_csv "${OBJECTS_CSV}"
    --use_merged_prompt
  )
else
  CMD_STEP1=(
    python3 "${REPO_ROOT}/visual_checker/main.py"
    --images_dir "${IMAGES_DIR}"
    --captions_csv "${CAPTIONS_CSV}"
    --output_dir "${OUT_DIR}"
    --output_csv "${OBJECTS_CSV}"
    --per_image_timing_csv "${TIMINGS_CSV}"
    --use_merged_prompt
  )
fi
if [[ "${RESUME}" == "1" ]]; then
  CMD_STEP1+=(--resume)
  # Defense-in-depth: keep a one-time backup of the existing objects CSV before resuming.
  # This protects against accidental truncation/corruption during interrupts.
  if [[ -s "${OBJECTS_CSV}" && ! -f "${OBJECTS_CSV}.bak" ]]; then
    cp -p "${OBJECTS_CSV}" "${OBJECTS_CSV}.bak" || true
  fi
fi
CUDA_STEP1=""
CUDA_STEP2=""
CUDA_STEP3=""
CUDA_STEP4=""
if [[ "${USE_GPU_SPLIT}" == "1" ]]; then
  CUDA_STEP1="${GPU0_VISIBLE}"
  CUDA_STEP2="${GPU1_VISIBLE}"
  CUDA_STEP3="${GPU0_VISIBLE}"
  CUDA_STEP4="${GPU1_VISIBLE}"
fi

if [[ "${USE_GPU_SPLIT}" == "1" && "${DETACH}" == "1" ]]; then
  echo "[pipeline] WARN: DETACH=1 is not compatible with gated step starts (step3 after step1, step4 after step2)."
  echo "[pipeline] WARN: Running as DETACH=0 for this run."
  DETACH="0"
fi

PID_STEP3=""
PID_STEP4=""

STEP1_SKIPPED="0"
if should_skip_step_by_length "step1_visual_checker" "${OBJECTS_CSV}" "${EXPECTED_ROWS}"; then
  STEP1_SKIPPED="1"
  PID_STEP1=""
else
  start_step_bg PID_STEP1 "step1_visual_checker" "${CUDA_STEP1}" "${STEP1_LOG}" "${RESUME}" "${DETACH}" "${CMD_STEP1[@]}"
fi

echo "[pipeline] Step 2/4: SAM3 checker (watches objects CSV -> sam3_realtime_progress.csv)"
CMD_STEP2=(
  python3 "${REPO_ROOT}/visual_checker/sam3_checker.py"
  --input_csv "${OBJECTS_CSV}"
  --realtime_progress_csv "${SAM3_PROGRESS_CSV}"
  --box_threshold "${SAM3_BOX_THRESHOLD}"
  --resume
  --watch
  --watch_poll_sec "${WATCH_POLL_SEC}"
  --watch_idle_minutes "${WATCH_IDLE_MIN}"
  --input_csv_wait_seconds "${INPUT_CSV_WAIT_SECONDS}"
)
if [[ -n "${SAM3_CHECKPOINT_PATH}" ]]; then
  CMD_STEP2+=(--checkpoint_path "${SAM3_CHECKPOINT_PATH}")
fi
if [[ -n "${IMAGES_ROOT}" ]]; then
  CMD_STEP2+=(--images_root "${IMAGES_ROOT}")
fi
STEP2_SKIPPED="0"
if should_skip_step_by_length "step2_sam3_checker" "${SAM3_PROGRESS_CSV}" "${EXPECTED_ROWS}"; then
  STEP2_SKIPPED="1"
  PID_STEP2=""
else
  start_step_bg PID_STEP2 "step2_sam3_checker" "${CUDA_STEP2}" "${STEP2_LOG}" "${RESUME}" "${DETACH}" "${CMD_STEP2[@]}"
fi

echo "[pipeline] Step 3/4: vLLM checker (watches sam3 progress CSV -> vllm_checked.csv)"
CMD_STEP3=(
  python3 "${REPO_ROOT}/vllm_checker/main.py"
  --input_csv "${SAM3_PROGRESS_CSV}"
  --output_csv "${VLLM_OUT_CSV}"
  --resume
  --llm_batch_size "${LLM_BATCH_SIZE}"
  --prompt_style "${VLLM_PROMPT_STYLE}"
  --watch
  --watch_poll_sec "${WATCH_POLL_SEC}"
  --watch_idle_minutes "${WATCH_IDLE_MIN}"
  --input_csv_wait_seconds "${INPUT_CSV_WAIT_SECONDS}"
)
if [[ -n "${IMAGES_ROOT}" ]]; then
  CMD_STEP3+=(--images_root "${IMAGES_ROOT}")
fi
if [[ -n "${VLLM_PROMPT_EXAMPLES_PATH}" ]]; then
  CMD_STEP3+=(--prompt_examples_path "${VLLM_PROMPT_EXAMPLES_PATH}")
fi
if [[ "${VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES}" == "1" ]]; then
  CMD_STEP3+=(--prompt_disable_default_examples)
fi
echo "[pipeline] Step 4/4 (watch): merged detectability+description filter (watches vllm_checked.csv -> llm_filter.csv)"
CMD_STEP4=(
  python3 "${REPO_ROOT}/filter_step/llm_filter_objects_detectability_and_description.py"
  --input_csv "${VLLM_OUT_CSV}"
  --output_csv "${FILTERED_CSV}"
  --summary_csv "${FILTERED_SUMMARY_CSV}"
  --backend "${FILTER_BACKEND}"
  --batch_size "${FILTER_BATCH_SIZE}"
  --objects_col "${FILTER_OBJECTS_COL}"
  --description_col "description"
  --image_col "image_path"
  --resume
  --watch
  --watch_poll_sec "${WATCH_POLL_SEC}"
  --watch_idle_minutes "${WATCH_IDLE_MIN}"
  --input_csv_wait_seconds "${INPUT_CSV_WAIT_SECONDS}"
)
if [[ "${FILTER_BACKEND}" == "hf" ]]; then
  CMD_STEP4+=(--hf_model "${FILTER_MODEL}")
  CMD_STEP4+=(--hf_device "${FILTER_DEVICE}")
else
  CMD_STEP4+=(--openai_model "${FILTER_MODEL}")
fi
if [[ -n "${FILTER_PROMPT_EXAMPLES_PATH}" ]]; then
  CMD_STEP4+=(--prompt_examples_path "${FILTER_PROMPT_EXAMPLES_PATH}")
fi
if [[ "${FILTER_PROMPT_DISABLE_DEFAULT_EXAMPLES}" == "1" ]]; then
  CMD_STEP4+=(--prompt_disable_default_examples)
fi

if [[ "${USE_GPU_SPLIT}" == "1" ]]; then
  echo ""
  echo "[pipeline] PIDs: step1=${PID_STEP1:-<skipped>}, step2=${PID_STEP2:-<skipped>}"

  # Gate step 3 on step 1 (GPU0 contention avoidance)
  if [[ "${STEP1_SKIPPED}" != "1" ]]; then
    echo "[pipeline] Waiting for step 1 to finish (step 2 continues in parallel)..."
    set +e
    wait "${PID_STEP1}"
    EC1="$?"
    set -e

    if [[ "${EC1}" -ne 0 ]]; then
      echo "[pipeline] ERROR: step 1 exited with code ${EC1}. Stopping step 2..." >&2
      if [[ "${STEP2_SKIPPED}" != "1" ]]; then
        kill "${PID_STEP2}" 2>/dev/null || true
      fi
      exit "${EC1}"
    fi
  else
    echo "[pipeline] Step 1 skipped (already complete). Starting step 3 immediately..."
  fi

  STEP3_SKIPPED="0"
  if should_skip_step_by_length "step3_vllm_checker" "${VLLM_OUT_CSV}" "${EXPECTED_ROWS}"; then
    STEP3_SKIPPED="1"
    PID_STEP3=""
  else
    start_step_bg PID_STEP3 "step3_vllm_checker" "${CUDA_STEP3}" "${STEP3_LOG}" "${RESUME}" "${DETACH}" "${CMD_STEP3[@]}"
  fi

  # Gate step 4 on step 2 (GPU1 contention avoidance)
  if [[ "${STEP2_SKIPPED}" != "1" ]]; then
    echo "[pipeline] Waiting for step 2 to finish (step 3 continues in parallel)..."
    set +e
    wait "${PID_STEP2}"
    EC2="$?"
    set -e

    if [[ "${EC2}" -ne 0 ]]; then
      echo "[pipeline] ERROR: step 2 exited with code ${EC2}. Stopping step 3..." >&2
      if [[ -n "${PID_STEP3-}" ]]; then
        kill "${PID_STEP3}" 2>/dev/null || true
      fi
      exit "${EC2}"
    fi
  else
    echo "[pipeline] Step 2 skipped (already complete). Starting step 4 immediately..."
  fi

  STEP4_SKIPPED="0"
  if should_skip_step_by_length "step4_detect_desc_filter" "${FILTERED_CSV}" "${EXPECTED_ROWS}"; then
    STEP4_SKIPPED="1"
    PID_STEP4=""
  else
    start_step_bg PID_STEP4 "step4_detect_desc_filter" "${CUDA_STEP4}" "${STEP4_LOG}" "${RESUME}" "${DETACH}" "${CMD_STEP4[@]}"
  fi

  echo ""
  echo "[pipeline] PIDs: step1=${PID_STEP1}, step2=${PID_STEP2}, step3=${PID_STEP3}, step4=${PID_STEP4}"
  echo "[pipeline] Waiting for step 3+4 to go idle and exit..."
  echo "[pipeline] Logs:"
  echo "  - ${STEP1_LOG}"
  echo "  - ${STEP2_LOG}"
  echo "  - ${STEP3_LOG}"
  echo "  - ${STEP4_LOG}"

  # DETACH is forced to 0 above for split runs (gated step starts require this process).
  if [[ -n "${PID_STEP3-}" ]]; then
    wait "${PID_STEP3}" || true
  fi
  if [[ -n "${PID_STEP4-}" ]]; then
    wait "${PID_STEP4}" || true
  fi
else
  # Original behavior: start steps 3 and 4 immediately, then wait on step 1 while
  # watchers run; once step 1 is done, watchers exit on idle.
  STEP3_SKIPPED="0"
  STEP4_SKIPPED="0"
  if should_skip_step_by_length "step3_vllm_checker" "${VLLM_OUT_CSV}" "${EXPECTED_ROWS}"; then
    STEP3_SKIPPED="1"
    PID_STEP3=""
  else
    start_step_bg PID_STEP3 "step3_vllm_checker" "${CUDA_STEP3}" "${STEP3_LOG}" "${RESUME}" "${DETACH}" "${CMD_STEP3[@]}"
  fi
  if should_skip_step_by_length "step4_detect_desc_filter" "${FILTERED_CSV}" "${EXPECTED_ROWS}"; then
    STEP4_SKIPPED="1"
    PID_STEP4=""
  else
    start_step_bg PID_STEP4 "step4_detect_desc_filter" "${CUDA_STEP4}" "${STEP4_LOG}" "${RESUME}" "${DETACH}" "${CMD_STEP4[@]}"
  fi

  echo ""
  echo "[pipeline] PIDs: step1=${PID_STEP1}, step2=${PID_STEP2}, step3=${PID_STEP3}, step4=${PID_STEP4}"
  echo "[pipeline] Waiting for step 1 to finish (steps 2+3+4 will auto-exit on idle)..."
  echo "[pipeline] Logs:"
  echo "  - ${STEP1_LOG}"
  echo "  - ${STEP2_LOG}"
  echo "  - ${STEP3_LOG}"
  echo "  - ${STEP4_LOG}"

  if [[ "${DETACH}" == "1" ]]; then
    echo ""
    echo "[pipeline] DETACH=1: not waiting. To monitor:"
    echo "  tail -f \"${STEP1_LOG}\""
    echo "  tail -f \"${STEP2_LOG}\""
    echo "  tail -f \"${STEP3_LOG}\""
    echo "  tail -f \"${STEP4_LOG}\""
    echo ""
    exit 0
  fi

  if [[ "${STEP1_SKIPPED}" != "1" ]]; then
    set +e
    wait "${PID_STEP1}"
    EC1="$?"
    set -e

    if [[ "${EC1}" -ne 0 ]]; then
      echo "[pipeline] ERROR: step 1 exited with code ${EC1}. Stopping watchers..." >&2
      if [[ "${STEP2_SKIPPED}" != "1" ]]; then
        kill "${PID_STEP2}" 2>/dev/null || true
      fi
      if [[ -n "${PID_STEP3-}" ]]; then
        kill "${PID_STEP3}" 2>/dev/null || true
      fi
      if [[ -n "${PID_STEP4-}" ]]; then
        kill "${PID_STEP4}" 2>/dev/null || true
      fi
      exit "${EC1}"
    fi
  fi

  echo "[pipeline] Step 1 finished. Waiting for step 2+3+4 to go idle and exit..."
  if [[ "${STEP2_SKIPPED}" != "1" ]]; then
    wait "${PID_STEP2}" || true
  fi
  if [[ -n "${PID_STEP3-}" ]]; then
    wait "${PID_STEP3}" || true
  fi
  if [[ -n "${PID_STEP4-}" ]]; then
    wait "${PID_STEP4}" || true
  fi
fi

echo ""
echo "[pipeline] Done."
echo "[pipeline] outputs:"
echo "  - ${OBJECTS_CSV}"
echo "  - ${SAM3_PROGRESS_CSV}"
echo "  - ${VLLM_OUT_CSV}"
echo "  - ${FILTERED_CSV}"
echo "  - ${FILTERED_SUMMARY_CSV}"
echo "[pipeline] command log:"
echo "  - ${COMMANDS_LOG}"


