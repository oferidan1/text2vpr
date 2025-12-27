#!/usr/bin/env bash
set -euo pipefail

# Pipeline:
# 1) visual_checker/main.py produces: WashingtonDC_objects.csv (written incrementally)
# 2) visual_checker/sam3_checker.py watches that CSV and produces: sam3_realtime_progress.csv (written incrementally)
# 3) vllm_checker/main.py watches sam3_realtime_progress.csv and produces: sam3_realtime_progress_vllm_checked.csv
#
# This script runs step 1, then runs steps 2 and 3 in --watch mode so they keep
# re-reading the growing CSVs until they go idle.

usage() {
  cat <<'EOF'
Usage:
  run_full_quality_pipeline.sh [options]

Options (general):
  --city <CityName>   (default: WashingtonDC)
  --input_csv_wait_seconds <seconds>  (default: 900)

Options (Step 3: vllm_checker prompt):
  --vllm_prompt_style <strict_yn|describe_then_yesno>
  --vllm_prompt_examples_path <path/to/examples.txt>
  --vllm_prompt_disable_default_examples

You can also set equivalent env vars:
  VLLM_PROMPT_STYLE
  VLLM_PROMPT_EXAMPLES_PATH
  VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES=1
EOF
}

REPO_ROOT="/mnt/d/dan/git_projects/text2vpr"

# Defaults (can be overridden via CLI args below or env vars)
CITY="${CITY:-WashingtonDC}"
CAPTIONS_CSV="${CAPTIONS_CSV:-/mnt/d/data/gsv_cities/gsv_cities_predictions_nan_fix.csv}"

# Watch settings
WATCH_POLL_SEC="${WATCH_POLL_SEC:-60}"
WATCH_IDLE_MIN="${WATCH_IDLE_MIN:-20}"
INPUT_CSV_WAIT_SECONDS="${INPUT_CSV_WAIT_SECONDS:-900}"

LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
DETACH="${DETACH:-0}" # if 1, start all steps and exit immediately (like running in 3 terminals)

# Step 3 prompt settings (defaults preserve legacy behavior)
VLLM_PROMPT_STYLE="${VLLM_PROMPT_STYLE:-strict_yn}"
VLLM_PROMPT_EXAMPLES_PATH="${VLLM_PROMPT_EXAMPLES_PATH:-}"
VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES="${VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --city)
      CITY="${2:-}"
      shift 2
      ;;
    --input_csv_wait_seconds)
      INPUT_CSV_WAIT_SECONDS="${2:-}"
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

# City-derived paths (allow env overrides if user set them explicitly)
IMAGES_DIR="${IMAGES_DIR:-/mnt/d/data/gsv_cities/Images/${CITY}}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/debug_yesno_low_sam_conf/${CITY}}"

OBJECTS_CSV="${OBJECTS_CSV:-${OUT_DIR}/${CITY}_objects.csv}"
TIMINGS_CSV="${TIMINGS_CSV:-${OUT_DIR}/out_timings.csv}"
SAM3_PROGRESS_CSV="${SAM3_PROGRESS_CSV:-${OUT_DIR}/sam3_realtime_progress.csv}"
VLLM_OUT_CSV="${VLLM_OUT_CSV:-${OUT_DIR}/sam3_realtime_progress_vllm_checked.csv}"

# Logs
COMMANDS_LOG="${COMMANDS_LOG:-${OUT_DIR}/pipeline_commands.log}"
STEP1_LOG="${STEP1_LOG:-${OUT_DIR}/step1_visual_checker.log}"
STEP2_LOG="${STEP2_LOG:-${OUT_DIR}/step2_sam3_checker.log}"
STEP3_LOG="${STEP3_LOG:-${OUT_DIR}/step3_vllm_checker.log}"

mkdir -p "${OUT_DIR}"

ts() { date -Iseconds; }

prepare_log_path() {
  # Some Windows/WSL mounts can intermittently fail creating certain filenames.
  # Ensure each log path is writable; if not, fall back to /tmp.
  local label="$1"
  local path="$2"
  local fallback_dir="/tmp/text2vpr_pipeline_logs/${CITY}"
  mkdir -p "${fallback_dir}" 2>/dev/null || true

  if ( : > "${path}" ) 2>/dev/null; then
    echo "${path}"
    return 0
  fi

  local base
  base="$(basename "${path}")"
  local fallback_path="${fallback_dir}/${base}"
  if ( : > "${fallback_path}" ) 2>/dev/null; then
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

echo "[pipeline] repo: ${REPO_ROOT}"
echo "[pipeline] city: ${CITY}"
echo "[pipeline] out:  ${OUT_DIR}"
echo "[pipeline] watch_poll_sec=${WATCH_POLL_SEC}, watch_idle_minutes=${WATCH_IDLE_MIN}"
echo "[pipeline] input_csv_wait_seconds=${INPUT_CSV_WAIT_SECONDS}"
echo "[pipeline] vllm_prompt_style=${VLLM_PROMPT_STYLE}"
if [[ -n "${VLLM_PROMPT_EXAMPLES_PATH}" ]]; then
  echo "[pipeline] vllm_prompt_examples_path=${VLLM_PROMPT_EXAMPLES_PATH}"
fi
if [[ "${VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES}" == "1" ]]; then
  echo "[pipeline] vllm_prompt_disable_default_examples=1"
fi
echo ""

COMMANDS_LOG="$(prepare_log_path "commands" "${COMMANDS_LOG}")"
STEP1_LOG="$(prepare_log_path "step1" "${STEP1_LOG}")"
STEP2_LOG="$(prepare_log_path "step2" "${STEP2_LOG}")"
STEP3_LOG="$(prepare_log_path "step3" "${STEP3_LOG}")"

echo "[pipeline] Writing command log to: ${COMMANDS_LOG}"
log_cmd "pipeline_boot" echo "logs:" "${STEP1_LOG}" "${STEP2_LOG}" "${STEP3_LOG}"

echo "[pipeline] Step 1/3: visual_checker (writes objects CSV incrementally)"
CMD_STEP1=(
  python3 "${REPO_ROOT}/visual_checker/main.py"
  --images_dir "${IMAGES_DIR}"
  --captions_csv "${CAPTIONS_CSV}"
  --output_dir "${OUT_DIR}"
  --output_csv "${OBJECTS_CSV}"
  --per_image_timing_csv "${TIMINGS_CSV}"
  --use_merged_prompt
)
log_cmd "step1_visual_checker" "${CMD_STEP1[@]}"
if [[ "${DETACH}" == "1" ]]; then
  nohup "${CMD_STEP1[@]}" > "${STEP1_LOG}" 2>&1 &
else
  "${CMD_STEP1[@]}" > "${STEP1_LOG}" 2>&1 &
fi
PID_STEP1="$!"

echo "[pipeline] Step 2/3: SAM3 checker (watches objects CSV -> sam3_realtime_progress.csv)"
CMD_STEP2=(
  python3 "${REPO_ROOT}/visual_checker/sam3_checker.py"
  --input_csv "${OBJECTS_CSV}"
  --realtime_progress_csv "${SAM3_PROGRESS_CSV}"
  --resume
  --watch
  --watch_poll_sec "${WATCH_POLL_SEC}"
  --watch_idle_minutes "${WATCH_IDLE_MIN}"
  --input_csv_wait_seconds "${INPUT_CSV_WAIT_SECONDS}"
)
log_cmd "step2_sam3_checker" "${CMD_STEP2[@]}"
if [[ "${DETACH}" == "1" ]]; then
  nohup "${CMD_STEP2[@]}" > "${STEP2_LOG}" 2>&1 &
else
  "${CMD_STEP2[@]}" > "${STEP2_LOG}" 2>&1 &
fi
PID_STEP2="$!"

echo "[pipeline] Step 3/3: vLLM checker (watches sam3 progress CSV -> vllm_checked.csv)"
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
if [[ -n "${VLLM_PROMPT_EXAMPLES_PATH}" ]]; then
  CMD_STEP3+=(--prompt_examples_path "${VLLM_PROMPT_EXAMPLES_PATH}")
fi
if [[ "${VLLM_PROMPT_DISABLE_DEFAULT_EXAMPLES}" == "1" ]]; then
  CMD_STEP3+=(--prompt_disable_default_examples)
fi
log_cmd "step3_vllm_checker" "${CMD_STEP3[@]}"
if [[ "${DETACH}" == "1" ]]; then
  nohup "${CMD_STEP3[@]}" > "${STEP3_LOG}" 2>&1 &
else
  "${CMD_STEP3[@]}" > "${STEP3_LOG}" 2>&1 &
fi
PID_STEP3="$!"

echo ""
echo "[pipeline] PIDs: step1=${PID_STEP1}, step2=${PID_STEP2}, step3=${PID_STEP3}"
echo "[pipeline] Waiting for step 1 to finish (steps 2+3 will auto-exit on idle)..."
echo "[pipeline] Logs:"
echo "  - ${STEP1_LOG}"
echo "  - ${STEP2_LOG}"
echo "  - ${STEP3_LOG}"

if [[ "${DETACH}" == "1" ]]; then
  echo ""
  echo "[pipeline] DETACH=1: not waiting. To monitor:"
  echo "  tail -f \"${STEP1_LOG}\""
  echo "  tail -f \"${STEP2_LOG}\""
  echo "  tail -f \"${STEP3_LOG}\""
  echo ""
  exit 0
fi

set +e
wait "${PID_STEP1}"
EC1="$?"
set -e

if [[ "${EC1}" -ne 0 ]]; then
  echo "[pipeline] ERROR: step 1 exited with code ${EC1}. Stopping watchers..." >&2
  kill "${PID_STEP2}" "${PID_STEP3}" 2>/dev/null || true
  exit "${EC1}"
fi

echo "[pipeline] Step 1 finished. Waiting for step 2+3 to go idle and exit..."
wait "${PID_STEP2}" || true
wait "${PID_STEP3}" || true

echo ""
echo "[pipeline] Done."
echo "[pipeline] outputs:"
echo "  - ${OBJECTS_CSV}"
echo "  - ${SAM3_PROGRESS_CSV}"
echo "  - ${VLLM_OUT_CSV}"
echo "[pipeline] command log:"
echo "  - ${COMMANDS_LOG}"


