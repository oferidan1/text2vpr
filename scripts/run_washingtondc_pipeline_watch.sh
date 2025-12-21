#!/usr/bin/env bash
set -euo pipefail

# Pipeline:
# 1) visual_checker/main.py produces: WashingtonDC_objects.csv (written incrementally)
# 2) visual_checker/sam3_checker.py watches that CSV and produces: sam3_realtime_progress.csv (written incrementally)
# 3) vllm_checker/main.py watches sam3_realtime_progress.csv and produces: sam3_realtime_progress_vllm_checked.csv
#
# This script runs step 1, then runs steps 2 and 3 in --watch mode so they keep
# re-reading the growing CSVs until they go idle.

REPO_ROOT="/mnt/d/dan/git_projects/text2vpr"

CITY="WashingtonDC"
IMAGES_DIR="/mnt/d/data/gsv_cities/Images/WashingtonDC"
CAPTIONS_CSV="/mnt/d/data/gsv_cities/gsv_cities_predictions_nan_fix.csv"
OUT_DIR="${REPO_ROOT}/results/${CITY}"

OBJECTS_CSV="${OUT_DIR}/${CITY}_objects.csv"
TIMINGS_CSV="${OUT_DIR}/out_timings.csv"
SAM3_PROGRESS_CSV="${OUT_DIR}/sam3_realtime_progress.csv"
VLLM_OUT_CSV="${OUT_DIR}/sam3_realtime_progress_vllm_checked.csv"

# Logs
COMMANDS_LOG="${OUT_DIR}/pipeline_commands.log"
STEP1_LOG="${OUT_DIR}/step1_visual_checker.log"
STEP2_LOG="${OUT_DIR}/step2_sam3_checker.log"
STEP3_LOG="${OUT_DIR}/step3_vllm_checker.log"

# Watch settings
WATCH_POLL_SEC="${WATCH_POLL_SEC:-60}"
WATCH_IDLE_MIN="${WATCH_IDLE_MIN:-20}"

LLM_BATCH_SIZE="${LLM_BATCH_SIZE:-4}"
DETACH="${DETACH:-0}" # if 1, start all steps and exit immediately (like running in 3 terminals)

mkdir -p "${OUT_DIR}"

ts() { date -Iseconds; }
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
echo ""

echo "[pipeline] Writing command log to: ${COMMANDS_LOG}"
: > "${COMMANDS_LOG}"

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
  --watch
  --watch_poll_sec "${WATCH_POLL_SEC}"
  --watch_idle_minutes "${WATCH_IDLE_MIN}"
)
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


