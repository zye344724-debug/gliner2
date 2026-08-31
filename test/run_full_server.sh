#!/usr/bin/env bash
# Full 76-field server training with rare/confusable-field curriculum.
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${TEST_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VARIANT="${VARIANT:-full_server}"
DATA_DIR="${TEST_DIR}/data/${VARIANT}"
NER_OUT="${TEST_DIR}/outputs/ner/${VARIANT}"
FOCUS_OUT="${TEST_DIR}/outputs/structure/${VARIANT}_focus"
FINAL_OUT="${TEST_DIR}/outputs/structure/${VARIANT}"
LOG_DIR="${TEST_DIR}/logs/${VARIANT}"

NER_EPOCHS="${NER_EPOCHS:-4}"
FOCUS_STRUCTURE_EPOCHS="${FOCUS_STRUCTURE_EPOCHS:-8}"
FULL_CALIBRATION_EPOCHS="${FULL_CALIBRATION_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_LEN="${MAX_LEN:-384}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_STEPS="${EVAL_STEPS:-250}"
RARE_FIELD_TARGET="${RARE_FIELD_TARGET:-800}"
FOCUS_MAX_REPEATS="${FOCUS_MAX_REPEATS:-4}"
PRECISION="${PRECISION:-bf16}"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1

case "${PRECISION}" in
  bf16) PRECISION_ARGS=(--bf16) ;;
  fp16) PRECISION_ARGS=(--fp16) ;;
  fp32) PRECISION_ARGS=() ;;
  *) echo "PRECISION must be bf16, fp16, or fp32" >&2; exit 2 ;;
esac

run_step() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  "$@" 2>&1 | tee "${LOG_DIR}/${name}.log"
}

# Keep training attached to the terminal. Piping tqdm through tee turns each
# carriage-return refresh into a separate line in many shells/log viewers.
run_training_step() {
  local name="$1"
  shift
  echo
  echo "==> ${name}"
  echo "    Progress is shown as one updating line; results are saved under outputs/."
  "$@"
}

run_step environment "${PYTHON_BIN}" -c \
  "import sys, torch, transformers; print('Python:', sys.executable); print('torch:', torch.__version__); print('transformers:', transformers.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30, 1))"

run_step prepare_full_focus \
  "${PYTHON_BIN}" "${TEST_DIR}/prepare_data.py" \
  --schema-mode full --variant "${VARIANT}" --split-multi \
  --retain-train-multi-max-deals 5 \
  --focus-training --rare-field-target "${RARE_FIELD_TARGET}" \
  --focus-max-repeats "${FOCUS_MAX_REPEATS}"

run_step validate_data \
  "${PYTHON_BIN}" "${TEST_DIR}/validate_data.py" \
  --data-dirs "${DATA_DIR}" --out "${LOG_DIR}/validate_data.json"

COMMON_ARGS=(
  --schema-mode full --data-variant "${VARIANT}"
  --batch-size "${BATCH_SIZE}" --eval-batch-size "${EVAL_BATCH_SIZE}"
  --grad-accum "${GRAD_ACCUM}" --max-len "${MAX_LEN}"
  --eval-steps "${EVAL_STEPS}" --num-workers "${NUM_WORKERS}"
  --gradient-checkpointing --early-stopping-patience 4
  "${PRECISION_ARGS[@]}"
)

# Stage 1: all fields plus focused family views.  This gives every rare field
# repeated span supervision before record-level structure training.
run_training_step train_ner_full_focus \
  "${PYTHON_BIN}" "${TEST_DIR}/train_ner.py" \
  --epochs "${NER_EPOCHS}" --output-dir "${NER_OUT}" \
  --train-file "${DATA_DIR}/ner_train_balanced_clean.jsonl" \
  "${COMMON_ARGS[@]}"

# Stage 2: structure curriculum containing every full-schema row plus focused
# small-schema rows for rare/confusable families.
run_training_step train_structure_focus \
  "${PYTHON_BIN}" "${TEST_DIR}/train_structure.py" \
  --epochs "${FOCUS_STRUCTURE_EPOCHS}" --init-from "${NER_OUT}" \
  --output-dir "${FOCUS_OUT}" \
  --train-file "${DATA_DIR}/structure_train_balanced_clean.jsonl" \
  "${COMMON_ARGS[@]}"

# Stage 3: short, lower-LR calibration on primary full-schema rows only.  This
# removes the train/inference schema-mixture gap without forgetting rare fields.
run_training_step calibrate_structure_full_schema \
  "${PYTHON_BIN}" "${TEST_DIR}/train_structure.py" \
  --epochs "${FULL_CALIBRATION_EPOCHS}" --init-from "${FOCUS_OUT}" \
  --output-dir "${FINAL_OUT}" \
  --train-file "${DATA_DIR}/structure_train_clean.jsonl" \
  --encoder-lr 5e-6 --task-lr 1e-4 \
  "${COMMON_ARGS[@]}"

run_step evaluate_full_76_fields \
  "${PYTHON_BIN}" "${TEST_DIR}/evaluate_sentence_acc.py" \
  --schema-mode full --data-variant "${VARIANT}" --model-dir "${FINAL_OUT}" \
  --max-len "${MAX_LEN}" --batch-size "${EVAL_BATCH_SIZE}" \
  --threshold 0.55 \
  --tune-field-thresholds 0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85 \
  --tune-limit -1 \
  --out "${FINAL_OUT}/eval_sentence_acc.json"

echo
echo "Finished: ${FINAL_OUT}/eval_sentence_acc.json"
