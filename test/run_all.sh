#!/usr/bin/env bash
# 两阶段流水线：可只准备数据 / 只训练 / 全流程
#
# 常用：
#   ./prepare_all.sh              # 只准备数据+校验（不训练）
#   PREPARE_ONLY=1 ./run_all.sh   # 同上 + 可选下载模型
#   TRAIN_ONLY=1 ./run_all.sh       # 跳过数据准备，直接训练
#   PROFILE=gpu4060 ./run_all.sh  # 4060 推荐超参
#   DATA_VARIANT=full_split ./run_all.sh
#
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_DIR"

PROFILE="${PROFILE:-mac}"
if [[ "${PROFILE}" == "gpu4060" ]]; then
  # shellcheck disable=SC1091
  source "${TEST_DIR}/gpu_env.sh"
else
  # shellcheck disable=SC1091
  source "${TEST_DIR}/mac_env.sh"
fi

PYTHON="${PYTHON:-/Users/zhangye/anaconda3/bin/python}"
SCHEMA_MODE="${SCHEMA_MODE:-full}"       # full | core
DATA_VARIANT="${DATA_VARIANT:-${SCHEMA_MODE}}"  # full | full_split | core
SMOKE="${SMOKE:-0}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
TRAIN_ONLY="${TRAIN_ONLY:-0}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

export PYTHONPATH="${TEST_DIR}/..:${PYTHONPATH:-}"
mkdir -p data outputs logs models

DATA_DIR="${TEST_DIR}/data/${DATA_VARIANT}"

if [[ "${TRAIN_ONLY}" != "1" ]]; then
  echo "==> prepare_data (schema=${SCHEMA_MODE}, variant=${DATA_VARIANT})"
  PREPARE_ARGS=(--schema-mode "${SCHEMA_MODE}")
  if [[ "${DATA_VARIANT}" == "full_split" ]]; then
    PREPARE_ARGS+=(--split-multi --variant full_split)
  elif [[ "${DATA_VARIANT}" != "${SCHEMA_MODE}" ]]; then
    PREPARE_ARGS+=(--variant "${DATA_VARIANT}")
  fi
  if [[ "${SMOKE}" == "1" ]]; then
    PREPARE_ARGS+=(--max-samples 200 --out-dir "${TEST_DIR}/data/smoke")
    DATA_DIR="${TEST_DIR}/data/smoke/${DATA_VARIANT}"
  fi
  "${PYTHON}" prepare_data.py "${PREPARE_ARGS[@]}" 2>&1 | tee "logs/prepare_${DATA_VARIANT}.log"
  "${PYTHON}" validate_data.py --data-dirs "${DATA_DIR}" 2>&1 | tee "logs/validate_${DATA_VARIANT}.log"
fi

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "PREPARE_ONLY=1, skip download & training."
  exit 0
fi

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  echo "==> ensure local model weights"
  "${PYTHON}" ensure_model.py 2>&1 | tee "logs/ensure_model.log"
fi

NER_ARGS=(
  --schema-mode "${SCHEMA_MODE}"
  --data-variant "${DATA_VARIANT}"
  --epochs "${NER_EPOCHS:-5}"
  --batch-size "${BATCH_SIZE:-2}"
  --grad-accum "${GRAD_ACCUM:-8}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}"
  --max-len "${MAX_LEN:-384}"
  --eval-steps "${EVAL_STEPS:-200}"
)
STRUCT_ARGS=(
  --schema-mode "${SCHEMA_MODE}"
  --data-variant "${DATA_VARIANT}"
  --epochs "${STRUCTURE_EPOCHS:-8}"
  --batch-size "${BATCH_SIZE:-2}"
  --grad-accum "${GRAD_ACCUM:-8}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}"
  --max-len "${MAX_LEN:-384}"
  --eval-steps "${EVAL_STEPS:-200}"
)
EVAL_ARGS=(--schema-mode "${SCHEMA_MODE}" --data-variant "${DATA_VARIANT}")

if [[ "${SMOKE}" == "1" ]]; then
  NER_ARGS+=(--epochs 1 --max-train-samples 64 --max-eval-samples 32 --eval-steps 20)
  STRUCT_ARGS+=(--epochs 1 --max-train-samples 64 --max-eval-samples 32 --eval-steps 20)
  NER_ARGS+=(--allow-missing-field-labels)
  STRUCT_ARGS+=(--allow-missing-field-labels)
  EVAL_ARGS+=(--limit 32)
  NER_ARGS+=(--train-file "${DATA_DIR}/ner_train_clean.jsonl" --eval-file "${DATA_DIR}/ner_val_clean.jsonl"
    --output-dir "${TEST_DIR}/outputs/ner/smoke_${DATA_VARIANT}")
  STRUCT_ARGS+=(--train-file "${DATA_DIR}/structure_train_clean.jsonl" --eval-file "${DATA_DIR}/structure_val_clean.jsonl"
    --output-dir "${TEST_DIR}/outputs/structure/smoke_${DATA_VARIANT}"
    --init-from "${TEST_DIR}/outputs/ner/smoke_${DATA_VARIANT}")
  EVAL_ARGS+=(--test-file "${DATA_DIR}/structure_test_clean.jsonl"
    --model-dir "${TEST_DIR}/outputs/structure/smoke_${DATA_VARIANT}")
fi

if "${PYTHON}" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
  NER_ARGS+=(--fp16)
  STRUCT_ARGS+=(--fp16)
fi

echo "==> train NER"
"${PYTHON}" train_ner.py "${NER_ARGS[@]}" 2>&1 | tee "logs/train_ner_${DATA_VARIANT}.log"

echo "==> train structure (from NER)"
"${PYTHON}" train_structure.py "${STRUCT_ARGS[@]}" 2>&1 | tee "logs/train_structure_${DATA_VARIANT}.log"

echo "==> evaluate sentence accuracy"
"${PYTHON}" evaluate_sentence_acc.py "${EVAL_ARGS[@]}" 2>&1 | tee "logs/eval_${DATA_VARIANT}.log"

echo "Done. profile=${PROFILE} variant=${DATA_VARIANT}"
