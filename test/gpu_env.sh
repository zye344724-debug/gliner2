#!/usr/bin/env bash
# RTX 4060 / CUDA 服务器环境变量
# 用法: source gpu_env.sh && ./run_all.sh

export PYTHON="${PYTHON:-python}"

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# 4060 8GB 推荐训练超参（可被命令行覆盖）
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export NER_EPOCHS="${NER_EPOCHS:-5}"
export STRUCTURE_EPOCHS="${STRUCTURE_EPOCHS:-8}"
export MAX_LEN="${MAX_LEN:-384}"
export EVAL_STEPS="${EVAL_STEPS:-200}"
