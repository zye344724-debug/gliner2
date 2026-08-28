#!/usr/bin/env bash
# 只准备数据 + 校验，不下载模型、不训练
set -euo pipefail

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TEST_DIR"

PYTHON="${PYTHON:-/Users/zhangye/anaconda3/bin/python}"
export PYTHONPATH="${TEST_DIR}/..:${PYTHONPATH:-}"

mkdir -p data logs

echo "==> [1/3] full schema (保留多笔成交)"
"${PYTHON}" prepare_data.py --schema-mode full --out-dir "${TEST_DIR}/data"

echo "==> [2/3] full + split-multi (可切分的多笔拆成单句)"
"${PYTHON}" prepare_data.py --schema-mode full --split-multi

echo "==> [3/3] validate full-field splits"
"${PYTHON}" validate_data.py --data-dirs "${TEST_DIR}/data/full" "${TEST_DIR}/data/full_split"

echo "Done. Data under: ${TEST_DIR}/data/{full,full_split}"
