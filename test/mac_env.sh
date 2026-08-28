#!/usr/bin/env bash
# Source this on Mac before training:  source test/mac_env.sh
export PYTHON="${PYTHON:-/Users/zhangye/anaconda3/bin/python}"

# Avoid HF xet transfer hangs behind Clash/Surge proxy on macOS
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_TELEMETRY=1

# Mac CPU training: keep threads reasonable
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

# Optional: uncomment to bypass system proxy for HF downloads
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
