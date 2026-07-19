#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m pip install --disable-pip-version-check --no-cache-dir -r requirements.txt
torchrun --standalone --nproc-per-node=8 train_reproduce.py --config config.json

