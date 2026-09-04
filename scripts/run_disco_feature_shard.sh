#!/usr/bin/env bash
set -eo pipefail

SHARD_INDEX="${1:?usage: run_disco_feature_shard.sh SHARD_INDEX}"
PROJECT=/home/67/Paper06-Unified-Geometric-Framework
DATA_ROOT=/data/67/paper18/datasets_v2/canonical_20260722T191500
HF_DATA_CACHE=/data/67/paper14/hf_cache/datasets

source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export HF_HOME=/data/Paper06/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "${PROJECT}"

python3 code/extract_npu_pool.py \
  code/mv_manifest.csv \
  /data/Paper06/features/mv_npu \
  --data-root "${DATA_ROOT}" \
  --hf-data-cache "${HF_DATA_CACHE}" \
  --npu "${SHARD_INDEX}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count 4 \
  --samples 1000 \
  --batch-size 64 \
  --workers 0

python3 code/extract_npu_pool.py \
  code/ex_manifest.csv \
  /data/Paper06/features/ex_npu \
  --data-root "${DATA_ROOT}" \
  --hf-data-cache "${HF_DATA_CACHE}" \
  --npu "${SHARD_INDEX}" \
  --shard-index "${SHARD_INDEX}" \
  --shard-count 4 \
  --samples 0 \
  --batch-size 64 \
  --workers 0
