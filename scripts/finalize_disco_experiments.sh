#!/usr/bin/env bash
set -eo pipefail

PROJECT=/home/67/Paper06-Unified-Geometric-Framework
RESULTS="${PROJECT}/results/disco"
mkdir -p "${RESULTS}"

while true; do
  mv_count="$(find /data/Paper06/features/mv_npu -type f -name '*.npz' | wc -l)"
  ex_count="$(find /data/Paper06/features/ex_npu -type f -name '*.npz' | wc -l)"
  if [[ "${mv_count}" -eq 660 && "${ex_count}" -eq 510 ]]; then
    date '+feature cache complete: %F %T'
    break
  fi
  if ! pgrep -f 'run_disco_feature_shard.sh [0-3]' >/dev/null; then
    echo "feature extraction stopped incomplete: MV=${mv_count}/660 EX=${ex_count}/510" >&2
    exit 1
  fi
  date "+waiting for feature shards: %F %T MV=${mv_count}/660 EX=${ex_count}/510"
  sleep 60
done

cd "${PROJECT}"
for pool in mv ex; do
  input="/data/Paper06/features/${pool}_npu"
  for groups in 6 8; do
    python3 code/run_official_disco.py \
      "${input}" \
      "${RESULTS}/${pool}_official_pca128_g${groups}.csv" \
      --groups "${groups}" \
      --pca-components 128 \
      --workers 16 \
      > "${RESULTS}/${pool}_pca128_g${groups}.log" 2>&1
  done
done

date '+all DiSCO scoring complete: %F %T'
