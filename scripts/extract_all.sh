#!/usr/bin/env bash
# Cache embeddings for every backbone, one GPU per model, in parallel.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
MODELS=(clip_l14_336 dfn5b_h14_378 siglip_so400m_384 siglip2_so400m_512 laion_bigg14 eva02_e14_plus
        pe_core_bigg_448 siglip2_gopt_384)
mkdir -p logs features
for i in "${!MODELS[@]}"; do
  m=${MODELS[$i]}
  [ -f "features/$m.npz" ] && { echo "skip $m"; continue; }
  CUDA_VISIBLE_DEVICES=$i "$PY" -u extract_features.py --model "$m" > "logs/extract_$m.log" 2>&1 &
done
wait
ls -la features
