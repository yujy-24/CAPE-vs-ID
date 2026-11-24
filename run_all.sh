#!/usr/bin/env bash
# Full pipeline (v2 protocol, design doc §11). Run inside WSL:
#   bash run_all.sh [stage]
# Stages: env, manifest, cache, smoke, train, eval, controls   (default: smoke)
set -e
PY=${PY:-$HOME/miniconda3/envs/lawam/bin/python}
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
STAGE=${1:-smoke}

if [ "$STAGE" = "env" ] || [ "$STAGE" = "smoke" ]; then
  echo "=== stage 1: env check ==="
  "$PY" 00_check_env.py
fi
if [ "$STAGE" = "manifest" ] || [ "$STAGE" = "smoke" ]; then
  echo "=== stage 2: manifest v2 (CAPE-faithful branch sampling) ==="
  "$PY" build_manifest.py
fi
if [ "$STAGE" = "smoke" ]; then
  echo "=== stage 3: module self-test ==="
  "$PY" modules.py
fi

case "$STAGE" in
  cache)
    echo "=== stage 4: feature cache v2 (long: ~1-2 h; ~17 GB) ==="
    "$PY" build_cache.py
    echo ">>> alignment already verified on v1 videos (5/5, log entry 三);"
    echo ">>> if dataset unchanged, skip the human jpg check."
    ;;
  train)
    echo "=== stage 5: three arms, 2 seeds (matched steps/seed; per-arm paper hyper) ==="
    for SEED in 42 43; do
      TAG=$([ "$SEED" = 42 ] && echo "" || echo "_seed43")
      "$PY" train.py --arm id   --seed "$SEED" --tag "$TAG"
      "$PY" train.py --arm cape --seed "$SEED" --tag "$TAG"
      "$PY" train.py --arm fwd  --seed "$SEED" --tag "$TAG"
    done
    ;;
  controls)
    echo "=== stage 6b: registered controls ==="
    "$PY" train.py --arm id --shuffle-labels --tag _shuf          # H3 within-task perm
    "$PY" train.py --arm id --form l1 --tag _l1                    # loss-form check
    ;;
  eval)
    echo "=== stage 6: metric battery (seed 42 arms) ==="
    "$PY" probe_r2.py --space frozen --y-shuffle                   # floor control
    "$PY" probe_r2.py --space frozen
    "$PY" probe_r2.py --space id_direct --ckpt cache_v2/ckpt/id/final.pt
    "$PY" probe_r2.py --space cape --ckpt cache_v2/ckpt/cape/final.pt
    "$PY" probe_r2.py --space fwd  --ckpt cache_v2/ckpt/fwd/final.pt
    "$PY" eval_retrieval.py --arm frozen
    "$PY" eval_retrieval.py --arm id
    "$PY" eval_retrieval.py --arm cape
    "$PY" eval_retrieval.py --arm fwd
    "$PY" diag_pairwise_cos.py
    "$PY" diag_pairwise_cos.py --ckpt cache_v2/ckpt/cape/final.pt
    ;;
  *)
    echo "unknown stage '$STAGE' (use env|manifest|cache|smoke|train|controls|eval)"
    ;;
esac
