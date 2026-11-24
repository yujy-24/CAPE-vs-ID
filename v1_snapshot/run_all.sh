#!/usr/bin/env bash
# Full pipeline. Run inside WSL:  bash run_all.sh [stage]
# Stages: env, manifest, cache, smoke, train, eval   (default: env+manifest+smoke)
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
  echo "=== stage 2: manifest (task-level 32/8 split) ==="
  "$PY" build_manifest.py
fi
if [ "$STAGE" = "smoke" ]; then
  echo "=== stage 3: module self-test ==="
  "$PY" modules.py
fi

case "$STAGE" in
  cache)
    echo "=== stage 4: feature cache (long: ~1-2 h) ==="
    "$PY" build_cache.py --dump-frames 5
    echo ">>> human step: open cache/align_check/*.jpg and verify action-frame alignment"
    ;;
  train)
    echo "=== stage 5: three arms (matched steps/bs/seed) ==="
    "$PY" train.py --arm id
    "$PY" train.py --arm cape
    "$PY" train.py --arm fwd
    ;;
  eval)
    echo "=== stage 6: metric battery ==="
    "$PY" probe_r2.py --space frozen
    "$PY" probe_r2.py --space id   --ckpt cache/ckpt/id/final.pt
    "$PY" probe_r2.py --space cape --ckpt cache/ckpt/cape/final.pt
    "$PY" probe_r2.py --space fwd  --ckpt cache/ckpt/fwd/final.pt
    "$PY" eval_retrieval.py --arm frozen
    "$PY" eval_retrieval.py --arm id
    "$PY" eval_retrieval.py --arm cape
    "$PY" eval_retrieval.py --arm fwd
    "$PY" diag_pairwise_cos.py
    "$PY" diag_pairwise_cos.py --ckpt cache/ckpt/cape/final.pt
    ;;
  *)
    echo "unknown stage '$STAGE' (use env|manifest|cache|smoke|train|eval)"
    ;;
esac
