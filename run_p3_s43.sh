#!/bin/bash
# P3 seed43 replication chain (log entry 二十三): train cape & cape_noswap, eval retrieval.
set -x
cd /mnt/e/world_model/id_vs_cape || exit 9
PY=/home/yjy/miniconda3/envs/lawam/bin/python
mkdir -p results/p3_s43

date
"$PY" train.py --arm cape --seed 43 --tag _s43
E1=$?
"$PY" eval_retrieval.py --arm cape --tag _s43
E2=$?
date
"$PY" train.py --arm cape --no-swap --seed 43 --tag _noswap_s43
E3=$?
"$PY" eval_retrieval.py --arm cape --tag _noswap_s43
E4=$?
date
echo "RESULT cape_s43: train=$E1 eval=$E2  cape_noswap_s43: train=$E3 eval=$E4"
