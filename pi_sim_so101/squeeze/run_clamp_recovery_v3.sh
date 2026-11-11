#!/bin/zsh
# Recovery fine-tune v3: same as v2 (re-planning teacher demos, PPO from the clean base, BC 0.5) plus
# TIP_PENALTY 60 and a box-shove penalty -- v2 learned that knocking the box over ends a bad episode for free.
set -e
cd "$(dirname "$0")/.."
export SQUEEZE_TASK=clamp CLAMP_RECOVERY=1
PY=~/.pyenv/versions/3.10.9/bin/python
echo "== ppo (recovery v3) $(date +%T)"; $PY -W ignore -u squeeze/train.py ppo --init-dir squeeze/runs_clamp/ppo --bc-weight 0.5 --steps 2000000
echo "== eval $(date +%T)";  $PY -W ignore -u squeeze/train.py eval squeeze/runs_clamp_recovery/ppo/best.zip --n 100
echo "== done $(date +%T)"
