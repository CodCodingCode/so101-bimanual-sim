#!/bin/zsh
# Recovery fine-tune v2: demos from the RE-PLANNING teacher (re-clamps after slips / nudges), then PPO from the
# 100 %-clean base policy with a stronger BC pull; 30 s episodes with nudges, mid-carry slips, arm pushes.
set -e
cd "$(dirname "$0")/.."
export SQUEEZE_TASK=clamp CLAMP_RECOVERY=1
PY=~/.pyenv/versions/3.10.9/bin/python
echo "== demos (recovery teacher) $(date +%T)"; $PY -W ignore -u squeeze/train.py demos --episodes 300
echo "== ppo (recovery) $(date +%T)"; $PY -W ignore -u squeeze/train.py ppo --init-dir squeeze/runs_clamp/ppo --bc-weight 0.5 --steps 2000000
echo "== eval $(date +%T)";  $PY -W ignore -u squeeze/train.py eval squeeze/runs_clamp_recovery/ppo/best.zip --n 100
echo "== done $(date +%T)"
