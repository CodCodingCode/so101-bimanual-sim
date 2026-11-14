#!/bin/zsh
# Clamp v3 policy (real joint ranges, gentle squeeze, camera-mount clearance): clean teacher demos with disturbances ->
# behaviour-cloning warm start -> DAgger (the re-planning teacher labels what the policy reaches) -> held-out test.
set -e
cd "$(dirname "$0")/.."
export SQUEEZE_TASK=clamp CLAMP_RECOVERY=1 CLAMP_RUNS=runs_clamp_v3
PY=~/.pyenv/versions/3.10.9/bin/python
echo "== demos $(date +%T)"; $PY -W ignore -u squeeze/train.py demos --episodes 300
echo "== bc $(date +%T)";    $PY -W ignore -u squeeze/train.py bc --epochs 30 --n 48
echo "== dagger $(date +%T)"; DAGGER_BASE=runs_clamp_v3/bc DAGGER_OUT=runs_clamp_v3/dagger $PY -W ignore -u squeeze/dagger.py --rounds 8 --betas 0.5,0.3,0.2,0.1,0,0,0,0
echo "== done $(date +%T)"
