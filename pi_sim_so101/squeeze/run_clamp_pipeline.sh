#!/bin/zsh
# Two-arm TIP CLAMP into the grey box (no jaws), calibrated sim (white arms, oak table photo, real jaw shapes,
# friction 0.35-0.6, fitted servos, solid camera post), ~70% placement randomisation, contact penalties.
# demos (scripted ClampPlan + DART noise) -> BC warm start -> PPO (BC term decays) -> eval. Output: runs_clamp/
set -e
cd "$(dirname "$0")/.."
export SQUEEZE_TASK=clamp
PY=~/.pyenv/versions/3.10.9/bin/python
echo "== demos $(date +%T)"; $PY -W ignore -u squeeze/train.py demos --episodes 240
echo "== bc $(date +%T)";    $PY -W ignore -u squeeze/train.py bc
echo "== ppo $(date +%T)";   $PY -W ignore -u squeeze/train.py ppo --steps 3000000
echo "== eval $(date +%T)";  $PY -W ignore -u squeeze/train.py eval squeeze/runs_clamp/ppo/best.zip --n 100
echo "== done $(date +%T)"
