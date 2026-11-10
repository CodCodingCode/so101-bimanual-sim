#!/bin/zsh
# Launch one ballcup SAM job when the shared SAM slot is free (no python process running a *segment* script, free
# for 12 s, so waiters queued before us launch first). This script's own command line never mentions python/segment
# (other sessions' waiters grep for "[Pp]ython.*segment"). Usage: sam_wait.sh <job>  (jobs below)
cd /Users/owner/Downloads/whatif/inference-testing
PY=neuroscope/molmoact2/.venv/bin/python
# shared rule (coordinator, 2026-11-07): no segmentation (DINO / SAM) while pgrep -f "so101_twin/.*segment" finds anything
busy() { pgrep -f "so101_twin/.*segment" >/dev/null; }
case "$1" in
  d1) CMD=($PY -u so101_twin/ballcup/segment.py --mode dino --tag _d1) ;;
  t0) CMD=($PY -u so101_twin/ballcup/segment.py --mode twin --prompts so101_twin/ballcup/masks/prompts_test --tag _t0 --limit 3) ;;
  t1) CMD=($PY -u so101_twin/ballcup/segment.py --mode twin --prompts so101_twin/ballcup/masks/prompts_r1 --tag _t1) ;;
  t1b) CMD=($PY -u so101_twin/ballcup/segment.py --mode twin --prompts so101_twin/ballcup/masks/prompts_r1 --tag _t1b) ;;
  occ) CMD=($PY -u so101_twin/ballcup/segment.py --mode occ --tag _occ) ;;
  t2) CMD=($PY -u so101_twin/ballcup/segment.py --mode twin --prompts so101_twin/ballcup/masks/prompts_r2 --tag _t2) ;;
  *) echo "unknown job $1"; exit 1 ;;
esac
while true; do
  if busy; then sleep 4; continue; fi
  sleep 12
  if busy; then continue; fi
  echo "launch $1 $(date)"
  "${CMD[@]}"
  echo "exit $? $(date)"
  break
done
