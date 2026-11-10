#!/bin/zsh
# GRAYBLOCK (copy of jar/sam_wait.sh): launch grayblock SAM jobs, one at a time, each only when the shared SAM slot is
# free: no PYTHON process whose command line matches "so101_twin/.*segment" (shell wrappers / monitors whose text
# merely contains the pattern are ignored), free for 12 s. This script's own command line never contains the pattern.
# Usage: sam_wait.sh <job> [<job> ...]   (jobs run in order, the slot is re-checked before each)
cd /Users/owner/Downloads/whatif/inference-testing
PY=neuroscope/molmoact2/.venv/bin/python
busy() {
  for p in $(pgrep -f "so101_twin/.*segment"); do
    c=$(ps -ww -o command= -p $p 2>/dev/null)
    case "$c" in
      *[Pp]ython*) echo "$c" | awk '{print $1}' | grep -qi python && return 0 ;;
    esac
  done
  return 1
}
for job in "$@"; do
  case "$job" in
    d1) CMD=($PY -u so101_twin/grayblock/segment.py --mode dino --tag _d1) ;;
    occ) CMD=($PY -u so101_twin/grayblock/segment.py --mode occ --tag _occ) ;;
    t1) CMD=($PY -u so101_twin/grayblock/segment.py --mode twin --prompts so101_twin/grayblock/masks/prompts_r1 --tag _t1 --cons-tag _d1kc --obj-tag _occ) ;;
    t2) CMD=($PY -u so101_twin/grayblock/segment.py --mode twin --prompts so101_twin/grayblock/masks/prompts_r2 --tag _t2 --cons-tag _d1kc --obj-tag _occ) ;;
    *) echo "unknown job $job"; exit 1 ;;
  esac
  while true; do
    if busy; then sleep 4; continue; fi
    sleep 12
    if busy; then continue; fi
    echo "launch $job $(date)"
    "${CMD[@]}"
    echo "exit $job $? $(date)"
    break
  done
done
