#!/bin/zsh
# REDLEGO (copy of jar/sam_wait.sh): launch one redlego SAM job when the shared SAM slot is free: no PYTHON process whose
# command line matches "so101_twin/.*segment" (shell wrappers / monitors whose text merely contains the pattern are
# ignored), free for 3 s (polled every second). This script's own command line never contains the pattern. Usage: sam_wait.sh <job>
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
case "$1" in
  test) CMD=($PY -u so101_twin/redlego/segment.py --mode dino --tag _dtest --limit 3) ;;
  d1) CMD=($PY -u so101_twin/redlego/segment.py --mode dino --tag _d1) ;;
  t1) CMD=($PY -u so101_twin/redlego/segment.py --mode twin --prompts so101_twin/redlego/masks/prompts_r1 --tag _t1 --cons-tag _d1k --obj-tag _d1) ;;
  t2) CMD=($PY -u so101_twin/redlego/segment.py --mode twin --prompts so101_twin/redlego/masks/prompts_r2 --tag _t2 --cons-tag _d1k --obj-tag _d1) ;;
  *) echo "unknown job $1"; exit 1 ;;
esac
while true; do
  if busy; then sleep 1; continue; fi
  sleep 3
  if busy; then continue; fi
  echo "launch $1 $(date)"
  "${CMD[@]}"
  echo "exit $? $(date)"
  break
done
