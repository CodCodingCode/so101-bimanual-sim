#!/bin/zsh
# Launch one orangebox SAM/DINO job when the shared segmentation slot is free: nothing matches
# pgrep -f "so101_twin/.*segment" (coordinator's rule, 07:50; ignoring wait-script shells and tail/grep log watchers),
# free for 12 s. Every segmentation script keeps "segment" in its name so the other sessions' checks see it.
# Usage: sam_wait.sh <job> [extra args]
cd /Users/owner/Downloads/whatif/inference-testing
PY=neuroscope/molmoact2/.venv/bin/python
# ignored: wait-script shells, tail/grep log watchers and shell wrappers (zsh/bash -c ...) whose command TEXT merely
# contains the pattern (another session's monitor loop did, 08:19); real segmentation runs are python processes.
busy() { pgrep -fl "so101_twin/.*segment" | grep -v -E "sam_wait|^[0-9]+ (-c |-?/bin/zsh|-?/bin/bash|-?zsh|-?bash|tail|grep|/usr/bin/tail|/usr/bin/grep)" | grep -q . ; }
job=$1; shift
case "$job" in
  dino) CMD=($PY -u so101_twin/orangebox/segment_dino.py --tag _dino "$@") ;;
  dinotest) CMD=($PY -u so101_twin/orangebox/segment_dino.py --tag _dinotest --frames 0,2,5,40,100,170,250,300) ;;
  cons) CMD=($PY -u so101_twin/orangebox/segment_dino.py "$@") ;;
  *) echo "unknown job $job"; exit 1 ;;
esac
while true; do
  if busy; then sleep 4; continue; fi
  sleep 12
  if busy; then continue; fi
  echo "launch $job $(date)"
  "${CMD[@]}"
  echo "exit $? $(date)"
  break
done
