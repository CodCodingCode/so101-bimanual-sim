#!/bin/zsh
# Run one pos5 SAM segmentation when the shared slot is free: waits while any PYTHON process matches
# "so101_twin/.*segment" (shell wrappers whose command line merely contains the pattern are ignored).
#   so101_twin/pos5/sam_wait.sh --prompts _tw2,_tw2s100 --ck _ck --tag _s2
cd "$(dirname $0)/../.."
while pgrep -f "so101_twin/.*segment" | xargs -I{} ps -o comm= -p {} 2>/dev/null | grep -qi python; do sleep 10; done
exec neuroscope/molmoact2/.venv/bin/python so101_twin/pos5/segment.py "$@"
