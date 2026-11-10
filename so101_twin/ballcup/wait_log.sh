#!/bin/zsh
# wait until the ballcup SAM job <tag> has finished (its log, out/seg*_<tag>.log, says done / exit / Traceback)
cd /Users/owner/Downloads/whatif/inference-testing/so101_twin/ballcup/out
until grep -q "^exit \|Traceback" seg*_$1.log 2>/dev/null; do sleep 4; done
grep -v -i "warn\|loading weights\|sam2_video\|processor" seg*_$1.log | tail -5
