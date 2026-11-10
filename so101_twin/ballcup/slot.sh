#!/bin/zsh
# who holds the shared SAM slot (rule: pgrep -f "so101_twin/.*segment"); kept in a file so this check's own command
# line never matches the pattern
for p in $(pgrep -f "so101_twin/.*segment"); do ps -ww -o pid,etime,command -p $p | tail -1 | cut -c1-160; done
