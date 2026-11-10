#!/bin/zsh
# GRAYBLOCK final pipeline (after the r3 fits): choose the final fit on EVEN episodes, score (strict + real-occluder-
# excluded; twin-independent check), meta extras, objects, render the twin set + before_after.jpg.
set -e
cd /Users/owner/Downloads/whatif/inference-testing
PY=.venv/bin/python
G=so101_twin/grayblock
CANDS=${CANDS:-out/grayblock_r2_so101_fit.json,out/grayblock_r2_so100_fit.json,out/grayblock_r3a_so101_fit.json,out/grayblock_r3b_so101_fit.json}
(cd $G && ../../$PY finalize.py --mask-tag _t2o --cands $CANDS)
$PY $G/score.py --fit $G/out/final_fit.json --mask-tag _t2o --occ-tag _occx --out $G/out/score_final.json
$PY $G/score.py --fit $G/out/final_fit.json --mask-tag _d1kc --out $G/out/score_final_d1kc.json
$PY $G/make_meta_extra.py
$PY $G/objects.py --fit $G/out/final_fit.json
$PY $G/render_set.py --fit $G/out/final_fit.json --score $G/out/score_final.json
