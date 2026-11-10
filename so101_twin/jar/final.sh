#!/bin/zsh
# JAR: final steps after the candidate fits exist (finalize -> scores -> objects -> twin set -> README)
set -e
cd /Users/owner/Downloads/whatif/inference-testing
PY=.venv/bin/python
(cd so101_twin/jar && ../../$PY finalize.py --mask-tag _t2o --cands "$1")
$PY so101_twin/jar/score.py --fit so101_twin/jar/out/final_fit.json --mask-tag _t2o --occ-tag _occ --out so101_twin/jar/out/score_final.json
$PY so101_twin/jar/score.py --fit so101_twin/jar/out/final_fit.json --mask-tag _d1k --out so101_twin/jar/out/score_final_d1k.json
$PY so101_twin/jar/make_meta_extra.py
$PY -u so101_twin/jar/objects.py --fit so101_twin/jar/out/final_fit.json
$PY -u so101_twin/jar/render_set.py --fit so101_twin/jar/out/final_fit.json --score so101_twin/jar/out/score_final.json
$PY so101_twin/jar/write_readme.py
echo FINAL DONE
