#!/bin/zsh
# One SAM refinement round for a scene:  sam_round.sh <scene> <model> <init_json> <tag> [rounds]
#   1. export twin prompts for every view from <init_json>   2. SAM masks (MolmoAct2 venv)   3. refit on SAM masks
set -e
cd "$(dirname $0)/../.."
SCENE=$1; MODEL=$2; INIT=$3; TAG=$4; ROUNDS=${5:-2}
OUT=so101_twin/fitkit/out
.venv/bin/python - "$SCENE" "$MODEL" "$INIT" <<'PY'
import sys, json; sys.path.insert(0, 'so101_twin/fitkit')
import armfit
scene, model, init = sys.argv[1:4]
cfg = armfit.scene_config(scene)
F = armfit.Fit(cfg, model)
F.load_state(init)
if cfg["real_mask"] == "auto":
    for v in F.views: F.train_colour(v)
json.dump({v: vc["cam"] for v, vc in F.views.items()}, open(f"so101_twin/fitkit/out/{scene}_views.json", "w"))
for v in F.views:
    print(v, "prompted frames", F.export_prompts(v, armfit.TWIN / "fitkit" / "masks" / f"prompts_{cfg['real']}__{v}.npy"))
PY
SET=$(.venv/bin/python -c "import sys; sys.path.insert(0,'so101_twin/fitkit'); import armfit; print(armfit.scene_config('$SCENE')['real'])")
for V in $(.venv/bin/python -c "import json; print(' '.join(json.load(open('$OUT/${SCENE}_views.json'))))"); do
  CAM=$(.venv/bin/python -c "import json; print(json.load(open('$OUT/${SCENE}_views.json'))['$V'])")
  # one SAM process at a time across sessions (agreed with inference-testing-72): wait while any is alive
  while pgrep -f "so101_twin/.*segment" >/dev/null; do sleep 10; done
  neuroscope/molmoact2/.venv/bin/python so101_twin/fitkit/segment_arm.py $SET $CAM --prompts so101_twin/fitkit/masks/prompts_${SET}__${V}.npy --view $V 2>&1 | grep -E "saved|/"
done
.venv/bin/python so101_twin/fitkit/armfit.py $SCENE --model $MODEL --init $INIT --sam --rounds $ROUNDS --tag $TAG
