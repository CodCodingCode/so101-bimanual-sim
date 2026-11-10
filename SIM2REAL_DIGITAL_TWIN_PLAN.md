# Digital twin for the sim-to-real adapter

Status: plan, 2026-11-06. Nothing built yet.

## The hypothesis this serves

pi05_libero's real-vs-sim action gap is carried by the wrist-camera K/V at backbone
layers 14-15 (findings F-sim2real_B-20261106-183203: 85% recovered, 97.5% on the
held-out half; random rows 6-11%; base camera 11%). MLP neurons carry none of it.
A low-rank adapter on that route, trained to map real wrist K/V onto the paired sim
frame's K/V, should close most of the offline action gap with everything else frozen.

The adapter needs real and sim frames of the _same robot state and scene_. The
current task-group pairing (any VIOLA mug frame vs any LIBERO mug frame) was good
enough to locate the route and is too loose to train against: much of the K/V
difference between two different scenes is legitimate. The twin fixes the pairing.

Both lever findings are still waiting for the skeptic. Phase 0 checks that before
anything is built.

## Why VIOLA + LIBERO, not DROID + the MuJoCo workcell

|          | VIOLA (real)                      | LIBERO (sim)                     |
| -------- | --------------------------------- | -------------------------------- |
| robot    | Franka Panda                      | Franka Panda (robosuite)         |
| cameras  | agentview + eye_in_hand, 224x224  | agentview + eye_in_hand, 256x256 |
| state    | 8-dim (ee pose + gripper)         | 8-dim, same layout               |
| policy   | pi05_libero, libero prompt format | same                             |
| episodes | 135, 3 tasks, 68,913 frames       | 1,693                            |

Same robot, same camera conventions, same lab, same policy. DROID is the fallback:
pi05_base, 7-dim joint state, but the wrist route carries only 25% there and the
rest arrives through the state text, so the adapter hypothesis is weaker.

VIOLA tasks: (0) bowl from lazy Susan onto plate, (1) arrange plate and fork,
(2) make coffee. Start with task 1: rigid objects, no articulated turntable, no
coffee machine.

## Phases

### Phase 0: gate (hours)

- Confirm the two lever findings pass the skeptic. If the wrist K/V share drops
  below ~50% on re-test, stop and rethink the route before building a twin.
- Decode the VIOLA `observation.state` layout (ee pos 3, ee rot 3, gripper 2 is
  the LIBERO convention; verify against the dataset README and a plotted episode).

### Phase 1: state replay (days)

Goal: for every VIOLA frame, a LIBERO render with the Panda in the same configuration.

- Build a robosuite scene with a plate and a fork on a table. LIBERO's object
  library has plates and utensils; check `libero_service` for how scenes are loaded.
- Drive the sim arm to VIOLA's ee pose each frame with robosuite's OSC controller
  or direct IK, set the gripper from VIOLA's gripper state.
- Objects: estimate initial placement per episode from the first agentview frame.
  While the gripper is closed and within reach of an object, attach the object to
  the gripper. Rigid, scripted, no physics needed for the images.
- Output: an npz per episode with real frames, twin frames, state, and the frame
  index, in the format the recorder already reads.

### Phase 2: camera calibration (days, the fidelity step)

The arm's geometry is known exactly, so use it as the calibration target.

- For each camera, optimise the sim camera pose and FOV to maximise silhouette
  overlap between the rendered Panda and the arm in the real frame, over ~20
  frames with varied poses. A segmentation of the real arm from a generic
  segmenter is enough.
- The eye-in-hand camera matters most: the lever is the wrist route, and
  F-sim2real_A-20261106-174149 shows the strongest wrist units fire on the
  fingertip band, which LIBERO's wrist view shows and VIOLA's does not. The twin's
  wrist camera must sit where VIOLA's does, not where LIBERO's does.
- Match render size (VIOLA is 224x224, LIBERO renders 256x256 and the policy
  resizes) and the letterbox padding, since padding alone moves the base model's
  actions by about the noise floor (F-embodiment_A-20261106-003501).

### Phase 3: measure the twin before training anything (hours)

Record real and twin frames through the runtime and answer, per pair, on the
odd-episode half:

1. Action gap real vs twin, per dim, vs the flow-noise floor. Compare with the
   task-group pairing's 0.34-0.48. If matched state and camera alone shrink the
   gap most of the way, the gap was geometry and content, and the adapter has
   little left to do. That is itself a result.
2. Wrist K/V patch at L14-15 from twin into real: share recovered. Expect it to
   hold or rise with clean pairs.
3. Meter check: siglip:10:2952 and the backbone L8/L11 units should still read
   the twin as sim. If the twin reads as real, the twin is too good and the
   adapter has nothing to learn; if real frames start reading as sim after the
   adapter, that is a side effect worth logging, not a success criterion.

### Phase 4: the adapter (days)

- LoRA on `k_proj` and `v_proj` of backbone layers 14 and 15, applied only to
  wrist-camera token positions (the runtime already knows the token layout).
- Loss: MSE between adapted real K/V and the twin's K/V at those layers, plus an
  action-consistency term: policy action on the adapted real frame vs policy
  action on the twin frame, same noise seed.
- Train on even episodes, evaluate on odd. Three controls, all required:
  no adapter; the same adapter on two random backbone layers; the flow-noise floor.
- Preservation: LIBERO actions with the adapter on must stay within the noise
  floor of actions with it off. The adapter touches only wrist tokens, so the
  only way this fails is if it also fires on LIBERO wrist frames; check it.
- Success: adapted real actions land within the noise floor of twin actions on
  held-out episodes, and the random-layer adapter does not.

### Phase 5: closed loop (needs hardware)

Offline success does not say the real actions were wrong to begin with. The only
test of that is a real VIOLA-style setup running pi05_libero with and without the
adapter. Until then the claim is "closes the offline action gap", not "improves
real success".

## What would kill it, and when we would know

- Phase 0: lever findings do not survive the skeptic.
- Phase 3: matched state and camera close the gap on their own. Good news for
  sim design, no adapter needed.
- Phase 4: random-layer adapter does as well as the targeted one. The route was
  real but not where the fix belongs.
- Phase 4: LIBERO actions move. The adapter is not confined to the wrist route.

## Not in scope

- Real-to-real embodiment transfer. The gripper concept is written by different
  expert neurons per action space (F-embodiment_B-20261106-025936), so a different
  robot is an output-head problem, not an attention-route problem.
- Tasks 0 and 2 until task 1 works.
- Training anything on activations other than the adapter.

## Update 2026-11-06: SO-101 variant

The user has SO-101 arms and few real objects, and can build an accurate sim scene.
This changes the plan in three ways.

1. **Closed loop becomes possible.** Phase 5 was the plan's weakest point with VIOLA
   (no hardware). With SO-101 on the bench, real success rate with and without the
   adapter is measurable. That is the only number that settles whether the offline
   action gap was behaviour that needed fixing.
2. **The route has to be re-found.** The wrist K/V L14-15 finding is for pi05_libero
   on a Franka. The SO-101 policy is a fresh pi0.5 fine-tune on SO-101 data, with
   a different camera rig and 6-dim action space. Run the per-layer, per-camera K/V
   patch on that policy first; do not assume L14-15 or the wrist camera. Community
   SO-101 pi0.5 fine-tunes exist (dongyoonkim/so101-pi05-base, ljt228/pi05-so101-finetune)
   and MuJoCo SO-101 models exist (lucagianforcaro/so101-sim-to-real, johnsutor/so101-nexus,
   mujoco_menagerie trs_so_arm100).
3. **Few objects is an advantage.** The twin needs a small, fixed object set that
   is modelled exactly. Pick 2-3 objects and model them well.

Revised order for SO-101:

- Build the sim scene to match the real bench: table, camera mounts, the 2-3 objects.
- Collect sim demos (scripted or teleop in sim) and fine-tune pi0.5 on them. This is
  the sim-trained policy whose real gap we want to close.
- Collect ~30 real episodes with the same objects. They do not need to succeed; they
  are the state trajectories to replay in sim for paired frames, and the real test set.
- Replay real states in sim -> paired frames (Phase 1-2 above, SO-101 kinematics).
- Localise: action gap per pair vs noise floor; K/V patch per layer and per camera.
- Adapter on whatever route commits. Controls as above.
- Closed loop on the real arm: success with vs without adapter, vs the policy
  LoRA-fine-tuned on the same 30 real episodes (the cheap baseline that has to be beaten).

Prior work checked 2026-11-06: no paper found that combines causal localisation of a
camera-specific attention route with a paired-twin alignment adapter. Closest:
Robotic Steering (arXiv 2511.22697) selects pi0/pi0.5 attention heads by k-NN and
LoRAs Q+O for few-shot real task adaptation, explicitly freezing K/V; Surgical
Fine-Tuning (arXiv 2210.11466) tunes a layer subset per shift type, non-robotic;
Real-is-Sim / TwinAligner build digital twins for evaluation and real2sim2real, not
as a paired training signal for a localised adapter.

## Update 2026-11-06 (later): bimanual SO-101, the method

No generalist runs a bimanual SO-101 zero-shot (MolmoAct2 bimanual = YAM only), so a
fine-tune is unavoidable. The study stays clean by fine-tuning on SIM data only and
controlling the small-data effect with a matched real-data fine-tune.

Policies (all the same base, same LoRA recipe, same step count):
- S     : fine-tuned on sim demos only. The subject.
- S+DR  : sim demos with domain randomisation. The standard sim2real baseline.
- S+A   : S plus the route adapter trained on twin pairs. The claim.
- S+rand: S plus the same adapter on random layers. The control.
- R     : fine-tuned on the N real demos. The oracle / cheap baseline.

Base model: pi0.5 first (recorder, attribution, K/V patch, skeptic already exist;
12-D bimanual action padded to 32). MolmoAct2-BimanualYAM as the replication target
later (14-D, top/left/right cameras, explicit P_K/P_V connectors).

Steps:
1. Twin: two SO-101s in MuJoCo, top + two wrist cameras at measured poses, 2-3
   objects modelled exactly. Calibrate camera poses by arm-silhouette overlap.
2. Sim demos: 1-2 bimanual tasks (handover, two-hand lift), 200-500 scripted or
   sim-teleop episodes, no randomisation for S; a randomised copy for S+DR.
3. Real demos: 30-50 teleop episodes, same tasks, same objects. Used three ways:
   training R, state replay for paired frames, real test set. Never used to train S.
4. Fine-tune S, S+DR, R.
5. Gap: deploy S on the real rig (success rate), and offline per-pair action gap vs
   the flow-noise floor on the twin pairs.
6. Localise on S: per-layer, per-camera K/V patch twin->real; attribution; MLP group
   patch as the expected null. Do not assume wrist or L14-15; measure.
7. Adapter on the route that commits; S+rand on random layers.
8. Closed loop: S, S+DR, S+A, S+rand, R, 30 trials per task each. The claim holds if
   S+A beats S+DR and S+rand and approaches R.

## Build log

- 2026-11-06: `so101_twin/` created. Bimanual SO-100 workcell MJCF builds and renders (top + two wrist
  cameras); `replay_real.py` (Phase 1 state replay) runs on a synthetic LeRobot dataset;
  `MOLMOACT2_PORT.md` lists the hook points and the adapter home (`ActionExpert.context_k_proj/v_proj`).
  Compute decision: M3 Max 64 GB, no CUDA -> inference and recording local, fine-tunes on a rented 80 GB GPU.
  Blocking on the user: object set and the two tasks.
