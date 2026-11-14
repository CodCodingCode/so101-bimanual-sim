"""Real-rig adapter for the fine-tuned SmolVLA clamp policy (VLA: raw camera images + joint state + instruction ->
joint-target ACTION CHUNKS of 50 steps; n_action_steps of each chunk are executed before it re-plans).

Trained ONLY on sim renders (record_vla.py: calibrated cameras, matched colours). Sim result (eval_smolvla.py, 20k steps,
n_action_steps 25): 8/12 delivered, 6/12 clean. NOT yet run on real frames. First real use = a DRY check: feed live
frames + measured joints, log the predicted joint targets, move nothing.

    from vla_runner_real import VLARunner
    vla = VLARunner("runs_vla/smolvla_20k", device="cuda", n_action_steps=25)
    vla.reset()
    q_cmd = vla.step(frames, joints_deg, gripper_pct)
        # frames: {"overhead", "left_wrist", "right_wrist"}: RGB uint8 (H, W, 3) from the rig cameras (any size;
        #         resized to 320 x 240 like the training data)
        # joints_deg: {"left": 5 angles, "right": 5 angles} in SIM units (= rig units incl. joint_offsets_deg)
        # gripper_pct: {"left": 0..100, "right": 0..100}  (0 = shut; the clamp keeps the jaws shut)
    -> q_cmd: {"left": 6 values, "right": 6 values}: 5 joint targets (deg) + gripper (%), for the runner's
              rate limit / squeeze limiter / safety guard / floor guard, exactly like any other joint command
"""
from __future__ import annotations

import numpy as np

W, H = 320, 240
CAMS = ("overhead", "left_wrist", "right_wrist")
TASK = "Squeeze the red cube between both grippers and drop it into the grey box."


class VLARunner:
    def __init__(self, ckpt_dir, device="cuda", n_action_steps=25):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.torch = torch
        self.policy = SmolVLAPolicy.from_pretrained(str(ckpt_dir)).to(device).eval()
        self.policy.config.n_action_steps = int(n_action_steps)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": device}})

    def reset(self):
        self.policy.reset()

    def obs(self, frames, joints_deg, gripper_pct):
        from PIL import Image
        torch = self.torch
        o = {}
        for c in CAMS:
            im = np.asarray(Image.fromarray(np.asarray(frames[c], np.uint8)).resize((W, H), Image.Resampling.BILINEAR))
            o[f"observation.images.{c}"] = torch.from_numpy(im).permute(2, 0, 1).float() / 255.0
        state = []
        for s in ("left", "right"):
            state += list(np.asarray(joints_deg[s], float)[:5]) + [float(gripper_pct[s])]
        o["observation.state"] = torch.tensor(state, dtype=torch.float32)
        o["task"] = TASK
        return o

    def step(self, frames, joints_deg, gripper_pct):
        with self.torch.no_grad():
            a = self.post(self.policy.select_action(self.pre(self.obs(frames, joints_deg, gripper_pct))))
        a = np.asarray(a.squeeze(0).cpu().numpy(), float)
        return {"left": a[0:6], "right": a[6:12]}
