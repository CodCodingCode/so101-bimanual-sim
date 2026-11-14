#!/usr/bin/env python3
"""Closed-loop sim evaluation of a fine-tuned SmolVLA (LeRobot) on the clamp task, in the SAME env the training data came
from (record_vla.py: clamp_wide.ClampWideEnv, calibrated cameras rendered at 240 x 320, 12-dim joint state / action,
one instruction). The VLA drives the arms with its own joint targets (action chunks of 50 steps; n_action_steps of
each chunk are executed before it re-plans). Runs where the GPU is (warprm: MUJOCO_GL=egl).

    MUJOCO_GL=egl CLAMP_GRIP_DR=1 CLAMP_SLIP=1 python squeeze/eval_smolvla.py \
        --ckpt ~/so101_vla/outputs/<run>/checkpoints/020000/pretrained_model --stage 4 --n 20 --n-action-steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# warprm is shared: unthrottled, one eval took 5-6 cores (torch / BLAS thread pools) and four of them slowed everyone's
# jobs (2026-11-13). The policy runs on the GPU; 2 CPU threads per eval are plenty.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=300000)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--video", default="", help="write an mp4 of the first episodes (overhead view)")
    ap.add_argument("--pushclamp", type=float, default=0.0,
                    help="push-then-clamp test: cube start anywhere in x +-12 cm, y +-this many cm (longer episodes)")
    ap.add_argument("--seeds", default="", help="comma list of episode seeds (overrides --seed0 / --n)")
    ap.add_argument("--cmd-filter", action="store_true",
                    help="vla_safety.CmdFilter between the VLA and the servos (needs CLAMP_GUARD=1 CLAMP_GUARD_CMD=1)")
    ap.add_argument("--relax", action="store_true",
                    help="with --cmd-filter: CmdFilter(relax=True): a rejected command is first pulled back toward the "
                         "MEASURED pose (keeps the carry, drops the too-deep squeeze) before the segment rule")
    ap.add_argument("--dr", action="store_true",
                    help="per-episode appearance randomisation (distill.Randomizer, as record_vla.py --dr), after reset")
    ap.add_argument("--demo", default="", help="write a captioned demo mp4 of EVERY episode: overhead 640x480 + both "
                                                "wrists, real time (the policy still sees the 320x240 training frames)")
    ap.add_argument("--temporal-ensemble", type=float, default=0.0,
                    help="ACT only: the ACT paper's temporal ensembling (e.g. 0.01). Still action-chunked: a 50-step "
                         "chunk is predicted EVERY tick and the overlapping chunks are averaged (w_i = exp(-k*i), "
                         "oldest first) instead of executing n_action_steps open loop")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s] or [a.seed0 + e for e in range(a.n)]
    a.n = len(seeds)

    import mujoco
    import torch
    torch.set_num_threads(2)
    from PIL import Image
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    import clamp_wide as W
    import design_squeeze as DS
    import record_vla as RV

    # the sim (and its EGL context) FIRST: creating the GL context after CUDA was initialised failed with EGLError
    lvl, yaw, bias = W.SCHEDULE[a.stage]
    if a.pushclamp:
        import push_clamp_wide as P
        import clamp_teacher as C
    env = W.make_env_cls()(seed=a.seed0, level=lvl, yaw_max=yaw,
                           episode_seconds=(P.EPISODE_S if a.pushclamp else None))
    env.set_bias(bias)
    m, d, sim = env.model, env.data, env.sim
    for c in RV.CAMS:
        sim.render_camera(c, native=True)

    # any LeRobot policy (SmolVLA, ACT, ...): the type comes from the checkpoint's own config.json
    ptype = json.loads((Path(a.ckpt).expanduser() / "config.json").read_text())["type"]
    if a.temporal_ensemble:
        if ptype != "act":
            sys.exit("--temporal-ensemble is an ACT option")
        from lerobot.configs.policies import PreTrainedConfig
        cfg = PreTrainedConfig.from_pretrained(a.ckpt)
        cfg.n_action_steps, cfg.temporal_ensemble_coeff = 1, a.temporal_ensemble   # the ensembler is built in __init__
        policy = get_policy_class(ptype).from_pretrained(a.ckpt, config=cfg)
        a.n_action_steps = 1
    else:
        policy = get_policy_class(ptype).from_pretrained(a.ckpt)
        policy.config.n_action_steps = a.n_action_steps
    # CUDA on a GPU box, Apple GPU (mps) on the Mac: LeRobot's config already falls back from the saved "cuda" when it
    # is not available; the saved preprocessor pins "cuda" too, so override its device step to match
    dev = policy.config.device
    policy = policy.to(dev).eval()
    use_task = ptype in ("smolvla", "pi0", "pi05")              # language-conditioned policies get the instruction
    print(f"policy type {ptype}, chunk {getattr(policy.config, 'chunk_size', None)}, "
          + (f"temporal ensembling k={a.temporal_ensemble} (chunk predicted every tick)" if a.temporal_ensemble
             else f"executing {a.n_action_steps} steps per chunk"), flush=True)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=a.ckpt,
                                         preprocessor_overrides={"device_processor": {"device": dev}})
    writer, vid_eps = None, 3
    if a.video:
        import av
        writer = av.open(a.video, "w")
        vs = writer.add_stream("libx264", rate=25)
        vs.width, vs.height, vs.pix_fmt = 320, 240, "yuv420p"
    demo = None
    if a.demo:
        import av
        from PIL import ImageDraw, ImageFont
        demo = av.open(a.demo, "w")
        ds = demo.add_stream("libx264", rate=25)
        ds.width, ds.height, ds.pix_fmt = 960, 540, "yuv420p"
        ds.options = {"crf": "20"}
        fdir = "/usr/share/fonts/truetype/dejavu/"
        try:
            F, FB = ImageFont.truetype(fdir + "DejaVuSans.ttf", 17), ImageFont.truetype(fdir + "DejaVuSans-Bold.ttf", 17)
            FBIG = ImageFont.truetype(fdir + "DejaVuSans-Bold.ttf", 44)
        except OSError:
            F = FB = FBIG = ImageFont.load_default()

        def demo_frame(full, head, line2, banner=None, colour=(62, 207, 142)):
            fr = Image.new("RGB", (960, 540), (15, 17, 21))
            fr.paste(Image.fromarray(full["overhead"]).resize((640, 480)), (0, 60))
            fr.paste(Image.fromarray(full["left_wrist"]).resize((320, 240)), (640, 60))
            fr.paste(Image.fromarray(full["right_wrist"]).resize((320, 240)), (640, 300))
            g = ImageDraw.Draw(fr)
            g.text((12, 8), head, font=FB, fill=(230, 232, 235))
            g.text((12, 33), line2, font=F, fill=(152, 162, 173))
            for txt, y in (("overhead", 64), ):
                g.text((8, y), txt, font=F, fill=(255, 255, 255))
            g.text((646, 64), "left wrist", font=F, fill=(255, 255, 255))
            g.text((646, 304), "right wrist", font=F, fill=(255, 255, 255))
            if banner:
                w = g.textlength(banner, font=FBIG)
                g.rectangle((320 - w / 2 - 18, 260, 320 + w / 2 + 18, 330), fill=(0, 0, 0))
                g.text((320 - w / 2, 268), banner, font=FBIG, fill=colour)
            for p in ds.encode(av.VideoFrame.from_image(fr)):
                demo.mux(p)

    res, all_sq_kept = [], []
    rnd = None
    if a.dr:
        import distill as D
        rnd = D.Randomizer(env)
    for e, seed in enumerate(seeds):
        env.reset(seed=seed)
        x0 = None
        if a.pushclamp:                                 # same start sampling as record_vla.py --pushclamp
            rng = np.random.default_rng(seed)
            x0 = rng.uniform(-0.12, 0.12)
            P._set_start(env, (x0, C.CUBE_Y + rng.uniform(-a.pushclamp, a.pushclamp) / 100))
        if rnd is not None:                             # AFTER reset (reset restores the nominal look)
            rnd.apply(np.random.default_rng([int(seed), 77]))
        policy.reset()
        filt = None
        if a.cmd_filter:
            from vla_safety import CmdFilter
            filt = CmdFilter(env, relax=a.relax, rate_deg=float(np.rad2deg(sim._rate_limit)))
        on_steps, squeezed, lifted, bad, tipped, t0 = 0, False, False, 0, False, time.time()
        first_stop = None              # the REAL rig's safety guard (CLAMP_GUARD=1 [+ CLAMP_GUARD_CMD=1]): first stop
        first_meas, sq_kept = None, []  # first MEASURED-pose stop; filter kept fraction on every squeezing tick
        yaw0 = float(np.degrees(env.base.cube_yaw()))
        task = "push + rotation" if a.pushclamp else "rotation"
        head = (f"{'SmolVLA' if ptype == 'smolvla' else ptype.upper()}{' + safety filter' if a.cmd_filter else ''} · "
                f"{task} · run {e + 1}/{a.n}: cube "
                + (f"{abs(x0) * 100:.1f} cm {'right' if x0 > 0 else 'left'}, " if x0 is not None else "")
                + f"rotated {yaw0:+.0f}°")
        for t in range(env.base.max_steps):
            imgs, full = {}, {}
            for c in RV.CAMS:
                full[c] = sim.render_camera(c, native=True)
                im = np.asarray(Image.fromarray(full[c]).resize((RV.W, RV.H), Image.Resampling.BILINEAR))
                imgs[c] = im
            obs = {f"observation.images.{c}": torch.from_numpy(imgs[c]).permute(2, 0, 1).float() / 255.0 for c in RV.CAMS}
            obs["observation.state"] = torch.from_numpy(RV.state_vec(env)).float()
            if use_task:
                obs["task"] = RV.TASK
            with torch.no_grad():
                act = post(policy.select_action(pre(obs)))
            act = np.asarray(act.squeeze(0).cpu().numpy(), float)          # (12,) degrees / gripper %
            q_prev, q_new = {}, {}                     # commanded poses (5 joints deg + gripper %), before / after
            for i, s in enumerate(("left", "right")):
                arm = sim.robot[s]
                tgt = np.deg2rad(act[i * 6:i * 6 + 5])
                cur = d.ctrl[arm.actuator_ids]
                lo, hi = m.actuator_ctrlrange[arm.gripper_actuator]
                q_prev[s] = np.r_[np.rad2deg(cur), 100.0 * (d.ctrl[arm.gripper_actuator] - lo) / (hi - lo)]
                q_new[s] = np.r_[np.rad2deg(cur + np.clip(tgt - cur, -sim._rate_limit, sim._rate_limit)),
                                 np.clip(act[i * 6 + 5], 0.0, 100.0)]
            kept_now = 1.0
            if filt is not None:                       # vla_safety.CmdFilter: never send a pose the cmd guard rejects
                q_new, kept_now = filt(q_prev, q_new)
            for s in ("left", "right"):
                arm = sim.robot[s]
                lo, hi = m.actuator_ctrlrange[arm.gripper_actuator]
                d.ctrl[arm.actuator_ids] = np.deg2rad(q_new[s][:5])
                d.ctrl[arm.gripper_actuator] = lo + q_new[s][5] / 100.0 * (hi - lo)
            for _ in range(sim._substeps):
                mujoco.mj_step(m, d)
            mujoco.mj_forward(m, d)
            env.base._step += 1
            sq = env.base.squeezed()
            squeezed |= sq
            lifted |= sq and env.base.cube_pos()[2] - 0.016 > 0.01
            bad += bool(DS.bad_contacts(m, d))
            if sq:
                sq_kept.append(kept_now)               # how much of each squeeze command the filter let through
            if getattr(env, "_guard", None) is not None and (first_stop is None or first_meas is None):
                _, stop = env._guard_check()           # measured pose (+ the commanded joint targets with CLAMP_GUARD_CMD)
                if stop and first_stop is None:
                    first_stop = (t + 1, stop)
                meas = [s_ for s_ in stop if "(meas)" in s_]
                if meas and first_meas is None:        # a REAL approach (measured pose), whatever the commands did
                    first_meas = (t + 1, meas)
            on_steps = on_steps + 1 if env.base.on_can() else 0
            if writer is not None and e < vid_eps and t % 2 == 0:
                for p in vs.encode(av.VideoFrame.from_ndarray(imgs["overhead"], format="rgb24")):
                    writer.mux(p)
            if demo is not None:
                phase = ("in the box" if env.base.on_can() else "carrying" if lifted and sq else
                         "squeezing" if sq else "lost grip" if lifted else "pushing / lining up")
                guard_txt = (f" · REAL RIG GUARD WOULD HAVE STOPPED at {first_stop[0] * 0.04:.1f} s"
                             if first_stop else (" · guard: clear" if getattr(env, "_guard", None) is not None else ""))
                demo_frame(full, head, f"t = {(t + 1) * 0.04:5.1f} s · {phase}{guard_txt}")
                last_full = full
            if env.base.can_tipped():
                tipped = True
                break
            if on_steps >= env.base.settle_steps:
                break
        ok = on_steps >= env.base.settle_steps
        # rig_success: delivered with NO guard stop on the way (on the real rig the first stop ends the run)
        res.append(dict(success=ok, clean=ok and bad == 0, squeezed=squeezed, lifted=lifted, tipped=tipped, ticks=t + 1,
                        guard_stop=first_stop is not None, rig_success=ok and first_stop is None,
                        meas_stop=first_meas is not None, sq_kept=sq_kept))
        all_sq_kept.extend(sq_kept)
        print(f"episode {e} (seed {seed}): success {ok} clean {ok and bad == 0} squeezed {squeezed} lifted {lifted} "
              f"tipped {tipped} bad-contact steps {bad} ({t + 1} ticks, {time.time() - t0:.0f}s)"
              + (f" | GUARD STOP at tick {first_stop[0]}: {', '.join(first_stop[1])}" if first_stop else "")
              + (f" | first MEASURED stop at tick {first_meas[0]}: {', '.join(first_meas[1])}" if first_meas else "")
              + (f" | filter cut {filt.n_cut} / held {filt.n_hold} / relaxed {filt.n_relax} of {filt.n} commands"
                 if filt else "")
              + (" (broke: " + ", ".join(f"{k} {v}" for k, v in sorted(getattr(filt, "cut_kinds", {}).items(),
                                                                        key=lambda kv: -kv[1])) + ")"
                 if filt and getattr(filt, "cut_kinds", None) else "")
              + (f" | squeeze kept median {np.median(sq_kept):.2f} p10 {np.percentile(sq_kept, 10):.2f} "
                 f"({len(sq_kept)} ticks)" if filt and sq_kept else ""), flush=True)
        if demo is not None:
            if first_stop:      # the rig would have ended the run there: never show it as a plain success
                ban, col = "GUARD STOP", (240, 82, 79)
                end_txt = (f"{'delivered in sim, but' if ok else 'not delivered;'} the real rig's guard would have "
                           f"stopped it at {first_stop[0] * 0.04:.1f} s ({', '.join(first_stop[1])})")
            else:
                ban, col = (("DELIVERED", (62, 207, 142)) if ok and bad == 0 else
                            ("DELIVERED (touched)", (245, 165, 36)) if ok else ("NOT DELIVERED", (240, 82, 79)))
                end_txt = "finished" + (" · guard clear the whole run" if getattr(env, "_guard", None) is not None else "")
            for _ in range(50):
                demo_frame(last_full, head, f"t = {(t + 1) * 0.04:5.1f} s · {end_txt}", ban, col)
    if demo is not None:
        for p in ds.encode():
            demo.mux(p)
        demo.close()
    if writer is not None:
        for p in vs.encode():
            writer.mux(p)
        writer.close()
    s = {k: float(np.mean([r[k] for r in res])) for k in ("success", "clean", "squeezed", "lifted", "tipped",
                                                           "guard_stop", "rig_success", "meas_stop")}
    if all_sq_kept and a.cmd_filter:
        s.update(squeeze_ticks=len(all_sq_kept), squeeze_kept_median=float(np.median(all_sq_kept)),
                 squeeze_kept_p10=float(np.percentile(all_sq_kept, 10)),
                 squeeze_ticks_cut=float(np.mean(np.array(all_sq_kept) < 1.0)))
    print("SUMMARY", json.dumps({"ckpt": a.ckpt, "stage": a.stage, "pushclamp": a.pushclamp, "n": a.n,
                                 "cmd_filter": bool(a.cmd_filter), "relax": bool(a.relax), "dr": bool(a.dr),
                                 "n_action_steps": a.n_action_steps, "temporal_ensemble": a.temporal_ensemble,
                                 **s}), flush=True)


if __name__ == "__main__":
    main()
