#!/usr/bin/env python3
"""Export the squeeze training logs to TensorBoard (they are written as CSV / text, not TB events).

    python squeeze/tb_export.py                 # once
    python squeeze/tb_export.py --watch 60      # refresh every 60 s
    tensorboard --logdir squeeze/runs_tb

Runs exported: PPO progress.csv of the current run (box, plain friction) and earlier runs kept for comparison
(box with noslip, can), plus the camera student's DAgger rounds and training loss.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TB = HERE / "runs_tb"
PPO_RUNS = {"CLAMP_RECOVERY v3 (current: + box-tip penalty)": HERE / "runs_clamp_recovery" / "ppo",
            "CLAMP_RECOVERY v2 (stopped: tipped the box)": HERE / "runs_clamp_recovery" / "ppo_v2_tipped",
            "CLAMP_into_box (base, 100% clean)": HERE / "runs_clamp" / "ppo",
            }
STUDENT_LOG = HERE / "runs_clamp_recovery" / "student" / "distill_log.txt"


def fresh_writer(name):
    from torch.utils.tensorboard import SummaryWriter
    d = TB / name
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("events.out.tfevents*"):
        os.remove(f)
    return SummaryWriter(str(d))


def export():
    n = 0
    for name, d in PPO_RUNS.items():
        f = d / "progress.csv"
        if not f.exists():
            continue
        w = fresh_writer(name)
        for row in csv.DictReader(open(f)):
            step = int(float(row["steps"]))
            for k in ("is_success", "placed", "cleared", "lifted", "squeezed", "can_tipped"):
                if row.get(k):
                    w.add_scalar(f"ppo_train_stochastic/{k}", float(row[k]), step)
            if row.get("ep_rew_mean"):
                w.add_scalar("ppo/episode_reward", float(row["ep_rew_mean"]), step)
            if row.get("bc_weight"):
                w.add_scalar("ppo/bc_weight", float(row["bc_weight"]), step)
            if row.get("greedy_success"):
                w.add_scalar("ppo_greedy_deterministic/is_success", float(row["greedy_success"]), step)
            n += 1
        w.close()
    if STUDENT_LOG.exists():
        w = fresh_writer("camera_student")
        rnd, ep = 0, 0
        for line in open(STUDENT_LOG):
            m = re.match(r"\[(\w+)\] (\d+) episodes, beta ([\d.]+).*?: (.*)", line)
            if m:
                rates = dict(kv.split("=") for kv in m.group(4).split())
                for k, v in rates.items():
                    w.add_scalar(f"student_rounds/{k}", float(v), rnd)
                w.add_scalar("student_rounds/teacher_share_beta", float(m.group(3)), rnd)
                w.add_text("student_rounds/round", f"{rnd}: {m.group(1)}", rnd)
                rnd += 1
            m = re.match(r"\s+train epoch \d+: mse ([\d.]+) \((\d+) frames", line)
            if m:
                w.add_scalar("student_train/action_mse", float(m.group(1)), ep)
                w.add_scalar("student_train/frames", int(m.group(2)), ep)
                ep += 1
        w.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0)
    a = ap.parse_args()
    while True:
        rows = export()
        print(time.strftime("%H:%M:%S"), f"exported {rows} PPO rows + student log", flush=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == "__main__":
    main()
