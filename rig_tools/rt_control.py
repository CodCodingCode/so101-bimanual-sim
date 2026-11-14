"""Real-time control helpers shared by every real SO-101 runner (2026-11-13).

The numbers come from so101_control.json (robots/so101/calibration/control/ on the Mac; C:\\robot\\real on the robot PC).
Why this exists: real_chunk_student ran its 25 Hz policy at ~13 Hz with one servo goal per tick; the servos (no
acceleration limit, P 32) snapped to each goal and stopped -> visible shake. See the config's _about notes.

    import rt_control as RT
    cfg = RT.load_config()
    clock = RT.TickClock(cfg)                    # adv = elapsed / 0.04 s, clamped; ema of the tick period
    stream = RT.ServoStreamer(arms, cfg)         # 100 Hz goal writer, linear between policy commands
    stream.start({s: last_goal_raw[s] for s in arms})
    perc = RT.Latest(fn, name="perception")      # fn() runs on its own thread; perc.get() -> (result, seq, t)
    saver = RT.FrameSaver(run / "frames", cfg)   # saver.maybe(t, frames, tag)
    RT.smooth_ramp(arms, {side: raw}, cfg, secs) # scripted moves (start pose, fold to rest): min-jerk at 100 Hz
    ...each tick:
        adv = clock.tick()
        stream.set(side, raw_goal, clock.period())   # or stream.hold(side)
    ...on ANY exit, before anything else writes goals:
        stream.stop(); perc.stop(); saver.stop()
"""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_CANDIDATES = [_HERE / "so101_control.json",                                         # robot PC: next to the runners
               _HERE.parent.parent / "so101" / "calibration" / "control" / "so101_control.json"]   # Mac repo


def load_config(path=None):
    for p in ([Path(path)] if path else _CANDIDATES):
        if p.exists():
            cfg = json.loads(p.read_text())
            cfg["_path"] = str(p)
            return cfg
    raise FileNotFoundError(f"so101_control.json not found in {[str(p) for p in _CANDIDATES]}")


# ------------------------------------------------------------------ time base
class TickClock:
    """Measures real tick periods. tick() -> adv = elapsed / nominal (clamped): how many TRAINING ticks this real
    tick is worth. period() -> EMA of the real tick period (what the servo streamer interpolates over)."""

    def __init__(self, cfg):
        c = cfg["control"]
        self.nominal, self.lo, self.hi = c["nominal_tick_s"], *c["advance_clamp"]
        self.alpha, self.budget = c["dt_ema_alpha"], c["tick_budget_ms"] / 1000
        self.last, self.ema, self.n, self.over = None, self.nominal, 0, 0

    def tick(self):
        now = time.perf_counter()
        if self.last is None:
            self.last = now
            return 1.0
        dt, self.last = now - self.last, now
        self.ema += self.alpha * (dt - self.ema)
        self.n += 1
        self.over += dt > self.budget * 1.1
        return float(np.clip(dt / self.nominal, self.lo, self.hi))

    def period(self):
        return self.ema

    def sleep_rest(self, tick_start):
        """Sleep out the rest of the nominal tick (no-op when already over budget)."""
        time.sleep(max(0.0, self.nominal - (time.perf_counter() - tick_start)))


# ------------------------------------------------------------------ servo goal streaming
class ServoStreamer:
    """Writes each arm's goal at cfg servo_stream.hz, sliding linearly from the current interpolated goal to the
    newest command over `period` seconds. arms: {side: so101_real.Arm} (Bus transactions are locked, so the main
    thread may keep reading positions). dry=True computes everything but writes nothing."""

    def __init__(self, arms, cfg, dry=False):
        s = cfg["servo_stream"]
        self.arms, self.dry = arms, dry
        self.dt, self.pmin, self.pmax = 1.0 / s["hz"], s["min_period_s"], s["max_period_s"]
        self.lock, self.stop_ev = threading.Lock(), threading.Event()
        self.seg, self.sent, self.writes, self.errors, self.thread = {}, {}, 0, [], None

    def start(self, start_raw):
        now = time.perf_counter()
        for side, raw in start_raw.items():
            r = np.asarray(raw, float)
            self.seg[side] = (r, r, now, self.dt)
            self.sent[side] = None
        self.thread = threading.Thread(target=self._run, name="servo-stream", daemon=True)
        self.thread.start()

    def _value(self, side, now):
        a, b, t0, T = self.seg[side]
        return a + (b - a) * min(1.0, (now - t0) / T)

    def set(self, side, raw_goal, period):
        now = time.perf_counter()
        with self.lock:
            cur = self._value(side, now)
            self.seg[side] = (cur, np.asarray(raw_goal, float), now, float(np.clip(period, self.pmin, self.pmax)))

    def hold(self, side):
        now = time.perf_counter()
        with self.lock:
            cur = self._value(side, now)
            self.seg[side] = (cur, cur, now, self.dt)

    def current(self, side):
        with self.lock:
            return self._value(side, time.perf_counter())

    def _run(self):
        nxt = time.perf_counter()
        while not self.stop_ev.is_set():
            now = time.perf_counter()
            with self.lock:
                goals = {s: np.round(self._value(s, now)).astype(int) for s in self.seg}
            for s, g in goals.items():
                if self.sent[s] is not None and np.array_equal(g, self.sent[s]):
                    continue
                if not self.dry:
                    try:
                        self.arms[s].goal(g)
                    except Exception as e:          # keep streaming the other arm; the main loop sees errors
                        self.errors.append(repr(e))
                        continue
                self.sent[s] = g
                self.writes += 1
            nxt += self.dt
            time.sleep(max(0.0, nxt - time.perf_counter()))
            if time.perf_counter() - nxt > 0.1:      # fell far behind (debugger, sleep): resync, don't burst
                nxt = time.perf_counter()

    def stop(self):
        """Stop writing. Call BEFORE any other code writes goals (fold to rest)."""
        self.stop_ev.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


# ------------------------------------------------------------------ scripted moves
def min_jerk(x):
    """0..1 -> 0..1 with zero velocity and acceleration at both ends."""
    x = min(1.0, max(0.0, x))
    return x * x * x * (10 - 15 * x + 6 * x * x)


def smooth_ramp(arms, targets, cfg, secs, check_stop=None, log=None, t_base=None):
    """Move each arm in `targets` ({side: raw goal}) from where it is now along a minimum-jerk S-curve, writing goals
    at cfg servo_stream.hz. Duration = max(secs, largest joint move / cfg ramp.max_deg_s). check_stop() -> True
    raises KeyboardInterrupt("STOP file"). log: list to append {"t", "units", "goal"} samples to (every
    cfg ramp.log_every_s), t relative to t_base (perf_counter)."""
    r, hz = cfg["ramp"], cfg["servo_stream"]["hz"]
    start = {s: np.asarray(arms[s].read_raw(), float) for s in targets}
    goal = {s: np.asarray(targets[s], float) for s in targets}
    move_deg = max(float(np.abs(goal[s] - start[s])[:5].max()) for s in targets) * 360.0 / 4095.0
    T = max(float(secs), move_deg / r["max_deg_s"])
    n = max(1, int(round(T * hz)))
    every = max(1, int(round(r["log_every_s"] * hz)))
    t0 = time.perf_counter()
    t_base = t0 if t_base is None else t_base
    for k in range(1, n + 1):
        if check_stop is not None and check_stop():
            raise KeyboardInterrupt("STOP file")
        f = min_jerk(k / n)
        for s in targets:
            arms[s].goal(np.round(start[s] + (goal[s] - start[s]) * f).astype(int))
        if log is not None and k % every == 0:
            log.append({"t": round(time.perf_counter() - t_base, 3),
                        "units": {s: arms[s].to_units(arms[s].read_raw()).round(2).tolist() for s in targets},
                        "goal": {s: arms[s].to_units(start[s] + (goal[s] - start[s]) * f).round(2).tolist() for s in targets}})
        time.sleep(max(0.0, t0 + k / hz - time.perf_counter()))
    return T


# ------------------------------------------------------------------ background work
class Latest:
    """Runs fn() repeatedly on its own thread; get() returns (result, seq, t_done) of the newest run without
    waiting (result None until the first run finishes). Exceptions are kept in .error and retried."""

    def __init__(self, fn, name="worker", min_period_s=0.0):
        self.fn, self.min_period = fn, min_period_s
        self.res, self.seq, self.t, self.error, self.ms = None, 0, 0.0, None, []
        self.lock, self.stop_ev, self.new = threading.Lock(), threading.Event(), threading.Event()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop_ev.is_set():
            t0 = time.perf_counter()
            try:
                r = self.fn()
                with self.lock:
                    self.res, self.seq, self.t = r, self.seq + 1, time.perf_counter()
                    self.ms = (self.ms + [(self.t - t0) * 1000])[-500:]
                self.new.set()
            except Exception as e:
                self.error = repr(e)
                time.sleep(0.01)
            rest = self.min_period - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)

    def get(self):
        with self.lock:
            return self.res, self.seq, self.t

    def wait_first(self, timeout=5.0):
        if not self.new.wait(timeout):
            raise RuntimeError(f"{self.thread.name}: no result within {timeout} s ({self.error})")

    def stop(self):
        self.stop_ev.set()
        self.thread.join(timeout=1.0)


class FrameSaver:
    """Saves overhead | left wrist | right wrist strips every cfg frames.save_every_s seconds on a thread."""

    def __init__(self, folder, cfg, cams=("overhead", "left_wrist", "right_wrist")):
        f = cfg["frames"]
        self.folder, self.cams = Path(folder), cams
        self.every, self.size, self.q = f["save_every_s"], tuple(f["tile_size"]), f["jpeg_quality"]
        self.folder.mkdir(parents=True, exist_ok=True)
        self.queue, self.next_t = queue.Queue(maxsize=8), 0.0
        self.thread = threading.Thread(target=self._run, name="frame-saver", daemon=True)
        self.thread.start()

    def maybe(self, t, frames, name):
        if t < self.next_t:
            return
        self.next_t = t + self.every
        try:
            self.queue.put_nowait((name, {c: frames[c] for c in self.cams}))
        except queue.Full:
            pass

    def _run(self):
        from PIL import Image
        while True:
            item = self.queue.get()
            if item is None:
                return
            name, frames = item
            try:
                tiles = [np.asarray(Image.fromarray(frames[c]).resize(self.size)) for c in self.cams]
                Image.fromarray(np.concatenate(tiles, 1)).save(self.folder / name, quality=self.q)
            except Exception:
                pass

    def stop(self):
        self.queue.put(None)
        self.thread.join(timeout=3.0)


# ------------------------------------------------------------------ filters
class PeriodicAngleFilter:
    """Angle with a symmetry period (a cube's yaw repeats every 90 deg). A reading more than jump_deg from the
    current estimate is only taken once `confirm` readings in a row agree with each other; the estimate is the
    circular mean of the last `window` accepted readings. update(None) holds."""

    def __init__(self, period_deg=90, window=5, jump_deg=20, confirm=3):
        self.k = 360.0 / period_deg
        self.window, self.jump, self.confirm = window, np.radians(jump_deg) * self.k, confirm
        self.hist, self.pending, self.value = [], [], None

    def _wrap(self, a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def update(self, angle_rad):
        if angle_rad is None:
            return self.value
        a = self._wrap(angle_rad * self.k)
        if self.hist and abs(self._wrap(a - self._mean())) > self.jump:
            self.pending.append(a)
            p = self.pending[-self.confirm:]
            if len(p) < self.confirm or max(abs(self._wrap(x - p[-1])) for x in p) > self.jump:
                return self.value
            self.hist, self.pending = list(p), []
        else:
            self.hist, self.pending = (self.hist + [a])[-self.window:], []
        self.value = self._mean() / self.k
        return self.value

    def _mean(self):
        return float(np.arctan2(np.mean(np.sin(self.hist)), np.mean(np.cos(self.hist))))
