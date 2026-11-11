"""Robot station UI at http://localhost:8110: all three cameras, a live 3-D twin of where both arms are, joint angles,
and buttons to run the camera policy, dry-run it, STOP, or bring the arms back to rest.

Feeds come from the rig monitor (http://127.0.0.1:8104 over the SSH tunnel). While a job runs on the robot the monitor
is stopped (the runner owns the cameras and COM ports), so the feeds pause and the page says so.

    ~/.pyenv/versions/3.10.9/bin/python rig_tools/rig_ui.py
Buttons:
    Run on robot      camera policy, trained speed, 25 s, then fold to rest + torque off
    Dry run           camera policy with torque off (prints what it would do)
    STOP              during a run: end the policy, fold to rest, torque off.  When idle: bring the arms to rest.
    Bring arms to rest   go_rest.py from any pose: raise, upright, fold, torque off
"""
import io
import json
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import place_guide as G      # noqa: E402  (projection + zone annotation)

MONITOR = "http://127.0.0.1:8104"
CAMS = ("overhead", "left_wrist", "right_wrist")
PYW = r"C:\Users\dzkch\.venvs\so101-pi\Scripts\python.exe"
REMOTE = (HERE / "remote_run.py").read_text()
MODES = {"rest": ["go_rest.py"],
         "clamp_dry": ["real_clamp.py", "--dry"],
         "clamp": ["real_clamp.py", "--live"]}
NOISE = ("Activate.ps1", "xecution", "CategoryInfo", "FullyQualified", "At C:\\Windows", "fwlink", "env ready")
JOINTS = ("pan", "lift", "elbow", "wrist flex", "wrist roll", "gripper")

FRAMES = {c: None for c in CAMS}
FRAMES["overhead_annot"] = None
STATE = {"data": None, "t": 0.0}
TWIN = {"jpg": None}
JOB = {"running": False, "mode": None, "lines": [], "last": None}


# ------------------------------------------------------------------ feeds
def pump(cam):
    while True:
        try:
            with urllib.request.urlopen(f"{MONITOR}/cam/{cam}.mjpg?t={time.time()}", timeout=5) as r:
                buf = b""
                while True:
                    chunk = r.read(32768)
                    if not chunk:
                        break
                    buf += chunk
                    while True:
                        s = buf.find(b"\xff\xd8")
                        e = buf.find(b"\xff\xd9", s + 2) if s >= 0 else -1
                        if e < 0:
                            break
                        jpg = buf[s:e + 2]
                        buf = buf[e + 2:]
                        FRAMES[cam] = jpg
                        if cam == "overhead":
                            try:
                                FRAMES["overhead_annot"] = G.annotate(jpg)
                            except Exception:
                                FRAMES["overhead_annot"] = jpg
        except Exception:
            time.sleep(1.0)


def poll_state():
    while True:
        try:
            STATE["data"] = json.load(urllib.request.urlopen(f"{MONITOR}/api/state", timeout=2))
            STATE["t"] = time.time()
        except Exception:
            pass
        time.sleep(0.2)


def twin():
    """Render the arms at their live joint angles in the calibrated sim (232 mm mount, real joint offsets)."""
    sys.path.insert(0, str(HERE.parent / "pi_sim_so101"))
    import mujoco
    from pi_sim.env import RobotEnv
    env = RobotEnv(task="pick_place", control_mode="joint", camera_rig=str(G.RIG))
    env.reset(seed=0, options={"randomize": False})
    m, d = env.model, env.data
    d.joint("cube_joint").qpos[:3] = [2, 2, -1]
    d.mocap_pos[0] = [2, 2, -1]
    glo, ghi = m.actuator_ctrlrange[env.robot["left"].gripper_actuator]
    r = mujoco.Renderer(m, 390, 520)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, -0.12, 0.12]
    cam.distance, cam.azimuth, cam.elevation = 0.75, -60, -28
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    while True:
        st = STATE["data"]
        if st and all(st["arms"][s].get("state") for s in ("left", "right")):
          try:
            for s in ("left", "right"):
                u = np.array(st["arms"][s]["state"]["units"], float)
                d.qpos[env.robot[s].qpos_ids] = np.deg2rad(u[:5])
                d.qpos[m.joint(f"{s}_gripper").qposadr[0]] = glo + np.clip(u[5], 0, 100) / 100 * (ghi - glo)
            mujoco.mj_forward(m, d)
            r.update_scene(d, cam)
            img = Image.fromarray(r.render())
            dr = ImageDraw.Draw(img, "RGBA")
            stale = time.time() - STATE["t"] > 2
            tq = {s: any(st["arms"][s]["state"]["torque"]) for s in ("left", "right")}
            dr.rectangle([0, 0, 520, 24], fill=(0, 0, 0, 160))
            dr.text((8, 4), ("STALE (robot busy) | " if stale else "live | ") +
                    "  ".join(f"{s} torque {'ON' if tq[s] else 'off'}" for s in ("left", "right")),
                    fill=(255, 170, 60) if stale or any(tq.values()) else (120, 230, 120), font=font)
            out = io.BytesIO()
            img.save(out, "JPEG", quality=85)
            TWIN["jpg"] = out.getvalue()
          except Exception as e:      # never let one bad state packet kill the twin view
            print("twin render skipped:", repr(e), flush=True)
        time.sleep(0.25)


# ------------------------------------------------------------------ robot jobs
def run_job(mode):
    JOB.update(running=True, mode=mode, lines=JOB["lines"][-60:] + [f"--- {time.strftime('%H:%M:%S')} {mode} ---"])
    script = REMOTE.replace("__ARGS__", repr(MODES[mode]))
    try:
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=10", "dzkchen", f"{PYW} -u -"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        p.stdin.write(script)
        p.stdin.close()
        for line in p.stdout:
            line = line.rstrip()
            if line and not any(n in line for n in NOISE) and not line.lstrip().startswith("+ "):
                JOB["lines"] = (JOB["lines"] + [line])[-400:]
        p.wait()
    except Exception as e:
        JOB["lines"].append(f"runner error: {e!r}")
    JOB.update(running=False, last=f"{mode} finished {time.strftime('%H:%M:%S')}")


def start(mode):
    if JOB["running"]:
        return False
    threading.Thread(target=run_job, args=(mode,), daemon=True).start()
    return True


def stop():
    if JOB["running"] and JOB["mode"] in ("clamp", "clamp_dry"):
        subprocess.run(["ssh", "-o", "BatchMode=yes", "dzkchen",
                        f"{PYW} -c \"open(r'C:\\robot\\real\\STOP','w').write('stop')\""], capture_output=True, timeout=20)
        JOB["lines"].append(f"--- STOP {time.strftime('%H:%M:%S')}: ending the policy, folding back to rest ---")
    elif not JOB["running"]:
        JOB["lines"].append(f"--- STOP while idle {time.strftime('%H:%M:%S')}: bringing the arms to rest ---")
        start("rest")


# ------------------------------------------------------------------ web
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SO-101 station</title>
<style>
body{background:#101113;color:#ddd;font:14px -apple-system,sans-serif;margin:14px}
h3{margin:4px 0 10px}button{font-size:16px;padding:9px 16px;margin:0 8px 8px 0;border-radius:6px;border:0;cursor:pointer;color:#fff}
#clamp{background:#2e9d4f}#clamp_dry{background:#3b6fd6}#live{background:#555}#dry{background:#3b6fd6}#rest{background:#6b6f7a}#stop{background:#d23c3c;font-weight:700;font-size:18px}
button:disabled{opacity:.35;cursor:default}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:10px}
.tile{background:#181a1d;border:1px solid #2a2d31;border-radius:6px;padding:6px}.tile img{width:100%;display:block;border-radius:4px}
.cap{color:#9aa;font-size:12px;margin:3px 2px}
table{border-collapse:collapse;width:100%}td,th{padding:3px 6px;text-align:right;font-variant-numeric:tabular-nums}th{color:#9aa;font-weight:500}
td:first-child,th:first-child{text-align:left}.on{color:#ff8a4c;font-weight:700}.off{color:#7fd17f}
pre{background:#000;color:#9f9;padding:8px;height:220px;overflow:auto;font-size:12px;margin:0}
#busy{display:none;color:#ffb347;font-weight:700;margin-left:6px}
</style></head><body>
<h3>SO-101 station: squeeze policy</h3>
<div><button id="clamp" onclick="go('clamp')">&#9654; Clamp test on robot (scripted)</button>
<button id="clamp_dry" onclick="go('clamp_dry')">Clamp: dry check</button>

<button id="rest" onclick="go('rest')">Bring arms to rest</button>
<button id="stop" onclick="stp()">&#9632; STOP</button>
<span id="status">idle</span><span id="busy">robot busy: camera feeds paused</span></div>
<div class="grid">
 <div class="tile"><img id="c_overhead" src="/cam/overhead_annot.mjpg"><div class="cap">Overhead: cube in the green square, box on the blue square (between the arms and the cube)</div></div>
 <div class="tile"><img src="/twin.mjpg"><div class="cap">Twin: both arms at their live joint angles (calibrated sim)</div></div>
 <div class="tile"><img src="/cam/left_wrist.mjpg"><div class="cap">Left wrist camera</div></div>
 <div class="tile"><img src="/cam/right_wrist.mjpg"><div class="cap">Right wrist camera</div></div>
 <div class="tile"><table id="arms"></table><div class="cap" id="age"></div></div>
 <div class="tile"><pre id="log"></pre></div>
</div>
<script>
const J=__JOINTS__;
async function go(m){ if((m==='live'||m==='clamp')&&!confirm('Move the real arms now? Stay clear. STOP ends the run and folds the arms back to rest.'))return;
 await fetch('/run/'+m,{method:'POST'});}
async function stp(){await fetch('/stop',{method:'POST'});}
function arms(st){ if(!st) return '<tr><td>no arm data</td></tr>';
 let h='<tr><th>joint</th><th>left</th><th>right</th></tr>';
 for(let i=0;i<6;i++) h+='<tr><td>'+J[i]+'</td><td>'+st.arms.left.state.units[i].toFixed(1)+'</td><td>'+st.arms.right.state.units[i].toFixed(1)+'</td></tr>';
 const tl=st.arms.left.state.torque.some(x=>x), tr=st.arms.right.state.torque.some(x=>x);
 h+='<tr><td>torque</td><td class="'+(tl?'on':'off')+'">'+(tl?'ON':'off')+'</td><td class="'+(tr?'on':'off')+'">'+(tr?'ON':'off')+'</td></tr>';
 return h;}
async function poll(){ try{
 const r=await (await fetch('/log')).json(); const s=await (await fetch('/state.json')).json();
 document.getElementById('status').textContent=r.running?('RUNNING: '+r.mode):('idle'+(r.last?' (last: '+r.last+')':''));
 document.getElementById('busy').style.display=r.running?'inline':'none';
 for(const id of ['clamp','clamp_dry','rest']) document.getElementById(id).disabled=r.running;
 const el=document.getElementById('log'); const end=el.scrollTop+el.clientHeight>=el.scrollHeight-5;
 el.textContent=r.lines.join('\\n'); if(end) el.scrollTop=el.scrollHeight;
 document.getElementById('arms').innerHTML=arms(s.data);
 document.getElementById('age').textContent=s.age<2?'joint angles (deg; gripper %): live':'joint angles: last known ('+Math.round(s.age)+' s ago)';
 }catch(e){} setTimeout(poll,500);}
poll();
</script></body></html>""".replace("__JOINTS__", json.dumps(JOINTS))


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _mjpeg(self, getter):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
        self.end_headers()
        try:
            while True:
                j = getter()
                if j:
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                time.sleep(0.12)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_GET(self):
        p = self.path.split("?")[0]
        if p.startswith("/cam/") and p.endswith(".mjpg"):
            name = p[5:-5]
            if name in FRAMES:
                return self._mjpeg(lambda: FRAMES[name])
        if p == "/twin.mjpg":
            return self._mjpeg(lambda: TWIN["jpg"])
        if p == "/state.json":
            return self._send(json.dumps({"data": STATE["data"], "age": time.time() - STATE["t"]}).encode(), "application/json")
        if p == "/log":
            return self._send(json.dumps({"running": JOB["running"], "mode": JOB["mode"], "lines": JOB["lines"][-200:],
                                          "last": JOB["last"]}).encode(), "application/json")
        if p == "/frame.jpg":
            return self._send(FRAMES["overhead_annot"] or b"", "image/jpeg")
        return self._send(PAGE.encode(), "text/html")

    def do_POST(self):
        if self.path.startswith("/run/"):
            mode = self.path.split("/")[-1]
            ok = mode in MODES and start(mode)
            return self._send(json.dumps({"started": mode if ok else None}).encode(), "application/json")
        if self.path.startswith("/stop"):
            threading.Thread(target=stop, daemon=True).start()
            return self._send(b'{"stop": true}', "application/json")
        self.send_response(404)
        self.end_headers()


if __name__ == "__main__":
    for c in CAMS:
        threading.Thread(target=pump, args=(c,), daemon=True).start()
    threading.Thread(target=poll_state, daemon=True).start()
    threading.Thread(target=twin, daemon=True).start()
    print("SO-101 station: http://localhost:8110", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8110), H).serve_forever()
