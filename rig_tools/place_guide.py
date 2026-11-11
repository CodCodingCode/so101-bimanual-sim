"""Live placement guide: the real overhead camera with the squeeze policy's training zones drawn on it.

Pulls the overhead MJPEG from the rig monitor (http://127.0.0.1:8104, via the SSH tunnel), projects the training
zones through the CALIBRATED overhead camera (so101_camera_rig.calibrated_full.json: pose, focal, k1/k2) and serves
an annotated stream at http://localhost:8110. It also finds the red cube in the image, back-projects it onto the table
and says whether it is inside the trained zone (and how far to move it).

    python rig_tools/place_guide.py
Zones (world frame: x across the table, +y away from the arms, origin midway between the arms):
    cube centre  x -2..+2 cm, y 0..5 cm      box centre  x -2..+2 cm, y -10..-6 cm
"""
import io
import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
RIG = HERE / "data" / "so101_camera_rig.calibrated_full.json"
MON = "http://127.0.0.1:8104/cam/overhead.mjpg"
CUBE_ZONE = ((-0.025, 0.025), (0.0122, 0.0622))   # clamp task: cube 26 cm out +-2.5 cm
BOX_ZONE = ((-0.02, 0.02), (-0.085, -0.055))      # box between the arms and the cube
CUBE, BOX = 0.032, 0.10
ARM_X, ARM_Y = 0.116, -0.22

cam = json.loads(RIG.read_text())["cameras"]["overhead"]
FX, FY, CX, CY = (cam["intrinsics"][k] for k in ("fx", "fy", "cx", "cy"))
K1, K2 = cam["distortion"][:2]
POS = np.array(cam["pose"]["pos"])
w, x, y, z = cam["pose"]["quat_wxyz"]
RWC = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(pw):
    """World points (N,3) -> pixels (N,2) through the calibrated camera (MuJoCo camera looks along -z, +y up)."""
    pc = (np.asarray(pw, float) - POS) @ RWC
    xc, yc, zc = pc[:, 0], -pc[:, 1], -pc[:, 2]
    xn, yn = xc / zc, yc / zc
    r2 = xn * xn + yn * yn
    dd = 1 + K1 * r2 + K2 * r2 * r2
    return np.c_[FX * xn * dd + CX, FY * yn * dd + CY]


def backproject(u, v, z_plane):
    """Pixel -> world point on the horizontal plane z = z_plane."""
    xd, yd = (u - CX) / FX, (v - CY) / FY
    xn, yn = xd, yd
    for _ in range(20):
        r2 = xn * xn + yn * yn
        dd = 1 + K1 * r2 + K2 * r2 * r2
        xn, yn = xd / dd, yd / dd
    ray_c = np.array([xn, -yn, -1.0])          # back to MuJoCo camera axes
    ray_w = RWC @ ray_c
    t = (z_plane - POS[2]) / ray_w[2]
    return POS + t * ray_w


def rect(x0, x1, y0, y1, zz, n=12):
    xs = np.r_[np.linspace(x0, x1, n), np.full(n, x1), np.linspace(x1, x0, n), np.full(n, x0)]
    ys = np.r_[np.full(n, y0), np.linspace(y0, y1, n), np.full(n, y1), np.linspace(y1, y0, n)]
    return project(np.c_[xs, ys, np.full_like(xs, zz)])


def find_cube(rgb):
    a = rgb.astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = (r > 120) & (r - g > 60) & (r - b > 40)
    if m.sum() < 30:
        return None
    vs, us = np.nonzero(m)
    return float(us.mean()), float(vs.mean()), int(m.sum())


latest = {"jpg": None}


def annotate(jpg):
    img = Image.open(io.BytesIO(jpg)).convert("RGB")
    rgb = np.asarray(img)
    dr = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    (bx0, bx1), (by0, by1) = BOX_ZONE
    bcx, bcy = (bx0 + bx1) / 2, (by0 + by1) / 2
    dr.polygon([tuple(p) for p in rect(bcx - BOX / 2, bcx + BOX / 2, bcy - BOX / 2, bcy + BOX / 2, 0.0)],
               outline=(70, 150, 255, 255), fill=(70, 150, 255, 45), width=3)
    dr.polygon([tuple(p) for p in rect(bx0, bx1, by0, by1, 0.0)], outline=(70, 150, 255, 255), width=2)
    (cx0, cx1), (cy0, cy1) = CUBE_ZONE
    ccx, ccy = (cx0 + cx1) / 2, (cy0 + cy1) / 2
    dr.polygon([tuple(p) for p in rect(cx0, cx1, cy0, cy1, 0.0)], outline=(60, 220, 90, 255), fill=(60, 220, 90, 50), width=3)
    dr.polygon([tuple(p) for p in rect(ccx - CUBE / 2, ccx + CUBE / 2, ccy - CUBE / 2, ccy + CUBE / 2, CUBE)],
               outline=(255, 255, 255, 220), width=2)
    for sx in (-ARM_X, ARM_X):
        u, v = project([[sx, ARM_Y, 0.0]])[0]
        dr.ellipse([u - 6, v - 6, u + 6, v + 6], outline=(255, 210, 0, 255), width=3)
    cl = project(np.c_[np.zeros(20), np.linspace(-0.2, 0.12, 20), np.zeros(20)])
    dr.line([tuple(p) for p in cl], fill=(255, 255, 255, 90), width=1)
    msg, col = "cube: not seen (is it on the table, unblocked?)", (255, 200, 0)
    c = find_cube(rgb)
    if c:
        u, v, n = c
        p = backproject(u, v, CUBE / 2)
        dr.ellipse([u - 8, v - 8, u + 8, v + 8], outline=(255, 0, 255, 255), width=3)
        dx = 0.0 if cx0 <= p[0] <= cx1 else (cx0 - p[0] if p[0] < cx0 else cx1 - p[0])
        dy = 0.0 if cy0 <= p[1] <= cy1 else (cy0 - p[1] if p[1] < cy0 else cy1 - p[1])
        if dx == 0 and dy == 0:
            msg, col = f"cube OK: x {p[0] * 100:+.1f} cm, {(p[1] - ARM_Y) * 100:.1f} cm in front of the bases", (60, 230, 90)
        else:
            parts = []
            if dx:
                parts.append(f"{abs(dx) * 100:.1f} cm {'right' if dx > 0 else 'left'} (camera view)")
            if dy:
                parts.append(f"{abs(dy) * 100:.1f} cm {'away from' if dy > 0 else 'toward'} the arms")
            msg, col = "move cube " + " and ".join(parts), (255, 120, 60)
    dr.rectangle([0, 0, img.width, 44], fill=(0, 0, 0, 170))
    dr.text((8, 4), msg, fill=col, font=font)
    dr.text((8, 24), "green = cube zone (white = ideal cube)   blue = box (outline = ideal, inner = zone for its centre)   "
                     "yellow = arm bases", fill=(230, 230, 230), font=font)
    out = io.BytesIO()
    img.save(out, "JPEG", quality=85)
    return out.getvalue()


def pump():
    while True:
        try:
            with urllib.request.urlopen(f"{MON}?t={time.time()}", timeout=5) as r:
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
                        latest["jpg"] = annotate(buf[s:e + 2])
                        buf = buf[e + 2:]
        except Exception as ex:   # monitor restarting / tunnel blip
            latest["err"] = repr(ex)
            time.sleep(1.0)


PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Squeeze runner</title>
<style>body{background:#111;color:#ddd;font:15px -apple-system,sans-serif;margin:16px}img{max-width:100%;border:1px solid #333}
button{font-size:17px;padding:10px 18px;margin:6px 8px 6px 0;border-radius:6px;border:0;cursor:pointer}
#live{background:#2e9d4f;color:#fff}#dry{background:#3b6fd6;color:#fff}#stop{background:#d23c3c;color:#fff;font-weight:700}
button:disabled{opacity:.4;cursor:default}pre{background:#000;color:#9f9;padding:10px;height:260px;overflow:auto;font-size:13px}
.row{display:flex;gap:16px;flex-wrap:wrap}.col{flex:1 1 560px}</style></head><body>
<h3>Bimanual squeeze: camera policy on the real arms</h3>
<div class="row"><div class="col"><img src="/stream.mjpg"><p>Box on the blue square, cube in the green square
(banner says "cube OK"). The camera view freezes while the robot runs (the runner owns the cameras).</p></div>
<div class="col"><div>
<button id="live" onclick="go('live')">&#9654; Run on robot (trained speed, 25 s)</button>
<button id="dry" onclick="go('dry')">Dry run (no motion)</button>
<button id="stop" onclick="stop()">&#9632; STOP</button></div>
<div id="status">idle</div><pre id="log"></pre></div></div>
<script>
async function go(mode){
  if(mode==='live' && !confirm('Move the real arms now? Keep clear of the arms; STOP aborts and folds them to rest.')) return;
  await fetch('/run/'+mode,{method:'POST'});}
async function stop(){await fetch('/stop',{method:'POST'});}
async function poll(){
  try{const r=await (await fetch('/log')).json();
    document.getElementById('status').textContent=r.running?('RUNNING: '+r.mode):'idle'+(r.last?(' (last: '+r.last+')'):'');
    document.getElementById('live').disabled=r.running;document.getElementById('dry').disabled=r.running;
    const el=document.getElementById('log');const atEnd=el.scrollTop+el.clientHeight>=el.scrollHeight-5;
    el.textContent=r.lines.join('\\n');if(atEnd)el.scrollTop=el.scrollHeight;}catch(e){}
  setTimeout(poll,700);}
poll();
</script></body></html>"""

JOB = {"running": False, "mode": None, "lines": [], "last": None}
REMOTE = (HERE / "remote_run.py").read_text()
PYW = r"C:\Users\dzkch\.venvs\so101-pi\Scripts\python.exe"
MODES = {"dry": ["real_squeeze.py", "--dry", "--seconds", "15"],
         "live": ["real_squeeze.py", "--live", "--speed", "1.0", "--seconds", "25"]}   # trained speed: 0.5 -> 0/20 in sim
NOISE = ("Activate.ps1", "xecution", "CategoryInfo", "FullyQualified", "At C:\\Windows", "fwlink", "env ready")


def run_job(mode):
    import subprocess
    JOB.update(running=True, mode=mode, lines=[f"--- {time.strftime('%H:%M:%S')} {mode} ---"])
    script = REMOTE.replace("__ARGS__", repr(MODES[mode]))
    try:
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=10", "dzkchen", f"{PYW} -u -"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        p.stdin.write(script)
        p.stdin.close()
        for line in p.stdout:
            line = line.rstrip()
            if line and not any(n in line for n in NOISE) and not line.lstrip().startswith("+ "):
                JOB["lines"].append(line)
                JOB["lines"] = JOB["lines"][-400:]
        p.wait()
    except Exception as e:
        JOB["lines"].append(f"runner error: {e!r}")
    JOB.update(running=False, last=f"{mode} finished {time.strftime('%H:%M:%S')}")


def send_stop():
    import subprocess
    subprocess.run(["ssh", "-o", "BatchMode=yes", "dzkchen", f"{PYW} -c \"open(r'C:\\robot\\real\\STOP','w').write('stop')\""],
                   capture_output=True, timeout=20)
    JOB["lines"].append(f"--- STOP sent {time.strftime('%H:%M:%S')} ---")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        if self.path.startswith("/run/"):
            mode = self.path.split("/")[-1]
            if mode in MODES and not JOB["running"]:
                threading.Thread(target=run_job, args=(mode,), daemon=True).start()
                return self._json({"started": mode})
            return self._json({"started": None, "running": JOB["running"]})
        if self.path.startswith("/stop"):
            threading.Thread(target=send_stop, daemon=True).start()
            return self._json({"stop": True})
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/log"):
            return self._json({"running": JOB["running"], "mode": JOB["mode"], "lines": JOB["lines"][-200:],
                               "last": JOB["last"]})
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            try:
                while True:
                    j = latest["jpg"]
                    if j:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.15)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path.startswith("/frame.jpg"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(latest["jpg"] or b"")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(PAGE)


if __name__ == "__main__":
    threading.Thread(target=pump, daemon=True).start()
    print("placement guide: http://localhost:8110", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8110), H).serve_forever()
