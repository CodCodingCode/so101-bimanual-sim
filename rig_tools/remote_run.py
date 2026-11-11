"""Run ON dzkchen (piped over ssh): stop monitor.py, run one command in C:\\robot\\real streaming its output, then
always restart monitor.py detached. The command is given in ARGS below (place_guide.py fills it in)."""
import os
import signal
import socket
import subprocess
import sys
import time

PY = r"C:\Users\dzkch\.venvs\so101-pi\Scripts\python.exe"
HERE = r"C:\robot\real"
ARGS = __ARGS__


def listening(port):
    s = socket.socket()
    s.settimeout(0.5)
    ok = s.connect_ex(("127.0.0.1", port)) == 0
    s.close()
    return ok


pids = {l.split()[-1] for l in subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout.splitlines()
        if ("127.0.0.1:8104 " in l or "127.0.0.1:8105 " in l) and "LISTENING" in l}
for p in pids:
    try:
        os.kill(int(p), signal.SIGTERM)
    except Exception as e:
        print("could not stop monitor", p, e, flush=True)
time.sleep(2.5)
print("monitor stopped" if not listening(8104) else "WARNING: monitor still running", flush=True)
try:
    p = subprocess.Popen([PY, "-u", *ARGS], cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    print("exit", p.wait(), flush=True)
finally:
    log = open(os.path.join(HERE, "monitor_restart.log"), "w")
    subprocess.Popen([PY, "-u", "monitor.py"], cwd=HERE, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                     creationflags=0x00000008 | 0x00000200 | 0x01000000, close_fds=True)
    for _ in range(60):
        if listening(8104):
            break
        time.sleep(0.5)
    print("monitor back" if listening(8104) else "monitor NOT back", flush=True)
