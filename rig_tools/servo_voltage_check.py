"""READ-ONLY servo voltage-rating check for both SO-101 arms (run on dzkchen with the monitor stopped).

Sends only Feetech READ (0x02) packets: model number (addr 3, 2 bytes), Max/Min_Voltage_Limit (addr 14/15),
Present_Voltage (addr 62), Present_Temperature (63), Hardware error status (65). Never writes a register and never
touches torque. A 12 V STS3215 reports Max_Voltage_Limit ~140 (14.0 V); a 7.4 V one ~80 (8.0 V).
"""
import serial

PORTS = {"left": "COM5", "right": "COM6"}
NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def read(ser, sid, addr, n):
    body = [sid, 4, 0x02, addr, n]
    pkt = bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])
    ser.reset_input_buffer()
    ser.write(pkt)
    hdr = ser.read(5)
    if len(hdr) < 5 or hdr[0] != 0xFF or hdr[1] != 0xFF or hdr[2] != sid:
        return None, None
    rest = ser.read(hdr[3] - 1)
    if len(rest) != hdr[3] - 1:
        return None, None
    data = rest[:-1]
    if (~sum(hdr[2:] + data)) & 0xFF != rest[-1]:
        return None, None
    return list(data), hdr[4]


for arm, port in PORTS.items():
    print(f"== {arm} ({port})")
    try:
        ser = serial.Serial(port, 1_000_000, timeout=0.05)
    except Exception as e:
        print("  cannot open:", e)
        continue
    with ser:
        for i, name in enumerate(NAMES, start=1):
            model, err = read(ser, i, 3, 2)
            lim, _ = read(ser, i, 14, 2)
            vt, _ = read(ser, i, 62, 2)
            hw, _ = read(ser, i, 65, 1)
            if model is None or lim is None or vt is None:
                print(f"  id {i} {name:13s} no reply")
                continue
            print(f"  id {i} {name:13s} model {model[0] | model[1] << 8:5d}  max_V {lim[0] / 10:4.1f}  min_V {lim[1] / 10:4.1f}"
                  f"  now {vt[0] / 10:4.1f} V  {vt[1]:3d} C  status_err 0x{err:02x}  hw_err {hw[0] if hw else '?'}")
