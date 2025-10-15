# Tools/evdev_autotest.py
# Script: map the evdev event codes for the Arduino Board and the Moza Stalks to a json file.

import json
import time
import os
from evdev import InputDevice, list_devices, ecodes
from select import select
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CFG_DIR = os.path.join(ROOT, "configs", "joystick_mappings")
OUT_PATH = os.path.join(CFG_DIR, "input_devices.json")
os.makedirs(CFG_DIR, exist_ok=True)
MOZA_HINTS = ["Gudsen MOZA Multi-function Stalk"]
ARDUINO_HINTS = ["Arduino Leonardo", "Arduino LLC", "Arduino"]

def _read_evkey_until(deadline_s, dev, want_codes=None):
    """Yield EV_KEY (code,value) until deadline."""
    fd = dev.fd
    while True:
        timeout = max(0.0, deadline_s - time.time())
        if timeout == 0.0:
            return
        r, _, _ = select([fd], [], [], timeout)
        if not r:
            return
        for e in dev.read():
            if e.type == ecodes.EV_KEY:
                if want_codes is None or e.code in want_codes:
                    yield e.code, e.value  # value: 1 down, 2 hold, 0 up

def _capture_single_edge(dev, prompt, timeout=8.0):
    """Capture the first KEYDOWN (value 1/2) code after prompt."""
    print(f">>> {prompt} (≤{int(timeout)}s)")
    deadline = time.time() + timeout
    for code, val in _read_evkey_until(deadline, dev):
        if val in (1, 2):
            print(f"  captured code={code}")
            return code
    print("  (timeout)")
    return None

def _capture_pair(dev, prompt_down, timeout=10.0, window_ms=300):
    """
    Capture a 'pair' press (e.g., pull = 293 & 292) and its release.
    Any KEYDOWNs within window_ms are grouped as the DOWN pair; later KEYUPs for
    any of those codes are captured as the UP pair.
    """
    print(f">>> {prompt_down} (≤{int(timeout)}s)")
    deadline = time.time() + timeout
    down_codes = set()

    # 1) wait for first KEYDOWN and collect cluster for window_ms
    first_code = None
    for code, val in _read_evkey_until(deadline, dev):
        if val in (1, 2):
            first_code = code
            down_codes.add(code)
            break
    if first_code is None:
        print("  (timeout on pull-down)")
        return [], []

    cluster_deadline = time.time() + (window_ms / 1000.0)
    for code, val in _read_evkey_until(cluster_deadline, dev):
        if val in (1, 2):
            down_codes.add(code)

    print(f"  down pair: {sorted(down_codes)}")
    return sorted(down_codes)

def _resolve_device(explicit_event=None, needle=None):
    if explicit_event:
        d = InputDevice(explicit_event)
        return explicit_event, (d.name or "")
    if needle:
        for p in list_devices():
            d = InputDevice(p)
            if needle.lower() in (d.name or "").lower():
                return p, (d.name or "")
    return None, None

def _find_device_by_hints(hints):
    for p in list_devices():
        try:
            d = InputDevice(p)
            name = d.name or ""
            if any(h.lower() in name.lower() for h in hints):
                return p, name
        except Exception:
            pass
    return None, None

def _learn_button_code(dev, prompt):
    print(f"\n>>> {prompt} — press once (you’ve got ~10s).")
    start = time.time()
    for e in dev.read_loop():
        if time.time() - start > 10:
            break
        if e.type == ecodes.EV_KEY and e.value in (1, 2):  # key down/hold
            print(f"  captured code={e.code}")
            return e.code
    print("  (timeout, skipped)")
    return None

def _learn_contact_code(dev):
    print("\n>>> Toggle the seatbelt contact a couple times (10s).")
    start = time.time()
    seen = {}
    while time.time() - start < 10:
        for e in dev.read():
            if e.type == ecodes.EV_KEY:
                seen[e.code] = seen.get(e.code, 0) + 1
    if not seen:
        print("  no EV_KEY events observed.")
        return None
    code = max(seen, key=seen.get)
    print(f"  latched contact code={code}")
    return code

def _capture_contact_probe_style(dev, timeout=15.0, verbose=True):
    """
    Probe-style event reader: non-blocking read loop that counts EV_KEY/EV_SW
    events for `timeout` seconds and returns the most active code.
    """
    try:
        dev.set_nonblocking(True)  # behave like evdev_probe's continuous reader
    except Exception:
        pass

    end = time.time() + timeout
    counts = {}
    total_counts = 0
    while time.time() < end:
        try:
            for e in dev.read():
                if e.type in (ecodes.EV_KEY, ecodes.EV_SW):
                    if verbose:
                        print(f"  event: type={e.type} code={e.code} value={e.value}")
                    counts[e.code] = counts.get(e.code, 0) + 1
                    total_counts += 1
                    if total_counts >= 5:
                        print(f"  captured code={e.code} (after {total_counts} events)")
                    elif total_counts>=20:
                        break
        except BlockingIOError:
            pass
        except OSError as ex:
            # EAGAIN on empty read; ignore
            if getattr(ex, "errno", None) not in (11,):
                break
        time.sleep(0.01)

    if not counts:
        return None
    return max(counts, key=counts.get)

def main():
    ap = argparse.ArgumentParser(
        description="Auto-learn evdev mappings for MOZA stalk + Arduino seatbelt"
    )
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--timeout", type=float, default=8.0, help="Seconds per capture step")

    # prefer explicit event paths; otherwise fall back to name needles
    ap.add_argument("--moza-event", help="e.g. /dev/input/event10")
    ap.add_argument("--arduino-event", help="see ./Tools/evdev_probe --needle . e.g. /dev/input/event15")
    ap.add_argument("--moza-needle", default="Gudsen MOZA Multi-function Stalk")
    ap.add_argument("--arduino-needle", default="Arduino LLC Arduino Leonardo")

    args = ap.parse_args()

    # Output path: ./configs/joystick_mappings/input_devices.json
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CFG_DIR = os.path.join(ROOT, "configs", "joystick_mappings")
    OUT_PATH = os.path.join(CFG_DIR, "input_devices.json")
    os.makedirs(CFG_DIR, exist_ok=True)

    result = {"evdev": {}}

    # ---------- MOZA ----------
    moza_path, moza_name = _resolve_device(args.moza_event, args.moza_needle)
    if not moza_path:
        print("[MOZA] Not found. Try --moza-event /dev/input/event10 "
              "or --moza-needle \"Gudsen MOZA Multi-function Stalk\".")
    else:
        print(f"[MOZA] {moza_name} @ {moza_path}")
        try:
            moza_dev = InputDevice(moza_path)
        except PermissionError:
            print(f"[ERROR] Permission denied opening {moza_path}. "
                  f"Add your user to the 'input' group or create a udev rule.")
            moza_dev = None

        if moza_dev:
            t = args.timeout
            left_on  = _capture_single_edge(moza_dev,  "LEFT engage (push left)",        timeout=t)
            left_off = _capture_single_edge(moza_dev,  "LEFT disengage (center return)", timeout=t)
            right_on = _capture_single_edge(moza_dev,  "RIGHT engage (push right)",      timeout=t)
            right_off= _capture_single_edge(moza_dev,  "RIGHT disengage (center return)",timeout=t)
            hazard   = _capture_single_edge(moza_dev,  "PUSH (toggle)",           timeout=t)
            # Pull is a pair (293 & 292). We’ll capture as a cluster; if user times out, fall back to [293,292].
            pull_dn= _capture_pair(moza_dev, "PULL (toggle)", timeout=t)                                             

            result["evdev"]["moza"] = {
                "name": moza_name,
                "event": moza_path,
                "button_map": {
                    "left_on": left_on,
                    "left_off": left_off,            # typically 296
                    "right_on": right_on,
                    "right_off": right_off,          # typically 296
                    "hazard_push": hazard,           # typically 291
                    "blinker_pull": pull_dn or [293, 292],

                }
            }
            try:
                moza_dev.close()
            except Exception:
                pass

    # ---------- Arduino (seatbelt) ----------
    ar_path, ar_name = _resolve_device(args.arduino_event, args.arduino_needle)
    if not ar_path:
        print("[Arduino] Not found. Try --arduino-event /dev/input/event11 or --arduino-needle \"Arduino\".")
    else:
        print(f"[Arduino] {ar_name} @ {ar_path}")
        try:
            ar_dev = InputDevice(ar_path)
        except PermissionError:
            print(f"[ERROR] Permission denied opening {ar_path}. "
                f"Add your user to the 'input' group or create a udev rule.")
            ar_dev = None

        if ar_dev:
            print(">>> Toggle the seatbelt contact a few times (≤10s)")
            contact = _capture_contact_probe_style(ar_dev, timeout=10.0, verbose=args.verbose)
            if not contact:
                print("  (no events seen; using known default code=288)")
                contact = 288

            result.setdefault("evdev", {})["arduino"] = {
                "name": ar_name,
                "event": ar_path,
                "contact_code": contact,
                "invert": False
            }
            try:
                ar_dev.close()
            except Exception:
                pass
    # ---------- Save ----------
    try:
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved mapping → {OUT_PATH}")
    except Exception as e:
        print(f"[ERROR] Could not write {OUT_PATH}: {e}")


if __name__ == "__main__":
    main()