import numpy as np
import joblib
import time, sys
import argparse
from step01 import AXES, WIN, HOP
from mlogger import try_connect, discover_ports, score_port

# ======================================================================================
# 1. Load pipeline back 
# ======================================================================================

loaded_model = joblib.load("random_forest.pkl")
loaded_encoder = joblib.load("label_encoder.pkl")


# ======================================================================================
# 2. Predict on new data 
# ======================================================================================


# ------------------------------ Feature Extraction -------------------------------------
def compute_features(window):
    window = np.array(window)
    d = np.diff(window, axis=0)
    mean = window.mean(0)
    std  = window.std(0)
    rms  = np.sqrt((window**2).mean(0))
    drms = np.sqrt((d**2).mean(0))
    p2p  = (window.max(0) - window.min(0))
    activity = float(np.sqrt((d**2).sum(1)).mean())  # simple activity proxy

    features = [activity]
    features += list(drms)
    features += list(mean)
    features += list(p2p)
    features += list(rms)
    features += list(std)

    return np.array(features).reshape(1,-1)

def main():
    # Buffer to store IMU data
    buffer = []

    ap = argparse.ArgumentParser(description="Platform-independent IMU serial logger")
    ap.add_argument("--port", "-p", help="Serial port (COM5, /dev/cu.usbmodem14101, etc.)")
    ap.add_argument("--baud", "-b", type=int, default=115200, help="Baud rate (default: 115200)")
    ap.add_argument("--duration", type=float, help="Stop after N seconds (default: run until Ctrl-C)")
    ap.add_argument("--no-auto", action="store_true", help="Do NOT auto-pick a port; require --port")
    ap.add_argument("--show-ports", action="store_true", help="List all ports and exit")
    ap.add_argument("--quiet", "-q", action="store_true", help="Less console output")
    args = ap.parse_args()

    if args.show_ports:
        print("[logger] Available serial ports:")
        for p in discover_ports():
            print(f"  {p.device:>20}  |  {p.description}")
        return

    if args.no_auto and not args.port:
        print("[logger] --no-auto requires explicit --port", file=sys.stderr)
        sys.exit(2)

    if not args.quiet:
        print("[logger] Ports (most likely first):")
        ports = discover_ports()
        ports.sort(key=score_port, reverse=True)
        for p in ports:
            print(f"  {p.device:>20}  |  {p.description}")

    ser, current_port = try_connect(args.port, args.baud, auto=not args.no_auto, verbose=not args.quiet)

    start_t = time.time()
    lines_ok = 0
    lines_bad = 0
    last_warn = 0.0

    try:
        while True:
            # Connect/reconnect loop
            if ser is None:
                time.sleep(0.5)
                ser, current_port = try_connect(args.port, args.baud, auto=not args.no_auto, verbose=not args.quiet)
                continue

            try:
                raw = ser.readline()
            except Exception as e:
                if not args.quiet:
                    print(f"\n[logger] Read error on {current_port}: {e}. Reconnecting...", file=sys.stderr)
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                continue

            if not raw:
                # No data for a moment → print gentle hint once in a while
                now = time.time()
                if now - last_warn > 3 and lines_ok == 0 and not args.quiet:
                    print("[logger] No data yet… Check: correct sketch? 115200 baud? Serial Monitor closed?")
                    last_warn = now
                # also check duration if set
                if args.duration and (now - start_t) >= args.duration:
                    break
                continue

            line = raw.decode(errors="ignore").strip()
            parts = line.split(",")
            if len(parts) < len(AXES):
                continue 
            # Convert to float
            values = [float(x) for x in parts[1:len(AXES)+1]] # skip timestamp
            buffer.append(values)

            # Keep only enough for one window
            if len(buffer) >= WIN:
                window_data = buffer[:WIN]
                data = compute_features(window_data)

                # Predict
                preds = loaded_model.predict(data)
                label = loaded_encoder.inverse_transform(preds)[0]
                print(f"Predicted motion: {label}")

                # Slide window
                buffer = buffer[HOP:]

    except KeyboardInterrupt:
        if not args.quiet:
            print("\n[logger] Stopping (Ctrl-C).")
    finally:
        try:
            if ser:
                ser.close()
        except Exception:
            pass
        if not args.quiet:
            print("[logger] Done.")

if __name__ == "__main__":
    main()