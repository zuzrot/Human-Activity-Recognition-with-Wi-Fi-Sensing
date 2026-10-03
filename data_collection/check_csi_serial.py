#!/usr/bin/env python3
import argparse
import time
import serial

p = argparse.ArgumentParser(description="Quick CARE-BED serial/CSI stream check.")
p.add_argument("--port", default="/dev/ttyACM0")
p.add_argument("--baud", type=int, default=921600)
p.add_argument("--seconds", type=int, default=10)
args = p.parse_args()

print(f"Opening {args.port} at {args.baud} bps for {args.seconds} s...")
ser = serial.Serial(args.port, args.baud, timeout=1)
deadline = time.monotonic() + args.seconds
count = 0
try:
    ser.reset_input_buffer()
    while time.monotonic() < deadline:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line:
            count += 1
            if count <= 5:
                print(line[:300])
finally:
    ser.close()

print(f"\nReceived {count} non-empty serial lines.")
if count == 0:
    print("No data received: check USB mapping, port, firmware and TX/RX operation.")
else:
    print("Serial stream is alive. Inspect the first lines above for CSI_DATA.")
