#!/usr/bin/env python3

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import serial


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_PORT = "COM9"
DEFAULT_BAUD = 921600
DEFAULT_MAC = "24:0A:C4:02:29:F0"


# ============================================================
# HELPERS
# ============================================================

def safe_name(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_-]+", "_", text)
    return text.strip("_")


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect one independent CARE-BED CSI recording session."
    )

    p.add_argument(
        "--activity",
        required=True,
        help="walking, lying_down, sitting_up, fidgeting, inactivity"
    )

    p.add_argument(
        "--session",
        required=True,
        type=int,
        help="session number, e.g. 1"
    )

    p.add_argument(
        "--environment",
        default="envB",
        help="environment identifier, default: envB"
    )

    p.add_argument(
        "--duration",
        type=float,
        default=180.0,
        help="recording duration in seconds"
    )

    p.add_argument(
        "--countdown",
        type=int,
        default=5,
        help="countdown before recording"
    )

    p.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"serial port, default: {DEFAULT_PORT}"
    )

    p.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"baud rate, default: {DEFAULT_BAUD}"
    )

    p.add_argument(
        "--mac",
        default=DEFAULT_MAC,
        help=f"expected AP/TX MAC, default: {DEFAULT_MAC}"
    )

    p.add_argument(
        "--output-dir",
        default="carebed_new_data",
        help="root output directory"
    )

    return p.parse_args()


# ============================================================
# CSI VALIDATION
# ============================================================

def validate_csi_line(line: str, expected_mac: str):
    """
    Validate one ESP32 CSI line.

    Current ACTIVE_AP firmware may return a CSI payload longer than
    128 values (e.g. 384 values). We preserve the complete raw CSI
    frame in the CSV.

    During later preprocessing/training, the first 128 CSI values
    can be selected to reproduce the original CARE-BED
    64-subcarrier representation.
    """

    # Exactly one CSI payload [...]
    if line.count("[") != 1 or line.count("]") != 1:
        return None

    parts = line.split(",")

    # Expected ESP32 CSI Tool format
    if len(parts) != 26:
        return None

    # Only CSI frames collected by AP from the intended transmitter
    if parts[0].strip() != "CSI_DATA":
        return None

    if parts[1].strip() != "AP":
        return None

    if parts[2].strip().upper() != expected_mac.strip().upper():
        return None

    try:
        payload = re.findall(r"\[(.*)\]", line)[0]
        values = [int(x) for x in payload.split()]

        # Field 24 is the CSI payload length reported by ESP32
        declared_len = int(parts[24].strip())

    except (IndexError, ValueError):
        return None

    # CARE-BED needs at least the first 128 I/Q values
    if len(values) < 128:
        return None

    # Check that ESP32-reported length matches received payload
    if declared_len != len(values):
        return None

    return [field.strip() for field in parts]


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    activity = safe_name(args.activity)
    environment = safe_name(args.environment)

    session_id = f"{environment}_{activity}_s{args.session:02d}"

    # Each activity gets its own directory
    out_dir = Path(args.output_dir) / activity
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{session_id}.csv"
    timing_path = out_dir / f"{session_id}_timing.csv"
    meta_path = out_dir / f"{session_id}_meta.json"

    # Never append to or overwrite an existing recording
    existing = [
        p for p in (raw_path, timing_path, meta_path)
        if p.exists()
    ]

    if existing:
        print("ERROR: session files already exist:")

        for p in existing:
            print(f"  {p}")

        print("Use a new session number. Nothing was overwritten or appended.")
        return 2

    print("CARE-BED independent-session collector")
    print(f"Session:      {session_id}")
    print(f"Activity:     {activity}")
    print(f"Environment:  {environment}")
    print(f"Port:         {args.port}")
    print(f"Baud:         {args.baud}")
    print(f"Expected MAC: {args.mac}")
    print(f"Duration:     {args.duration:.1f} s")
    print(f"Raw CSV:      {raw_path}")
    print(f"Timing CSV:   {timing_path}")
    print()

    # --------------------------------------------------------
    # Open serial port
    # --------------------------------------------------------

    try:
        ser = serial.Serial(
            args.port,
            args.baud,
            timeout=1
        )

    except serial.SerialException as e:
        print(f"ERROR: cannot open {args.port}: {e}")
        return 1

    accepted = 0
    rejected = 0

    started_wall = None
    started_monotonic = None

    try:
        print("Serial port opened.")
        print("Check the setup and move to the starting position.")

        # Countdown before actual recording
        for left in range(args.countdown, 0, -1):
            print(f"Recording starts in {left}...")
            time.sleep(1)

        # Discard CSI accumulated during setup/countdown
        ser.reset_input_buffer()

        started_wall = datetime.now().astimezone()
        started_monotonic = time.monotonic()

        print("=== START RECORDING ===")

        # "x" = create new file and fail if it already exists
        with raw_path.open(
            "x",
            newline="",
            encoding="utf-8"
        ) as raw_f, timing_path.open(
            "x",
            newline="",
            encoding="utf-8"
        ) as timing_f:

            raw_writer = csv.writer(raw_f)
            timing_writer = csv.writer(timing_f)

            timing_writer.writerow([
                "frame_index",
                "host_timestamp_iso",
                "host_time_ns",
                "elapsed_s"
            ])

            while True:
                elapsed = time.monotonic() - started_monotonic

                if elapsed >= args.duration:
                    break

                try:
                    line = (
                        ser.readline()
                        .decode("utf-8", errors="ignore")
                        .strip()
                    )

                except serial.SerialException as e:
                    print(f"\nSerial error: {e}")
                    break

                if not line:
                    continue

                parsed = validate_csi_line(
                    line,
                    args.mac
                )

                if parsed is None:
                    rejected += 1
                    continue

                now_wall = datetime.now().astimezone()
                now_ns = time.time_ns()

                elapsed = time.monotonic() - started_monotonic

                # Save complete raw CSI frame
                raw_writer.writerow(parsed)

                # Save independent host-side timing information
                timing_writer.writerow([
                    accepted,
                    now_wall.isoformat(timespec="microseconds"),
                    now_ns,
                    f"{elapsed:.6f}"
                ])

                accepted += 1

                # Status every 50 accepted frames
                if accepted % 50 == 0:
                    print(
                        f"\rAccepted frames: {accepted:5d} | "
                        f"elapsed: {elapsed:6.1f}/{args.duration:.1f} s",
                        end="",
                        flush=True
                    )

                    raw_f.flush()
                    timing_f.flush()

        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        finished_wall = datetime.now().astimezone()

        actual_duration = (
            time.monotonic() - started_monotonic
        )

        fps = (
            accepted / actual_duration
            if actual_duration > 0
            else None
        )

        metadata = {
            "session_id": session_id,
            "environment": environment,
            "activity": activity,
            "session_number": args.session,

            "serial_port": args.port,
            "baud_rate": args.baud,
            "expected_ap_mac": args.mac,

            "requested_duration_s": args.duration,
            "actual_duration_s": actual_duration,

            "accepted_frames": accepted,
            "rejected_lines": rejected,

            "estimated_receiver_frame_rate_hz": fps,

            "recording_start":
                started_wall.isoformat(timespec="microseconds"),

            "recording_end":
                finished_wall.isoformat(timespec="microseconds"),

            "raw_csv": str(raw_path),
            "timing_csv": str(timing_path),

            "csi_storage": "full_raw_payload",
            "carebed_preprocessing_expected_values": 128
        }

        with meta_path.open(
            "x",
            encoding="utf-8"
        ) as meta_f:

            json.dump(
                metadata,
                meta_f,
                indent=2,
                ensure_ascii=False
            )

        print("\n=== STOP RECORDING ===")
        print(f"Accepted frames: {accepted}")
        print(f"Rejected lines:  {rejected}")
        print(f"Actual duration: {actual_duration:.2f} s")

        if fps is not None:
            print(
                f"Estimated RX frame rate: "
                f"{fps:.3f} Hz"
            )

        print(f"Saved: {raw_path}")
        print(f"Saved: {timing_path}")
        print(f"Saved: {meta_path}")

        if accepted < 20:
            print(
                "WARNING: fewer than 20 accepted frames; "
                "this session is too short for a 20-frame segment."
            )

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user. "
            "The current session is incomplete; "
            "do not use it as a normal session."
        )
        return 130

    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())