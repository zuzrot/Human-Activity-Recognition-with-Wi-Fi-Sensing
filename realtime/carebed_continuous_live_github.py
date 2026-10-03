#!/usr/bin/env python3
"""
CARE-BED — continuous live activity recognition.

This script is intended for real-time deployment with an ESP32-based
CSI acquisition setup. It continuously:

    serial CSI stream
    -> validates CSI frames
    -> converts the first 128 I/Q values into 64 amplitudes
    -> buffers 20 new valid CSI frames
    -> applies Hampel filtering and Savitzky-Golay smoothing
    -> removes 10 unreliable subcarriers
    -> applies the saved StandardScaler
    -> runs the trained BiLSTM classifier
    -> prints the predicted activity and confidence
    -> repeats with the next non-overlapping 20-frame segment

Required deployment artifacts
-----------------------------
Place the following files in the same directory as this script:

    carebed_W20_HSG_single_split_live_model.keras
    carebed_W20_HSG_single_split_scaler.pkl
    carebed_W20_HSG_single_split_label_encoder.pkl
    carebed_W20_HSG_single_split_metadata.json

Install dependencies
--------------------
    pip install pyserial numpy pandas scipy hampel tensorflow joblib

Hardware configuration
----------------------
Before running the script, provide the serial port and the MAC address used
by your own ESP32 setup.

You can either:
    1. edit DEFAULT_PORT and DEFAULT_MAC below, or
    2. provide them from the command line with --port and --mac.

Example:
    python carebed_continuous_live.py --port COM9 --mac 24:0A:C4:02:29:F0

Press Ctrl+C to stop.
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import serial
import tensorflow as tf
from hampel import hampel
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------
# User hardware configuration
# ---------------------------------------------------------------------

# Set these values according to your own hardware configuration.
#
# Example for Windows:
# DEFAULT_PORT = "COM9"
#
# Example ESP32 MAC address:
# DEFAULT_MAC = "24:0A:C4:02:29:F0"

DEFAULT_PORT = None
DEFAULT_MAC = None

DEFAULT_BAUD = 921600
EXPECTED_ROLE = "AP"


# ---------------------------------------------------------------------
# Deployment artifacts
# ---------------------------------------------------------------------

MODEL_FILE = "carebed_W20_HSG_single_split_live_model.keras"
SCALER_FILE = "carebed_W20_HSG_single_split_scaler.pkl"
ENCODER_FILE = "carebed_W20_HSG_single_split_label_encoder.pkl"
METADATA_FILE = "carebed_W20_HSG_single_split_metadata.json"

LOG_FILE = "carebed_live_predictions.csv"


# ---------------------------------------------------------------------
# Final CARE-BED preprocessing configuration
# ---------------------------------------------------------------------

SEGMENT_LENGTH = 20
FIRST_CSI_VALUES = 128
RAW_FEATURES = 64

COLUMNS_TO_DROP = [2, 3, 4, 5, 32, 59, 60, 61, 62, 63]
N_FEATURES = 54

HAMPEL_WINDOW = 10
SAVGOL_WINDOW = 10
SAVGOL_POLYORDER = 3

NO_DATA_WARNING_SECONDS = 10.0


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="CARE-BED continuous live activity recognition"
    )

    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=(
            "Serial port connected to the ESP32 receiver, e.g. COM9 "
            "on Windows or /dev/ttyUSB0 on Linux."
        ),
    )

    parser.add_argument(
        "--mac",
        default=DEFAULT_MAC,
        help=(
            "Expected MAC address reported in the CSI_DATA stream. "
            "Set this according to your ESP32 setup."
        ),
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Serial baud rate (default: {DEFAULT_BAUD}).",
    )

    parser.add_argument(
        "--log",
        default=LOG_FILE,
        help=(
            f"CSV prediction log (default: {LOG_FILE}). "
            "Use an empty string to disable logging."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
# Label / artifact handling
# ---------------------------------------------------------------------

def canonical_label(value):
    return str(value).strip().lower().replace("_", " ")


def validate_hardware_config(port, mac):
    if not port:
        raise SystemExit(
            "Serial port is not configured.\n"
            "Either edit DEFAULT_PORT in the script or run, for example:\n"
            "  python carebed_continuous_live.py --port COM9 "
            "--mac 24:0A:C4:02:29:F0"
        )

    if not mac:
        raise SystemExit(
            "ESP32 MAC address is not configured.\n"
            "Either edit DEFAULT_MAC in the script or provide --mac."
        )


def load_artifacts(base_dir):
    model_path = base_dir / MODEL_FILE
    scaler_path = base_dir / SCALER_FILE
    encoder_path = base_dir / ENCODER_FILE
    metadata_path = base_dir / METADATA_FILE

    for path in (
        model_path,
        scaler_path,
        encoder_path,
        metadata_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required deployment file: {path.name}\n"
                "Place this script and all four deployment artifacts "
                "in the same directory."
            )

    print("Loading CARE-BED deployment artifacts...")

    model = tf.keras.models.load_model(
        model_path
    )

    scaler = joblib.load(
        scaler_path
    )

    encoder = joblib.load(
        encoder_path
    )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    classes = [
        canonical_label(x)
        for x in encoder.classes_
    ]

    if tuple(model.input_shape[1:]) != (
        SEGMENT_LENGTH,
        N_FEATURES,
    ):
        raise ValueError(
            f"Model input shape is {model.input_shape[1:]}, "
            f"expected {(SEGMENT_LENGTH, N_FEATURES)}."
        )

    if int(model.output_shape[-1]) != len(classes):
        raise ValueError(
            "Model output size does not match LabelEncoder classes."
        )

    if int(scaler.n_features_in_) != (
        SEGMENT_LENGTH * N_FEATURES
    ):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} values, "
            f"expected {SEGMENT_LENGTH * N_FEATURES}."
        )

    expected_classes = {
        "inactivity",
        "lying down",
        "sitting up",
        "fidgeting",
    }

    if set(classes) != expected_classes:
        raise ValueError(
            f"Unexpected model classes: {classes}. "
            f"Expected: {sorted(expected_classes)}"
        )

    metadata_classes = [
        canonical_label(x)
        for x in metadata.get(
            "classes",
            classes,
        )
    ]

    if metadata_classes != classes:
        raise ValueError(
            "Metadata class order does not match "
            "LabelEncoder class order."
        )

    print("Deployment artifacts loaded successfully.")
    print("Classes:", classes)

    return (
        model,
        scaler,
        encoder,
        metadata,
        classes,
    )


# ---------------------------------------------------------------------
# CSI parsing
# ---------------------------------------------------------------------

def parse_csi_line(
    line,
    expected_mac,
):
    """
    Validate one ESP32 CSI line and convert it to 64 amplitudes.

    The final CARE-BED live pipeline intentionally uses only the first
    128 CSI integers (64 complex values) even if the firmware reports
    a longer payload.
    """

    if line.count("[") != 1 or line.count("]") != 1:
        return None

    parts = line.split(",")

    if len(parts) != 26:
        return None

    if parts[0].strip() != "CSI_DATA":
        return None

    if parts[1].strip() != EXPECTED_ROLE:
        return None

    if (
        parts[2].strip().upper()
        != expected_mac.upper()
    ):
        return None

    try:
        declared_len = int(
            parts[24].strip()
        )

        match = re.search(
            r"\[(.*)\]",
            parts[25],
        )

        if match is None:
            return None

        payload = [
            int(x)
            for x in match.group(1).split()
        ]

    except (ValueError, IndexError):
        return None

    if len(payload) != declared_len:
        return None

    if len(payload) < FIRST_CSI_VALUES:
        return None

    values = np.asarray(
        payload[:FIRST_CSI_VALUES],
        dtype=np.float32,
    )

    imaginary = values[0::2]
    real = values[1::2]

    amplitudes = np.sqrt(
        imaginary ** 2
        + real ** 2
    ).astype(np.float32)

    if amplitudes.shape != (
        RAW_FEATURES,
    ):
        return None

    return amplitudes


# ---------------------------------------------------------------------
# Final CARE-BED W20 preprocessing
# ---------------------------------------------------------------------

def preprocess_window(
    raw_window,
    scaler,
):
    """
    Final live preprocessing path:

        20 x 64 amplitude window
        -> Hampel filtering
        -> Savitzky-Golay smoothing
        -> remove 10 unreliable features
        -> 20 x 54
        -> saved StandardScaler
        -> 1 x 20 x 54
    """

    arr = np.asarray(
        raw_window,
        dtype=np.float32,
    )

    if arr.shape != (
        SEGMENT_LENGTH,
        RAW_FEATURES,
    ):
        raise ValueError(
            f"Expected raw window "
            f"{(SEGMENT_LENGTH, RAW_FEATURES)}, "
            f"got {arr.shape}"
        )

    filtered_columns = []

    for col_idx in range(
        RAW_FEATURES
    ):
        series = pd.Series(
            arr[:, col_idx].astype(float)
        )

        hampel_result = hampel(
            series,
            window_size=HAMPEL_WINDOW,
        )

        hampel_values = np.asarray(
            hampel_result.filtered_data,
            dtype=float,
        )

        sg_values = savgol_filter(
            hampel_values,
            window_length=SAVGOL_WINDOW,
            polyorder=SAVGOL_POLYORDER,
        )

        filtered_columns.append(
            sg_values
        )

    filtered = np.stack(
        filtered_columns,
        axis=1,
    )

    trimmed = np.delete(
        filtered,
        np.asarray(
            COLUMNS_TO_DROP,
            dtype=int,
        ),
        axis=1,
    ).astype(np.float32)

    if trimmed.shape != (
        SEGMENT_LENGTH,
        N_FEATURES,
    ):
        raise ValueError(
            f"Expected processed window "
            f"{(SEGMENT_LENGTH, N_FEATURES)}, "
            f"got {trimmed.shape}"
        )

    scaled = scaler.transform(
        trimmed.reshape(
            1,
            SEGMENT_LENGTH * N_FEATURES,
        )
    )

    return scaled.reshape(
        1,
        SEGMENT_LENGTH,
        N_FEATURES,
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

def append_prediction(
    log_path,
    classes,
    predicted,
    confidence,
    probabilities,
    acquisition_s,
    processing_ms,
):
    if log_path is None:
        return

    new_file = not log_path.exists()

    with log_path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as fh:
        writer = csv.writer(
            fh
        )

        if new_file:
            writer.writerow(
                [
                    "timestamp_iso",
                    "predicted_activity",
                    "confidence",
                    "acquisition_time_s",
                    "processing_inference_ms",
                ]
                + [
                    f"p_{label.replace(' ', '_')}"
                    for label in classes
                ]
            )

        writer.writerow(
            [
                datetime.now()
                .astimezone()
                .isoformat(
                    timespec="milliseconds"
                ),
                predicted,
                f"{confidence:.8f}",
                f"{acquisition_s:.6f}",
                f"{processing_ms:.3f}",
            ]
            + [
                f"{float(prob):.8f}"
                for prob in probabilities
            ]
        )


# ---------------------------------------------------------------------
# Continuous live inference
# ---------------------------------------------------------------------

def main():
    args = parse_args()

    validate_hardware_config(
        args.port,
        args.mac,
    )

    base_dir = Path(
        __file__
    ).resolve().parent

    (
        model,
        scaler,
        encoder,
        metadata,
        classes,
    ) = load_artifacts(
        base_dir
    )

    log_path = (
        Path(args.log)
        if args.log
        else None
    )

    print()
    print("=" * 72)
    print("CARE-BED — CONTINUOUS LIVE ACTIVITY RECOGNITION")
    print("=" * 72)
    print(f"Serial port:     {args.port}")
    print(f"Baud rate:       {args.baud}")
    print(f"Expected MAC:    {args.mac}")
    print(f"Model input:     {SEGMENT_LENGTH} x {N_FEATURES}")
    print(f"Classes:         {classes}")
    print("Segment policy:  non-overlapping 20-frame windows")
    print(
        "Logging:         "
        + (
            str(log_path)
            if log_path is not None
            else "disabled"
        )
    )
    print("=" * 72)
    print()

    try:
        ser = serial.Serial(
            args.port,
            args.baud,
            timeout=1,
        )

    except serial.SerialException as exc:
        raise SystemExit(
            f"Could not open {args.port}: {exc}"
        )

    print(
        f"{args.port} opened successfully."
    )

    print(
        "CARE-BED is now monitoring continuously."
    )

    print(
        "Press Ctrl+C to stop.\n"
    )

    all_amplitudes = []
    segment_start = None
    last_valid_frame_time = time.monotonic()

    try:
        while True:
            raw = ser.readline()

            if not raw:
                if (
                    time.monotonic()
                    - last_valid_frame_time
                    >= NO_DATA_WARNING_SECONDS
                ):
                    print(
                        "\nWARNING: no valid CSI frame "
                        "for more than 10 seconds."
                    )

                    last_valid_frame_time = (
                        time.monotonic()
                    )

                continue

            line = raw.decode(
                "utf-8",
                errors="ignore",
            ).strip()

            if not line:
                continue

            amplitude = parse_csi_line(
                line,
                args.mac,
            )

            if amplitude is None:
                continue

            last_valid_frame_time = (
                time.monotonic()
            )

            if not all_amplitudes:
                segment_start = (
                    time.perf_counter()
                )

                print(
                    "Waiting for data for a new segment..."
                )

            all_amplitudes.append(
                amplitude
            )

            current_count = len(
                all_amplitudes
            )

            # Match the staged progress display used in the CARE-BED
            # console examples while keeping the final W20 segment size.
            if current_count % 5 == 0:
                print(
                    "." * 42
                    + f"  "
                    f"[{current_count}/{SEGMENT_LENGTH}]"
                )

            if current_count < SEGMENT_LENGTH:
                continue

            acquisition_s = (
                time.perf_counter()
                - segment_start
            )

            raw_window = np.stack(
                all_amplitudes
            ).astype(np.float32)

            print()
            print(
                f"Collected a segment of "
                f"{SEGMENT_LENGTH} samples. "
                "Processing..."
            )

            processing_start = (
                time.perf_counter()
            )

            print(
                "Step 1: Applying filters "
                "(Hampel, Savitzky-Golay)..."
            )

            # Step 1 + Step 2 + Step 3 are executed inside
            # preprocess_window(). The printed messages describe the
            # same processing chain exposed in the console.
            print(
                "Step 2: Dropping unreliable subcarriers..."
            )

            print(
                "Step 3: Scaling and reshaping..."
            )

            model_input = preprocess_window(
                raw_window,
                scaler,
            )

            print(
                "Step 4: BiLSTM model prediction..."
            )

            probabilities = model.predict(
                model_input,
                verbose=0,
            )[0]

            processing_ms = (
                time.perf_counter()
                - processing_start
            ) * 1000.0

            predicted_index = int(
                np.argmax(
                    probabilities
                )
            )

            predicted = classes[
                predicted_index
            ]

            confidence = float(
                probabilities[
                    predicted_index
                ]
            )

            print()
            print("=" * 58)

            print(
                f"ACTIVITY: {predicted.upper()} "
                f"(Confidence: "
                f"{100.0 * confidence:.2f}%)"
            )

            print(
                "Probabilities: "
                + " | ".join(
                    f"{label}="
                    f"{100.0 * float(prob):.2f}%"
                    for label, prob in zip(
                        classes,
                        probabilities,
                    )
                )
            )

            print(
                f"Acquisition: "
                f"{acquisition_s:.3f} s"
            )

            print(
                f"Preprocess+inference: "
                f"{processing_ms:.1f} ms"
            )

            print("=" * 58)
            print()

            append_prediction(
                log_path,
                classes,
                predicted,
                confidence,
                probabilities,
                acquisition_s,
                processing_ms,
            )

            # Start a completely new non-overlapping W20 segment.
            all_amplitudes = []
            segment_start = None

    except KeyboardInterrupt:
        print(
            "\nStopping CARE-BED..."
        )

    finally:
        if ser.is_open:
            ser.close()

        print(
            "Serial port closed."
        )

        if log_path is not None:
            print(
                f"Prediction log: "
                f"{log_path.resolve()}"
            )


if __name__ == "__main__":
    main()
