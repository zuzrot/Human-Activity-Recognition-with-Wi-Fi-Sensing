#!/usr/bin/env python3
"""Reproduce Configuration A timestamp statistics reported for CARE-BED.

Uses the original archived ESP32 32-bit local timestamps (microseconds) in
field 19 (zero-based index 18) of raw CSI_DATA rows. The 13 recording-level
units are reconstructed using the same 1-based row ranges as the final
CARE_BED_FINAL_A13_B12_W20_HSG_25fold_LORO.ipynb notebook.

Example from the repository root:
    python notebooks/CARE_BED_ConfigurationA_timestamp_audit.py \
        --input datasets/envA_dataset.zip \
        --output notebooks/CARE_BED_ConfigurationA_timing_by_unit.csv

Expected reference values: 12,026 frames, total elapsed 6,270.680645 s,
~1.917814 valid CSI frames/s, and W20 median first-to-last time ~10.47323 s.
"""

import argparse
import csv
import io
import statistics
import zipfile
from pathlib import Path

TIMESTAMP_MODULUS = 2 ** 32
MICROSECONDS_PER_SECOND = 1_000_000
WINDOW_FRAMES = 20
WINDOW_STRIDE = 5

# IMPORTANT: these are not newly collected recordings; the ranges reconstruct
# the 13 historical recording-level units in Configuration A.
RECORDINGS = {
    "inactivity": ("br_ak_merged.csv", [("A_inactivity_s01", 1, 1765),
                                           ("A_inactivity_s02", 1766, 2427),
                                           ("A_inactivity_s03", 2428, 3095),
                                           ("A_inactivity_s04", 3096, 3202)]),
    "lying down": ("lezenie_merged.csv", [("A_lying_down_s01", 1, 607),
                                           ("A_lying_down_s02", 608, 1779),
                                           ("A_lying_down_s03", 1780, 2947)]),
    "sitting up": ("siedzenie_merged.csv", [("A_sitting_up_s01", 1, 601),
                                            ("A_sitting_up_s02", 602, 1739),
                                            ("A_sitting_up_s03", 1740, 2922)]),
    "fidgeting": ("wiercenie_się_test.csv", [("A_fidgeting_s01", 1, 1123),
                                               ("A_fidgeting_s02", 1124, 1810),
                                               ("A_fidgeting_s03", 1811, 2955)]),
}


def elapsed_seconds(first_us: int, last_us: int) -> float:
    """Elapsed first-to-last time, accounting for one 32-bit ESP timer wrap."""
    return ((last_us - first_us) % TIMESTAMP_MODULUS) / MICROSECONDS_PER_SECOND


def parse_timestamp(line: str) -> int:
    # Some old raw files consist of a single quoted CSV field containing the
    # entire CSI_DATA line. Others contain normal unquoted CSV rows.
    fields = next(csv.reader([line]))
    if len(fields) == 1:
        fields = next(csv.reader([fields[0]]))
    if len(fields) < 19 or fields[0].strip() != "CSI_DATA":
        raise ValueError("Unexpected CSI_DATA line format")
    return int(fields[18].strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("datasets/envA_dataset.zip"))
    ap.add_argument("--output", type=Path,
                    default=Path("notebooks/CARE_BED_ConfigurationA_timing_by_unit.csv"))
    args = ap.parse_args()
    rows = []
    all_w20_elapsed = []
    with zipfile.ZipFile(args.input, "r") as archive:
        for activity, (filename, intervals) in RECORDINGS.items():
            matching = [n for n in archive.namelist() if n.endswith("/" + filename) or n == filename]
            if len(matching) != 1:
                raise ValueError(f"Expected exactly one archive entry for {filename}; found: {matching}")
            text = archive.read(matching[0]).decode("utf-8-sig", errors="strict")
            timestamps = [parse_timestamp(line) for line in text.splitlines() if line.strip()]
            if len(timestamps) != intervals[-1][2]:
                raise ValueError(f"Unexpected row count for {filename}: {len(timestamps)}")
            for unit, start, end in intervals:
                series = timestamps[start - 1:end]
                elapsed = elapsed_seconds(series[0], series[-1])
                w20 = [elapsed_seconds(series[k], series[k + WINDOW_FRAMES - 1])
                       for k in range(0, len(series) - WINDOW_FRAMES + 1, WINDOW_STRIDE)]
                all_w20_elapsed.extend(w20)
                rows.append({
                    "recording_id": unit,
                    "activity": activity,
                    "source_file": filename,
                    "start_row_1based": start,
                    "end_row_1based": end,
                    "valid_frames": len(series),
                    "first_timestamp_us": series[0],
                    "last_timestamp_us": series[-1],
                    "elapsed_s": f"{elapsed:.6f}",
                    "valid_frames_per_elapsed_s": f"{len(series)/elapsed:.6f}",
                    "W20_windows_stride5": len(w20),
                    "W20_median_first_to_last_s": f"{statistics.median(w20):.6f}",
                })
    total_frames = sum(r["valid_frames"] for r in rows)
    total_elapsed = sum(float(r["elapsed_s"]) for r in rows)
    median_w20 = statistics.median(all_w20_elapsed)
    assert len(rows) == 13 and total_frames == 12026
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reconstructed recording-level units: {len(rows)}")
    print(f"Valid CSI frames: {total_frames:,}")
    print(f"Sum of first-to-last unit intervals: {total_elapsed:.6f} s")
    print(f"Effective rate (total frames / sum of intervals): {total_frames / total_elapsed:.6f} frames/s")
    print(f"Median first-to-last-frame interval for W20, stride=5: {median_w20:.6f} s")
    print(f"W20 windows: {len(all_w20_elapsed)}")
    print(f"Unit-level output: {args.output}")
    print("Note: archival, irregular timestamp-derived timing; not sustained receiver throughput.")


if __name__ == "__main__":
    main()
