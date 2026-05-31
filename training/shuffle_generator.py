#!/usr/bin/env python3
"""Generate shuffled training landmark CSVs from random letter clips."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from pathlib import Path


DEFAULT_MIN_LENGTH = 8
DEFAULT_MAX_LENGTH = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random concatenated landmark CSV sequences from training/letters."
    )
    parser.add_argument(
        "-n",
        "--num-sequences",
        type=int,
        required=True,
        help="Number of shuffled sequences to generate.",
    )
    parser.add_argument(
        "--letters-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "letters",
        help="Folder containing source mp4 letter clips.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data_output",
        help="Folder where generated sequence CSVs and labels will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=Path,
        default=None,
        help=(
            "Folder where per-letter landmark output folders will be written. "
            "Defaults to training/letter_landmarks."
        ),
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=None,
        help=(
            "Folder where concatenated sequence CSVs will be written. "
            "Defaults to OUTPUT_DIR/sequences."
        ),
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Path to the label manifest CSV. Defaults to OUTPUT_DIR/labels.csv.",
    )
    parser.add_argument(
        "--record-webcam-script",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "src" / "record_webcam.py",
        help="Path to src/record_webcam.py.",
    )
    return parser.parse_args()


def find_mp4_files(letters_dir: Path) -> list[Path]:
    if not letters_dir.exists():
        raise FileNotFoundError(f"Letters folder does not exist: {letters_dir}")

    mp4_files = sorted(letters_dir.glob("*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No .mp4 files found in: {letters_dir}")

    return mp4_files


def pick_random_sequence(mp4_files: list[Path], sequence_length: int) -> list[Path]:
    if sequence_length > len(mp4_files):
        raise ValueError(
            f"Cannot build a {sequence_length}-clip sequence from only "
            f"{len(mp4_files)} source clips without repeats."
        )

    available_indices = list(range(len(mp4_files)))
    chosen_indices = random.sample(available_indices, k=sequence_length)
    return [mp4_files[index] for index in chosen_indices]


def expected_text_for_sequence(sequence: list[Path]) -> str:
    return "".join(video.name[0].upper() for video in sequence)


def make_portable_path(path: Path) -> str:
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(resolved_path)


def generate_letter_landmark_csv(
    record_webcam_script: Path,
    letter_video: Path,
    landmarks_dir: Path,
) -> Path:
    landmark_output_dir = landmarks_dir / letter_video.stem
    landmark_csv_path = landmark_output_dir / "holistic_landmarks.csv"
    if landmark_csv_path.exists() and landmark_csv_path.stat().st_size > 0:
        return landmark_csv_path

    landmark_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(record_webcam_script),
        "--input-video",
        str(letter_video),
        "--output-dir",
        str(landmark_output_dir),
    ]
    subprocess.run(command, check=True)
    return landmark_csv_path


def generate_letter_landmark_csvs(
    letter_videos: list[Path],
    landmarks_dir: Path,
    record_webcam_script: Path,
) -> dict[Path, Path]:
    landmark_csvs = {}
    for index, letter_video in enumerate(letter_videos, start=1):
        landmark_csv_path = generate_letter_landmark_csv(
            record_webcam_script,
            letter_video,
            landmarks_dir,
        )
        landmark_csvs[letter_video] = landmark_csv_path
        print(f"[{index}/{len(letter_videos)}] Ready {landmark_csv_path}")

    return landmark_csvs


def offset_row_value(row: dict[str, str], key: str, offset: int) -> None:
    value = row.get(key)
    if value not in {None, ""}:
        row[key] = str(int(value) + offset)


def concatenate_landmark_csvs(source_csvs: list[Path], output_csv_path: Path) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    sample_offset = 0
    frame_offset = 0
    timestamp_offset = 0

    with output_csv_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = None
        for source_csv in source_csvs:
            with source_csv.open(newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV has no header: {source_csv}")
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                elif reader.fieldnames != fieldnames:
                    raise ValueError(
                        f"CSV header mismatch in {source_csv}: "
                        f"expected {fieldnames}, got {reader.fieldnames}"
                    )

                max_sample = -1
                max_frame = -1
                max_timestamp = -1
                for row in reader:
                    if row.get("sample") not in {None, ""}:
                        max_sample = max(max_sample, int(row["sample"]))
                    if row.get("frame") not in {None, ""}:
                        max_frame = max(max_frame, int(row["frame"]))
                    if row.get("timestamp_ms") not in {None, ""}:
                        max_timestamp = max(max_timestamp, int(row["timestamp_ms"]))

                    offset_row_value(row, "sample", sample_offset)
                    offset_row_value(row, "frame", frame_offset)
                    offset_row_value(row, "timestamp_ms", timestamp_offset)
                    writer.writerow(row)

            sample_offset += max_sample + 1
            frame_offset += max_frame + 1
            timestamp_offset += max_timestamp + 1

    if fieldnames is None:
        raise ValueError(f"No source CSVs were provided for {output_csv_path}")


def generate_sequences(
    mp4_files: list[Path],
    sequence_csvs_dir: Path,
    letter_landmark_csvs: dict[Path, Path],
    labels_csv_path: Path,
    num_sequences: int,
) -> None:
    sequence_csvs_dir.mkdir(parents=True, exist_ok=True)
    labels_csv_path.parent.mkdir(parents=True, exist_ok=True)
    label_rows = []

    for index in range(1, num_sequences + 1):
        clip_id = f"sequence_{index:04d}"
        sequence_length = random.randint(DEFAULT_MIN_LENGTH, DEFAULT_MAX_LENGTH)
        sequence = pick_random_sequence(mp4_files, sequence_length)
        sequence_csv_path = sequence_csvs_dir / f"{clip_id}.csv"
        source_csvs = [letter_landmark_csvs[letter_video] for letter_video in sequence]

        concatenate_landmark_csvs(source_csvs, sequence_csv_path)
        expected_text = expected_text_for_sequence(sequence)
        label_rows.append(
            {
                "clip_id": clip_id,
                "landmark_csv_path": make_portable_path(sequence_csv_path),
                "expected_text": expected_text,
            }
        )

        picked_names = ", ".join(video.name for video in sequence)
        print(
            f"Wrote {sequence_csv_path} "
            f"({sequence_length} letters, expected text {expected_text}: {picked_names})"
        )

    with labels_csv_path.open("w", newline="", encoding="utf-8") as labels_file:
        fieldnames = ["clip_id", "landmark_csv_path", "expected_text"]
        writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(label_rows)

    print(f"Wrote labels CSV to {labels_csv_path}")


def main() -> None:
    args = parse_args()
    if args.num_sequences < 1:
        raise ValueError("--num-sequences must be at least 1")

    if args.seed is not None:
        random.seed(args.seed)

    mp4_files = find_mp4_files(args.letters_dir)
    landmarks_dir = args.landmarks_dir or Path(__file__).resolve().parent / "letter_landmarks"
    sequences_dir = args.sequences_dir or args.output_dir / "sequences"
    labels_csv_path = args.labels_csv or args.output_dir / "labels.csv"
    letter_landmark_csvs = generate_letter_landmark_csvs(
        mp4_files,
        landmarks_dir,
        args.record_webcam_script,
    )
    generate_sequences(
        mp4_files,
        sequences_dir,
        letter_landmark_csvs,
        labels_csv_path,
        args.num_sequences,
    )


if __name__ == "__main__":
    main()
