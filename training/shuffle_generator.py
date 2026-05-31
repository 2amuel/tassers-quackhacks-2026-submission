#!/usr/bin/env python3
"""Generate shuffled training videos by concatenating random letter clips."""

from __future__ import annotations

import argparse
import random
import subprocess
import tempfile
from pathlib import Path


DEFAULT_MIN_LENGTH = 8
DEFAULT_MAX_LENGTH = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random concatenated mp4 sequences from training/letters."
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
        help="Folder where generated mp4 files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible output.",
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


def ffmpeg_concat(sequence: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as list_file:
        list_path = Path(list_file.name)
        for video_path in sequence:
            escaped_path = str(video_path.resolve()).replace("'", "'\\''")
            list_file.write(f"file '{escaped_path}'\n")

    try:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
        subprocess.run(command, check=True)
    finally:
        list_path.unlink(missing_ok=True)


def generate_sequences(
    mp4_files: list[Path], output_dir: Path, num_sequences: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(1, num_sequences + 1):
        sequence_length = random.randint(DEFAULT_MIN_LENGTH, DEFAULT_MAX_LENGTH)
        sequence = pick_random_sequence(mp4_files, sequence_length)
        output_path = output_dir / f"sequence_{index:04d}.mp4"

        ffmpeg_concat(sequence, output_path)

        picked_names = ", ".join(video.name for video in sequence)
        print(
            f"Wrote {output_path} "
            f"({sequence_length} clips: {picked_names})"
        )


def main() -> None:
    args = parse_args()
    if args.num_sequences < 1:
        raise ValueError("--num-sequences must be at least 1")

    if args.seed is not None:
        random.seed(args.seed)

    mp4_files = find_mp4_files(args.letters_dir)
    generate_sequences(mp4_files, args.output_dir, args.num_sequences)


if __name__ == "__main__":
    main()
