"""Collect one-letter ASL fingerspelling clips in A-Z cycles.

This is a fast single-letter variant of collect_fingerspelling_data.py. It saves
clips, landmark CSVs, and labels.csv rows using the same dataset layout.
"""

from __future__ import annotations

import argparse
import csv
import string
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from collect_fingerspelling_data import (
    CLIPS_DIR,
    LABELS_CSV,
    LABEL_FIELDNAMES,
    LANDMARK_FIELDNAMES,
    LANDMARKS_DIR,
    add_preview_overlay,
    append_label,
    cleanup_temp,
    create_holistic_landmarker,
    draw_prompt,
    draw_tracking_overlay,
    ensure_dirs,
    holistic_result_to_csv_rows,
    next_clip_number,
    open_camera,
    open_video_writer,
)


@dataclass(frozen=True)
class SavedClip:
    clip_id: str
    video_path: Path
    landmark_csv_path: Path
    letter: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one-letter ASL fingerspelling clips in repeated A-Z order."
    )
    parser.add_argument("--fps", type=float, default=20.0, help="Recording and processing FPS.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--signer-id", default="unknown", help="Identifier for the signer.")
    parser.add_argument(
        "--start-letter",
        default="A",
        help="Letter to start from before cycling through A-Z.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Optional number of full A-Z cycles to collect before stopping.",
    )
    parser.add_argument(
        "--ignore-letters",
        default="",
        help="Letters to skip while cycling, e.g. ABC or A,B,C.",
    )
    parser.add_argument(
        "--no-preview-overlay",
        action="store_true",
        help="Disable MediaPipe landmark overlays in the live camera preview.",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror webcam frames.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.start_letter = args.start_letter.strip().upper()
    args.ignore_letters = parse_letter_set(args.ignore_letters)
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")
    if len(args.start_letter) != 1 or args.start_letter not in string.ascii_uppercase:
        raise ValueError("--start-letter must be one letter from A-Z")
    if args.repetitions is not None and args.repetitions <= 0:
        raise ValueError("--repetitions must be greater than 0")
    if args.start_letter in args.ignore_letters:
        raise ValueError("--start-letter cannot also be in --ignore-letters")
    if len(args.ignore_letters) >= len(string.ascii_uppercase):
        raise ValueError("--ignore-letters cannot skip every letter")


def parse_letter_set(value: str) -> set[str]:
    separators = {",", " ", "\t", "\n", "\r"}
    invalid_characters = {
        character
        for character in value.upper()
        if character not in string.ascii_uppercase and character not in separators
    }
    if invalid_characters:
        raise ValueError(f"Invalid --ignore-letters characters: {''.join(sorted(invalid_characters))}")

    letters = {character for character in value.upper() if character in string.ascii_uppercase}
    return letters


def alphabet_from(start_letter: str, ignored_letters: set[str]) -> list[str]:
    letters = list(string.ascii_uppercase)
    start_index = letters.index(start_letter)
    ordered_letters = letters[start_index:] + letters[:start_index]
    return [letter for letter in ordered_letters if letter not in ignored_letters]


def wait_for_letter_action(
    cap: cv2.VideoCapture,
    letter: str,
    landmarker,
    preview_started_at: float,
    mirror: bool,
    last_saved: SavedClip | None,
) -> str:
    print()
    print(f"Current letter: {letter}")
    print("Press s to record, u to undo the latest saved clip, or q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Webcam frame read failed before recording started")

        if mirror:
            frame = cv2.flip(frame, 1)

        if landmarker is not None:
            add_preview_overlay(frame, landmarker, preview_started_at)
        undo_text = "u undo last" if last_saved is not None else "u undo unavailable"
        draw_prompt(frame, letter, f"Press s to record. {undo_text}. Press q to quit.")
        cv2.imshow("ASL Fingerspelling Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            return "start"
        if key == ord("u"):
            return "undo"
        if key == ord("q"):
            return "quit"


def next_video_timestamp_ms(preview_started_at: float) -> int:
    timestamp_ms = int((time.monotonic() - preview_started_at) * 1000)
    last_timestamp_ms = getattr(add_preview_overlay, "_last_timestamp_ms", -1)
    if timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    add_preview_overlay._last_timestamp_ms = timestamp_ms
    return timestamp_ms


def record_clip_with_landmarks(
    cap: cv2.VideoCapture,
    letter: str,
    clip_path: Path,
    fps: float,
    landmarker,
    preview_started_at: float,
    mirror: bool,
    show_overlay: bool,
) -> tuple[int, float, list[dict]]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = open_video_writer(clip_path, fps, width, height)

    landmark_rows: list[dict] = []
    frame_count = 0
    frame_interval = 1.0 / fps
    started_at = time.monotonic()
    next_frame_at = started_at

    print("Recording. Press e to stop.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Webcam frame read failed during recording")

            if mirror:
                frame = cv2.flip(frame, 1)

            preview = frame.copy()
            now = time.monotonic()
            if now >= next_frame_at:
                writer.write(frame)
                timestamp_ms = next_video_timestamp_ms(preview_started_at)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(image, timestamp_ms)
                landmark_rows.extend(
                    holistic_result_to_csv_rows(
                        result,
                        frame_count,
                        frame_count,
                        timestamp_ms,
                    )
                )
                if show_overlay:
                    draw_tracking_overlay(preview, result)

                frame_count += 1
                while next_frame_at <= now:
                    next_frame_at += frame_interval
            elif show_overlay:
                add_preview_overlay(preview, landmarker, preview_started_at)

            draw_prompt(preview, letter, "Recording. Press e to stop.")
            cv2.imshow("ASL Fingerspelling Data Collection", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                break
    finally:
        writer.release()

    duration = time.monotonic() - started_at
    return frame_count, duration, landmark_rows


def write_landmark_csv(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as landmarks_file:
        writer = csv.DictWriter(landmarks_file, fieldnames=LANDMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def save_letter_clip(
    temp_path: Path,
    clip_number_value: int,
    letter: str,
    fps: float,
    signer_id: str,
    num_frames: int,
    duration_seconds: float,
    landmark_rows: list[dict],
) -> tuple[int, SavedClip]:
    clip_id = f"clip_{clip_number_value:06d}"
    video_path = CLIPS_DIR / f"{clip_id}.mp4"
    landmark_csv_path = LANDMARKS_DIR / f"{clip_id}.csv"
    while video_path.exists() or landmark_csv_path.exists():
        clip_number_value += 1
        clip_id = f"clip_{clip_number_value:06d}"
        video_path = CLIPS_DIR / f"{clip_id}.mp4"
        landmark_csv_path = LANDMARKS_DIR / f"{clip_id}.csv"

    temp_path.replace(video_path)
    write_landmark_csv(landmark_csv_path, landmark_rows)
    append_label(
        {
            "clip_id": clip_id,
            "landmark_csv_path": str(landmark_csv_path),
            "expected_text": letter,
            "video_path": str(video_path),
            "fps": fps,
            "num_frames": num_frames,
            "duration_seconds": f"{duration_seconds:.3f}",
            "signer_id": signer_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "notes": "single-letter",
        }
    )

    print(f"Saved {video_path}")
    print(f"Saved {landmark_csv_path}")
    return clip_number_value + 1, SavedClip(
        clip_id=clip_id,
        video_path=video_path,
        landmark_csv_path=landmark_csv_path,
        letter=letter,
    )


def remove_label_row(clip_id: str) -> None:
    if not LABELS_CSV.exists():
        return

    with LABELS_CSV.open(newline="", encoding="utf-8") as labels_file:
        rows = list(csv.DictReader(labels_file))

    kept_rows = [row for row in rows if row.get("clip_id") != clip_id]
    with LABELS_CSV.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=LABEL_FIELDNAMES)
        writer.writeheader()
        writer.writerows(kept_rows)


def undo_saved_clip(saved_clip: SavedClip) -> None:
    cleanup_temp(saved_clip.video_path)
    cleanup_temp(saved_clip.landmark_csv_path)
    remove_label_row(saved_clip.clip_id)
    print(f"Removed {saved_clip.clip_id} ({saved_clip.letter})")


def collect() -> None:
    args = parse_args()
    validate_args(args)
    ensure_dirs()

    cap = open_camera(args.camera, args.fps)
    clip_number_value = next_clip_number()
    letters = alphabet_from(args.start_letter, args.ignore_letters)
    preview_started_at = time.monotonic()
    landmarker = create_holistic_landmarker()
    mirror = not args.no_mirror
    completed_cycles = 0
    saved_clips: list[SavedClip] = []
    if args.ignore_letters:
        print(f"Ignoring letters: {''.join(sorted(args.ignore_letters))}")

    try:
        while args.repetitions is None or completed_cycles < args.repetitions:
            for letter in letters:
                if args.repetitions is not None and completed_cycles >= args.repetitions:
                    break

                while True:
                    action = wait_for_letter_action(
                        cap,
                        letter,
                        None if args.no_preview_overlay else landmarker,
                        preview_started_at,
                        mirror=mirror,
                        last_saved=saved_clips[-1] if saved_clips else None,
                    )
                    if action == "quit":
                        return
                    if action == "undo":
                        if saved_clips:
                            undo_saved_clip(saved_clips.pop())
                        else:
                            print("No saved clip to remove in this session.")
                        continue

                    temp_path = CLIPS_DIR / "_pending_single_letter_clip.mp4"
                    cleanup_temp(temp_path)
                    recorded_frames, recorded_duration, landmark_rows = record_clip_with_landmarks(
                        cap,
                        letter,
                        temp_path,
                        args.fps,
                        landmarker,
                        preview_started_at,
                        mirror=mirror,
                        show_overlay=not args.no_preview_overlay,
                    )
                    print(
                        f"Recorded {recorded_frames} frames "
                        f"({recorded_duration:.2f} seconds at target {args.fps:g} FPS)."
                    )
                    clip_number_value, saved_clip = save_letter_clip(
                        temp_path,
                        clip_number_value,
                        letter,
                        args.fps,
                        args.signer_id,
                        recorded_frames,
                        recorded_duration,
                        landmark_rows,
                    )
                    saved_clips.append(saved_clip)
                    break

            completed_cycles += 1
            letters = alphabet_from("A", args.ignore_letters)
    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collect()
