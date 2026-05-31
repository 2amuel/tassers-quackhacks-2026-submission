"""Record labeled ASL training data with MediaPipe Holistic landmarks.

The workflow is:
1. Show 10-15 random letters to sign.
2. Let the user record a webcam clip.
3. Let the user re-record or label the clip.
4. Sample the clip at 5 FPS and run MediaPipe Holistic.
5. Ask for the frame ranges where each prompted letter is being signed.
6. Write a holistic_landmarks-style CSV with one extra `label` column.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import string
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp

from record_webcam import (
    DEFAULT_HOLISTIC_MODEL_PATH,
    LANDMARK_FIELDNAMES,
    CameraInfo,
    camera_backend,
    choose_camera,
    create_holistic_landmarker,
    ensure_holistic_model,
    holistic_result_to_csv_rows,
    make_output_dir,
    open_writer,
)


TRAINING_FIELDNAMES = [*LANDMARK_FIELDNAMES, "label"]
BLANK_LABEL = "<blank>"
SAMPLE_INTERVAL_SECONDS = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record labeled ASL letter training data at 5 sampled frames per second."
    )
    parser.add_argument("--camera", type=int, default=None, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested video FPS.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/training-YYYYmmdd-HHMMSS.",
    )
    parser.add_argument(
        "--holistic-model-path",
        type=Path,
        default=DEFAULT_HOLISTIC_MODEL_PATH,
        help="Path to a MediaPipe Holistic Landmarker .task model.",
    )
    parser.add_argument(
        "--letter-count",
        type=int,
        default=None,
        help="Number of random letters to prompt. Defaults to a random count from 10 to 15.",
    )
    parser.add_argument("--filename", default="training_capture.mp4", help="Output video filename.")
    parser.add_argument("--no-preview", action="store_true", help="Disable live preview while recording.")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror webcam frames.")
    parser.add_argument("--list-cameras", action="store_true", help="Print detected cameras and exit.")
    parser.add_argument(
        "--min-face-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for detecting a face.",
    )
    parser.add_argument(
        "--min-face-landmarks-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for accepting face landmarks.",
    )
    parser.add_argument(
        "--min-pose-detection-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for detecting body pose.",
    )
    parser.add_argument(
        "--min-pose-landmarks-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for accepting pose landmarks.",
    )
    parser.add_argument(
        "--min-hand-landmarks-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for accepting hand landmarks.",
    )

    args = parser.parse_args()
    if args.letter_count is not None and not 10 <= args.letter_count <= 15:
        parser.error("--letter-count must be between 10 and 15")
    return args


def make_training_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs") / f"training-{stamp}"
    return make_output_dir(output_dir)


def random_prompt_letters(letter_count: int | None) -> list[str]:
    count = letter_count if letter_count is not None else random.randint(10, 15)
    return random.choices(string.ascii_uppercase, k=count)


def wait_for_enter(message: str) -> None:
    input(f"{message}\nPress Enter when ready.")


def record_clip(args: argparse.Namespace, camera: CameraInfo, output_path: Path) -> dict:
    cap = cv2.VideoCapture(camera.index, camera_backend())
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at camera index {camera.index} ({camera.name})")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.width
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.height
    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    if fps <= 1:
        fps = args.fps

    writer = open_writer(output_path, fps, width, height)
    frame_count = 0
    start_time = time.monotonic()

    print("\nRecording. Press q in the preview window to stop, or Ctrl-C in the terminal.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed; stopping recording.")
                break

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            writer.write(frame)
            frame_count += 1

            if not args.no_preview:
                cv2.imshow("ASL Training Recorder", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Stopping capture.")
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    runtime = time.monotonic() - start_time
    return {
        "frame_count": frame_count,
        "runtime_seconds": runtime,
        "fps": fps,
        "width": width,
        "height": height,
    }


def prompt_recording_action() -> str:
    while True:
        action = input("\nType r to re-record, or l to add labels: ").strip().lower()
        if action in {"r", "record", "rerecord", "re-record"}:
            return "rerecord"
        if action in {"l", "label", "labels"}:
            return "label"
        print("Please type r or l.")


def load_video_samples(
    video_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[dict], int]:
    holistic_model_path = ensure_holistic_model(args.holistic_model_path)
    sampled_frames_dir = output_dir / "sampled_frames"
    sampled_frames_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open recorded video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    if source_fps <= 1:
        source_fps = args.fps

    landmark_rows: list[dict] = []
    sample_count = 0
    next_sample_time = 0.0
    frame_index = 0

    with create_holistic_landmarker(holistic_model_path, args) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_seconds = frame_index / source_fps
            if timestamp_seconds + 1e-9 >= next_sample_time:
                timestamp_ms = int(round(timestamp_seconds * 1000))
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(image, timestamp_ms)

                sampled_rows = holistic_result_to_csv_rows(
                    result,
                    frame_index,
                    sample_count,
                    timestamp_ms,
                )
                landmark_rows.extend(sampled_rows)

                sample_path = sampled_frames_dir / f"sample_{sample_count:06d}.jpg"
                cv2.imwrite(str(sample_path), frame)

                sample_count += 1
                next_sample_time += SAMPLE_INTERVAL_SECONDS

            frame_index += 1

    cap.release()
    return landmark_rows, sample_count


def prompt_label_ranges(prompt_letters: list[str], sample_count: int) -> list[str]:
    labels = [BLANK_LABEL] * sample_count

    print("\nLabel the sampled frames.")
    print(f"There are {sample_count} samples: 0 through {sample_count - 1}.")
    print("Each sample is 0.2 seconds apart.")
    print("Leave transition/rest frames unassigned; they will remain <blank>.")
    print("Enter ranges as start-end, for example 4-8. Use s to skip a letter.")

    for index, letter in enumerate(prompt_letters, start=1):
        while True:
            raw = input(f"{index}. {letter} range: ").strip().lower()
            if raw in {"", "s", "skip"}:
                break

            try:
                start_text, end_text = raw.replace(" ", "").split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                print("Please enter a range like 4-8, or s to skip.")
                continue

            if start > end:
                start, end = end, start
            if start < 0 or end >= sample_count:
                print(f"Range must be between 0 and {sample_count - 1}.")
                continue

            for sample_index in range(start, end + 1):
                labels[sample_index] = letter
            break

    return labels


def add_labels_to_rows(rows: list[dict], labels: list[str]) -> list[dict]:
    labeled_rows: list[dict] = []
    for row in rows:
        labeled_row = dict(row)
        labeled_row["label"] = labels[int(row["sample"])]
        labeled_rows.append(labeled_row)
    return labeled_rows


def write_labeled_csv(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAINING_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_label_summary(output_dir: Path, prompt_letters: list[str], labels: list[str]) -> None:
    summary_path = output_dir / "labels.json"
    payload = {
        "prompt_letters": prompt_letters,
        "blank_label": BLANK_LABEL,
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "sample_labels": [
            {
                "sample": sample_index,
                "timestamp_ms": int(round(sample_index * SAMPLE_INTERVAL_SECONDS * 1000)),
                "label": label,
            }
            for sample_index, label in enumerate(labels)
        ],
    }
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(payload, summary_file, indent=2)


def main() -> None:
    args = parse_args()
    camera = choose_camera(args)
    output_dir = make_training_output_dir(args.output_dir)
    video_path = output_dir / args.filename
    csv_path = output_dir / "labeled_holistic_landmarks.csv"
    prompt_letters = random_prompt_letters(args.letter_count)

    print("\nSign these letters in order:")
    print(" ".join(prompt_letters))
    print("\nLeave a short pause between letters so blank frames can be labeled.")

    clip_info = None
    while True:
        wait_for_enter("\nGet ready to record.")
        clip_info = record_clip(args, camera, video_path)
        print(f"\nRecorded {clip_info['runtime_seconds']:.1f} seconds to {video_path}")

        action = prompt_recording_action()
        if action == "label":
            break

    print("\nConverting recording to 5 FPS landmark samples...")
    landmark_rows, sample_count = load_video_samples(video_path, output_dir, args)
    labels = prompt_label_ranges(prompt_letters, sample_count)
    labeled_rows = add_labels_to_rows(landmark_rows, labels)
    write_labeled_csv(csv_path, labeled_rows)
    write_label_summary(output_dir, prompt_letters, labels)

    metadata = {
        "camera": asdict(camera),
        "prompt_letters": prompt_letters,
        "video_path": str(video_path),
        "labeled_landmark_csv_path": str(csv_path),
        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS,
        "sample_rate_fps": 1 / SAMPLE_INTERVAL_SECONDS,
        "sample_count": sample_count,
        "clip": clip_info,
        "landmark_groups": [
            "pose_world",
            "pose",
            "left_hand",
            "left_hand_world",
            "right_hand",
            "right_hand_world",
            "face",
        ],
    }
    with (output_dir / "training_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    print(f"\nWrote labeled landmark CSV to {csv_path}")
    print(f"Wrote labels summary to {output_dir / 'labels.json'}")
    print(f"Wrote metadata to {output_dir / 'training_metadata.json'}")


if __name__ == "__main__":
    main()
