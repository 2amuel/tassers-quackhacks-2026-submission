"""Collect ASL fingerspelling clips and MediaPipe Holistic landmark CSVs.

The script records variable-length webcam clips, asks whether to accept them,
then processes accepted clips at 20 FPS into the same long-format Holistic CSV
schema used by src/record_webcam.py.
"""

from __future__ import annotations

import argparse
import csv
import random
import string
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


DATA_DIR = Path("data")
CLIPS_DIR = DATA_DIR / "clips"
LANDMARKS_DIR = DATA_DIR / "landmarks"
LABELS_CSV = DATA_DIR / "labels.csv"
HOLISTIC_MODEL_PATH = Path("src/models/holistic_landmarker.task")
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

LABEL_FIELDNAMES = [
    "clip_id",
    "landmark_csv_path",
    "expected_text",
    "video_path",
    "fps",
    "num_frames",
    "duration_seconds",
    "signer_id",
    "timestamp",
    "notes",
]

LANDMARK_FIELDNAMES = [
    "frame",
    "sample",
    "timestamp_ms",
    "landmark_group",
    "coordinate_space",
    "index",
    "x",
    "y",
    "z",
    "visibility",
    "presence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ASL fingerspelling clips and MediaPipe hand landmarks."
    )
    parser.add_argument("--min-len", type=int, default=5, help="Minimum target length.")
    parser.add_argument("--max-len", type=int, default=15, help="Maximum target length.")
    parser.add_argument("--fps", type=float, default=20.0, help="Recording and processing FPS.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--signer-id", default="unknown", help="Identifier for the signer.")
    return parser.parse_args()


def ensure_dirs() -> None:
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if args.min_len <= 0:
        raise ValueError("--min-len must be greater than 0")
    if args.max_len < args.min_len:
        raise ValueError("--max-len must be greater than or equal to --min-len")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than 0")


def generate_target(min_len: int, max_len: int) -> str:
    length = random.randint(min_len, max_len)
    return "".join(random.choice(string.ascii_uppercase) for _ in range(length))


def next_clip_number() -> int:
    max_number = 0

    if LABELS_CSV.exists():
        with LABELS_CSV.open(newline="", encoding="utf-8") as labels_file:
            reader = csv.DictReader(labels_file)
            for row in reader:
                max_number = max(max_number, clip_number(row.get("clip_id", "")))

    for path in CLIPS_DIR.glob("clip_*.mp4"):
        max_number = max(max_number, clip_number(path.stem))

    for path in LANDMARKS_DIR.glob("clip_*.csv"):
        max_number = max(max_number, clip_number(path.stem))

    return max_number + 1


def clip_number(clip_id: str) -> int:
    try:
        return int(clip_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def open_camera(camera_index: int, fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def open_video_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def draw_prompt(frame, text: str, status: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Target: {text}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        status,
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (180, 230, 255),
        2,
        cv2.LINE_AA,
    )


def wait_for_start(cap: cv2.VideoCapture, target: str) -> bool:
    print()
    print(f"Target sequence: {target}")
    print("Press s in the preview window to start recording, or q to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Webcam frame read failed before recording started")

        draw_prompt(frame, target, "Press s to start. Press q to quit.")
        cv2.imshow("ASL Fingerspelling Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            return True
        if key == ord("q"):
            return False


def record_clip(
    cap: cv2.VideoCapture,
    target: str,
    clip_path: Path,
    fps: float,
) -> tuple[int, float]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = open_video_writer(clip_path, fps, width, height)

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

            now = time.monotonic()
            if now >= next_frame_at:
                writer.write(frame)
                frame_count += 1
                while next_frame_at <= now:
                    next_frame_at += frame_interval

            preview = frame.copy()
            draw_prompt(preview, target, "Recording. Press e to stop.")
            cv2.imshow("ASL Fingerspelling Data Collection", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("e"):
                break
    finally:
        writer.release()

    duration = time.monotonic() - started_at
    return frame_count, duration


def landmark_row(
    frame_index: int,
    sample_index: int,
    timestamp_ms: int,
    group: str,
    coordinate_space: str,
    landmark_index: int,
    landmark,
) -> dict:
    return {
        "frame": frame_index,
        "sample": sample_index,
        "timestamp_ms": timestamp_ms,
        "landmark_group": group,
        "coordinate_space": coordinate_space,
        "index": landmark_index,
        "x": landmark.x,
        "y": landmark.y,
        "z": landmark.z,
        "visibility": getattr(landmark, "visibility", None),
        "presence": getattr(landmark, "presence", None),
    }


def extend_landmark_rows(
    rows: list[dict],
    frame_index: int,
    sample_index: int,
    timestamp_ms: int,
    group: str,
    coordinate_space: str,
    landmarks,
) -> None:
    for landmark_index, landmark in enumerate(landmarks):
        rows.append(
            landmark_row(
                frame_index,
                sample_index,
                timestamp_ms,
                group,
                coordinate_space,
                landmark_index,
                landmark,
            )
        )


def holistic_result_to_csv_rows(
    result,
    frame_index: int,
    sample_index: int,
    timestamp_ms: int,
) -> list[dict]:
    rows: list[dict] = []
    extend_landmark_rows(
        rows,
        frame_index,
        sample_index,
        timestamp_ms,
        "pose_world",
        "world",
        result.pose_world_landmarks,
    )
    extend_landmark_rows(
        rows,
        frame_index,
        sample_index,
        timestamp_ms,
        "pose",
        "normalized",
        result.pose_landmarks,
    )
    extend_landmark_rows(
        rows,
        frame_index,
        sample_index,
        timestamp_ms,
        "left_hand",
        "normalized",
        result.left_hand_landmarks,
    )
    extend_landmark_rows(
        rows,
        frame_index,
        sample_index,
        timestamp_ms,
        "left_hand_world",
        "world",
        result.left_hand_world_landmarks,
    )
    extend_landmark_rows(
        rows,
        frame_index,
        sample_index,
        timestamp_ms,
        "right_hand",
        "normalized",
        result.right_hand_landmarks,
    )
    extend_landmark_rows(
        rows,
        frame_index,
        sample_index,
        timestamp_ms,
        "right_hand_world",
        "world",
        result.right_hand_world_landmarks,
    )
    extend_landmark_rows(
        rows,
        frame_index,
        sample_index,
        timestamp_ms,
        "face",
        "normalized",
        result.face_landmarks,
    )
    return rows


def ensure_holistic_model() -> Path:
    if HOLISTIC_MODEL_PATH.exists():
        return HOLISTIC_MODEL_PATH

    HOLISTIC_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe Holistic model to {HOLISTIC_MODEL_PATH}...")
    try:
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, HOLISTIC_MODEL_PATH)
    except OSError as exc:
        raise RuntimeError(
            f"Could not download the Holistic model from {HOLISTIC_MODEL_URL}. "
            f"Download it manually and save it as {HOLISTIC_MODEL_PATH}."
        ) from exc
    return HOLISTIC_MODEL_PATH


def create_holistic_landmarker() -> vision.HolisticLandmarker:
    model_path = ensure_holistic_model()

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def video_to_landmark_csv(video_path: Path, csv_path: Path, fps: float) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open recorded video for processing: {video_path}")

    rows: list[dict] = []
    frame_interval_ms = int(round(1000.0 / fps))

    with create_holistic_landmarker() as landmarker:
        frame_index = 0
        while True:
            timestamp_ms = frame_index * frame_interval_ms
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect_for_video(image, timestamp_ms)
            rows.extend(
                holistic_result_to_csv_rows(
                    result,
                    frame_index,
                    frame_index,
                    timestamp_ms,
                )
            )
            frame_index += 1

    cap.release()

    with csv_path.open("w", newline="", encoding="utf-8") as landmarks_file:
        writer = csv.DictWriter(landmarks_file, fieldnames=LANDMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    duration = frame_index / fps if frame_index else 0.0
    return frame_index, duration


def append_label(metadata: dict[str, str | int | float]) -> None:
    labels_exist = LABELS_CSV.exists()
    with LABELS_CSV.open("a", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=LABEL_FIELDNAMES)
        if not labels_exist:
            writer.writeheader()
        writer.writerow(metadata)


def prompt_after_recording() -> tuple[str, str]:
    print()
    print("Choose next action:")
    print("  a = accept/save recording and generated landmark CSV")
    print("  r = re-record the same target sequence")
    print("  d = discard and generate a new sequence")
    print("  q = quit")

    while True:
        choice = input("Action [a/r/d/q]: ").strip().lower()
        if choice in {"a", "r", "d", "q"}:
            notes = ""
            if choice == "a":
                notes = input("Notes (optional): ").strip()
            return choice, notes
        print("Please enter a, r, d, or q.")


def cleanup_temp(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        print(f"Warning: could not remove temporary file {path}")


def collect() -> None:
    args = parse_args()
    validate_args(args)
    ensure_dirs()

    cap = open_camera(args.camera, args.fps)
    clip_number_value = next_clip_number()
    target = generate_target(args.min_len, args.max_len)

    try:
        while True:
            should_start = wait_for_start(cap, target)
            if not should_start:
                break

            temp_path = CLIPS_DIR / "_pending_clip.mp4"
            cleanup_temp(temp_path)
            recorded_frames, recorded_duration = record_clip(cap, target, temp_path, args.fps)
            print(
                f"Recorded {recorded_frames} frames "
                f"({recorded_duration:.2f} seconds at target {args.fps:g} FPS)."
            )

            action, notes = prompt_after_recording()
            if action == "r":
                cleanup_temp(temp_path)
                continue
            if action == "d":
                cleanup_temp(temp_path)
                target = generate_target(args.min_len, args.max_len)
                continue
            if action == "q":
                cleanup_temp(temp_path)
                break

            clip_id = f"clip_{clip_number_value:06d}"
            video_path = CLIPS_DIR / f"{clip_id}.mp4"
            landmark_csv_path = LANDMARKS_DIR / f"{clip_id}.csv"
            while video_path.exists() or landmark_csv_path.exists():
                clip_number_value += 1
                clip_id = f"clip_{clip_number_value:06d}"
                video_path = CLIPS_DIR / f"{clip_id}.mp4"
                landmark_csv_path = LANDMARKS_DIR / f"{clip_id}.csv"

            temp_path.replace(video_path)
            num_frames, duration_seconds = video_to_landmark_csv(video_path, landmark_csv_path, args.fps)

            timestamp = datetime.now().isoformat(timespec="seconds")
            append_label(
                {
                    "clip_id": clip_id,
                    "landmark_csv_path": str(landmark_csv_path),
                    "expected_text": target,
                    "video_path": str(video_path),
                    "fps": args.fps,
                    "num_frames": num_frames,
                    "duration_seconds": f"{duration_seconds:.3f}",
                    "signer_id": args.signer_id,
                    "timestamp": timestamp,
                    "notes": notes,
                }
            )

            print(f"Saved {video_path}")
            print(f"Saved {landmark_csv_path}")
            print(f"Updated {LABELS_CSV}")

            clip_number_value += 1
            target = generate_target(args.min_len, args.max_len)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    collect()
