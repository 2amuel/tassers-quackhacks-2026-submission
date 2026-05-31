"""Record webcam video and save MediaPipe Holistic landmark data."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))
os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision


HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
DEFAULT_HOLISTIC_MODEL_PATH = Path("models/holistic_landmarker.task")

POSE_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)

HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


@dataclass
class CameraInfo:
    index: int
    name: str


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
        description="Record webcam footage and MediaPipe Holistic landmark CSV data."
    )
    parser.add_argument("--camera", type=int, default=None, help="OpenCV camera index.")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional maximum runtime in seconds. By default, run until q or Ctrl-C.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        dest="max_duration",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.05,
        help="Seconds between sampled images used for MediaPipe CSV output.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested camera frame width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera frame height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested output frames per second.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to outputs/recording-YYYYmmdd-HHMMSS.",
    )
    parser.add_argument(
        "--holistic-model-path",
        type=Path,
        default=DEFAULT_HOLISTIC_MODEL_PATH,
        help="Path to a MediaPipe Holistic Landmarker .task model.",
    )
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
    parser.add_argument("--filename", default="webcam_capture.mp4", help="Output video filename.")
    parser.add_argument("--no-save-video", action="store_true", help="Do not save the continuous feed video.")
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Do not save sampled image frames alongside the CSV.",
    )
    parser.add_argument("--no-preview", action="store_true", help="Disable the live preview window.")
    parser.add_argument("--no-mirror", action="store_true", help="Do not mirror webcam frames.")
    parser.add_argument("--list-cameras", action="store_true", help="Print detected cameras and exit.")
    args = parser.parse_args()
    if args.sample_interval <= 0:
        parser.error("--sample-interval must be greater than 0")
    return args


def detected_camera_names() -> list[str]:
    if sys.platform != "darwin":
        return []

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []

    return [item["_name"] for item in payload.get("SPCameraDataType", []) if item.get("_name")]


def detected_cameras() -> list[CameraInfo]:
    return [CameraInfo(index=index, name=name) for index, name in enumerate(detected_camera_names())]


def choose_camera(args: argparse.Namespace) -> CameraInfo:
    cameras = detected_cameras()

    if args.list_cameras:
        if cameras:
            for camera in cameras:
                print(f"{camera.index}: {camera.name}")
        else:
            print("No named cameras detected. Try --camera 0, or check camera permissions.")
        raise SystemExit(0)

    if args.camera is not None:
        name = cameras[args.camera].name if 0 <= args.camera < len(cameras) else f"OpenCV camera {args.camera}"
        return CameraInfo(index=args.camera, name=name)

    if cameras:
        return cameras[0]
    return CameraInfo(index=0, name="OpenCV camera 0")


def camera_backend() -> int:
    if sys.platform == "darwin":
        return cv2.CAP_AVFOUNDATION
    return cv2.CAP_ANY


def make_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs") / f"recording-{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def ensure_holistic_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe Holistic model to {model_path}...")
    try:
        urllib.request.urlretrieve(HOLISTIC_MODEL_URL, model_path)
    except OSError as exc:
        raise RuntimeError(
            f"Could not download the Holistic model from {HOLISTIC_MODEL_URL}. "
            f"Download it manually and pass --holistic-model-path {model_path}."
        ) from exc
    return model_path


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def create_holistic_landmarker(
    model_path: Path,
    args: argparse.Namespace,
) -> vision.HolisticLandmarker:
    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=args.min_face_detection_confidence,
        min_face_landmarks_confidence=args.min_face_landmarks_confidence,
        min_pose_detection_confidence=args.min_pose_detection_confidence,
        min_pose_landmarks_confidence=args.min_pose_landmarks_confidence,
        min_hand_landmarks_confidence=args.min_hand_landmarks_confidence,
    )
    return vision.HolisticLandmarker.create_from_options(options)


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


def write_landmark_csv(csv_path: Path, rows: list[dict]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LANDMARK_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def rows_for_group(rows: list[dict], group: str) -> list[dict]:
    return [row for row in rows if row["landmark_group"] == group]


def indexed_points(rows: list[dict]) -> dict[int, dict]:
    return {int(row["index"]): row for row in rows}


def plot_connections_2d(ax, rows: list[dict], connections: tuple[tuple[int, int], ...], color: str) -> None:
    points = indexed_points(rows)
    for start_index, end_index in connections:
        start = points.get(start_index)
        end = points.get(end_index)
        if start is not None and end is not None:
            ax.plot([start["x"], end["x"]], [start["y"], end["y"]], color=color, linewidth=1.8)


def scatter_2d(ax, rows: list[dict], color: str, label: str, size: int = 18, alpha: float = 0.9) -> None:
    if not rows:
        return
    ax.scatter(
        [row["x"] for row in rows],
        [row["y"] for row in rows],
        color=color,
        s=size,
        alpha=alpha,
        label=label,
    )


def plot_pose_world_3d(ax, rows: list[dict]) -> None:
    points = indexed_points(rows)
    for start_index, end_index in POSE_CONNECTIONS:
        start = points.get(start_index)
        end = points.get(end_index)
        if start is not None and end is not None:
            ax.plot(
                [start["x"], end["x"]],
                [start["z"], end["z"]],
                [-start["y"], -end["y"]],
                color="#2563eb",
                linewidth=1.8,
            )

    if rows:
        ax.scatter(
            [row["x"] for row in rows],
            [row["z"] for row in rows],
            [-row["y"] for row in rows],
            color="#10b981",
            s=24,
        )

    ax.set_title("Pose world landmarks")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")


def save_landmark_visualization(plot_path: Path, rows: list[dict]) -> None:
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No landmarks detected", ha="center", va="center")
        ax.axis("off")
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        return

    frame_ids = sorted({int(row["frame"]) for row in rows})
    frame_id = max(frame_ids, key=lambda value: sum(int(row["frame"]) == value for row in rows))
    frame_rows = [row for row in rows if int(row["frame"]) == frame_id]

    pose_rows = rows_for_group(frame_rows, "pose")
    pose_world_rows = rows_for_group(frame_rows, "pose_world")
    left_hand_rows = rows_for_group(frame_rows, "left_hand")
    right_hand_rows = rows_for_group(frame_rows, "right_hand")
    face_rows = rows_for_group(frame_rows, "face")

    fig = plt.figure(figsize=(13, 6))
    ax_image = fig.add_subplot(1, 2, 1)
    ax_world = fig.add_subplot(1, 2, 2, projection="3d")

    ax_image.set_title(f"Normalized landmarks, frame {frame_id}")
    ax_image.set_xlim(0, 1)
    ax_image.set_ylim(1, 0)
    ax_image.set_aspect("equal", adjustable="box")
    ax_image.grid(True, alpha=0.2)
    ax_image.set_xlabel("x")
    ax_image.set_ylabel("y")

    scatter_2d(ax_image, face_rows, "#64748b", "face", size=5, alpha=0.35)
    plot_connections_2d(ax_image, pose_rows, POSE_CONNECTIONS, "#2563eb")
    plot_connections_2d(ax_image, left_hand_rows, HAND_CONNECTIONS, "#f59e0b")
    plot_connections_2d(ax_image, right_hand_rows, HAND_CONNECTIONS, "#ec4899")
    scatter_2d(ax_image, pose_rows, "#10b981", "pose", size=18)
    scatter_2d(ax_image, left_hand_rows, "#f59e0b", "left hand", size=18)
    scatter_2d(ax_image, right_hand_rows, "#ec4899", "right hand", size=18)
    ax_image.legend(loc="upper right")

    plot_pose_world_3d(ax_world, pose_world_rows)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def record() -> Path:
    args = parse_args()
    camera = choose_camera(args)
    output_dir = make_output_dir(args.output_dir)
    output_path = output_dir / args.filename
    csv_path = output_dir / "holistic_landmarks.csv"
    visualization_path = output_dir / "landmark_visualization.png"
    sampled_frames_dir = output_dir / "sampled_frames"
    holistic_model_path = ensure_holistic_model(args.holistic_model_path)
    if not args.no_save_images:
        sampled_frames_dir.mkdir(exist_ok=True)

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

    writer = None if args.no_save_video else open_writer(output_path, fps, width, height)
    start_time = time.monotonic()
    frame_count = 0
    sample_count = 0
    next_sample_time = 0.0
    landmark_rows: list[dict] = []

    print(f"Recording camera {camera.index}: {camera.name}")
    print("Press q in the preview window, or Ctrl-C in the terminal, to stop.")
    if writer is not None:
        print(f"Saving video feed to {output_path}")
    if not args.no_save_images:
        print(f"Saving sampled images to {sampled_frames_dir}")
    print(f"Writing MediaPipe Holistic landmarks live to {csv_path}")

    try:
        with (
            csv_path.open("w", newline="", encoding="utf-8") as csv_file,
            create_holistic_landmarker(holistic_model_path, args) as landmarker,
        ):
            csv_writer = csv.DictWriter(csv_file, fieldnames=LANDMARK_FIELDNAMES)
            csv_writer.writeheader()
            csv_file.flush()

            while True:
                elapsed = time.monotonic() - start_time
                if args.max_duration is not None and elapsed >= args.max_duration:
                    break

                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed; stopping recording.")
                    break

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                if writer is not None:
                    writer.write(frame)

                if elapsed >= next_sample_time:
                    timestamp_ms = int(elapsed * 1000)
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    sampled_rows = holistic_result_to_csv_rows(
                        result,
                        frame_count,
                        sample_count,
                        timestamp_ms,
                    )
                    csv_writer.writerows(sampled_rows)
                    csv_file.flush()
                    landmark_rows.extend(sampled_rows)

                    if not args.no_save_images:
                        sample_path = sampled_frames_dir / f"sample_{sample_count:06d}.jpg"
                        cv2.imwrite(str(sample_path), frame)

                    sample_count += 1
                    next_sample_time += args.sample_interval
                    while next_sample_time <= elapsed:
                        next_sample_time += args.sample_interval

                frame_count += 1

                if not args.no_preview:
                    cv2.imshow("Webcam Recorder + Holistic Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    except KeyboardInterrupt:
        print("Stopping capture.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    save_landmark_visualization(visualization_path, landmark_rows)

    metadata = {
        "camera": asdict(camera),
        "runtime_seconds": time.monotonic() - start_time,
        "max_duration_seconds": args.max_duration,
        "sample_interval_seconds": args.sample_interval,
        "frame_count": frame_count,
        "sample_count": sample_count,
        "fps": fps,
        "width": width,
        "height": height,
        "video_path": str(output_path) if writer is not None else None,
        "sampled_frames_dir": str(sampled_frames_dir) if not args.no_save_images else None,
        "landmark_csv_path": str(csv_path),
        "landmark_visualization_path": str(visualization_path),
        "holistic_model_path": str(holistic_model_path),
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
    with (output_dir / "recording_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    if writer is not None:
        print(f"Wrote {frame_count} video frames to {output_path}")
    print(f"Processed {sample_count} sampled images")
    print(f"Wrote {len(landmark_rows)} landmark rows to {csv_path}")
    print(f"Wrote landmark visualization to {visualization_path}")
    return output_path


if __name__ == "__main__":
    record()
