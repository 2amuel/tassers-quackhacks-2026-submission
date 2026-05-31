"""Visualize MediaPipe Holistic landmarks stored in a recording CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 10.0

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render pose/hand/face landmarks from a MediaPipe Holistic CSV."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to holistic_landmarks.csv or to a recording output folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image/video path. Defaults to <csv folder>/landmark_csv_visualization.mp4.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Canvas width.")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Canvas height.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Output video FPS. Defaults to metadata/sample interval when available.",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Write one PNG frame instead of an MP4 video.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample index to render when using --image. Defaults to the middle sample.",
    )
    parser.add_argument(
        "--blank-background",
        action="store_true",
        help="Use a blank background even when sampled frame images exist.",
    )
    parser.add_argument(
        "--draw-face-contours",
        action="store_true",
        help="Accepted for CLI compatibility; face landmarks are drawn as small points.",
    )
    return parser.parse_args()


def resolve_csv_path(input_path: Path) -> Path:
    if input_path.is_dir():
        csv_path = input_path / "holistic_landmarks.csv"
    else:
        csv_path = input_path

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find landmark CSV: {csv_path}")
    return csv_path


def read_metadata(csv_path: Path) -> dict:
    metadata_path = csv_path.parent / "recording_metadata.json"
    if not metadata_path.exists():
        return {}

    with metadata_path.open(encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def output_fps(args: argparse.Namespace, metadata: dict) -> float:
    if args.fps is not None:
        return args.fps

    sample_interval = metadata.get("sample_interval_seconds")
    if sample_interval:
        return 1.0 / float(sample_interval)

    return DEFAULT_FPS


def read_csv(
    csv_path: Path,
) -> dict[int, dict[str, dict[int, dict[str, float]]]]:
    samples: dict[int, dict[str, dict[int, dict[str, float]]]] = {}

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row["coordinate_space"] != "normalized":
                continue

            sample = int(row["sample"])
            group = row["landmark_group"]
            index = int(row["index"])
            samples.setdefault(sample, {}).setdefault(group, {})[index] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "visibility": parse_optional_float(row.get("visibility")),
                "presence": parse_optional_float(row.get("presence")),
            }

    if not samples:
        raise RuntimeError(f"No normalized landmark rows found in {csv_path}")
    return samples


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def blank_canvas(width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    grid_color = (225, 225, 225)
    for x in range(0, width, max(width // 8, 1)):
        cv2.line(canvas, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, max(height // 8, 1)):
        cv2.line(canvas, (0, y), (width, y), grid_color, 1)
    return canvas


def pixel_point(
    landmark: dict[str, float],
    width: int,
    height: int,
) -> tuple[int, int]:
    return (
        int(round(landmark["x"] * width)),
        int(round(landmark["y"] * height)),
    )


def draw_connections(
    image: np.ndarray,
    landmarks: dict[int, dict[str, float]],
    connections,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    height, width = image.shape[:2]
    for start, end in connections:
        start_point = landmarks.get(start)
        end_point = landmarks.get(end)
        if start_point is None or end_point is None:
            continue
        cv2.line(
            image,
            pixel_point(start_point, width, height),
            pixel_point(end_point, width, height),
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_points(
    image: np.ndarray,
    landmarks: dict[int, dict[str, float]],
    color: tuple[int, int, int],
    radius: int,
) -> None:
    height, width = image.shape[:2]
    for landmark in landmarks.values():
        cv2.circle(
            image,
            pixel_point(landmark, width, height),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


def load_background(
    csv_path: Path,
    sample: int,
    width: int,
    height: int,
    blank_background: bool,
) -> np.ndarray:
    if not blank_background:
        sample_path = csv_path.parent / "sampled_frames" / f"sample_{sample:06d}.jpg"
        image = cv2.imread(str(sample_path))
        if image is not None:
            return cv2.resize(image, (width, height))

    return blank_canvas(width, height)


def draw_sample(
    canvas: np.ndarray,
    groups: dict[str, dict[int, dict[str, float]]],
    sample: int,
    draw_face_contours: bool,
) -> np.ndarray:
    image = canvas.copy()

    face = groups.get("face", {})
    pose = groups.get("pose", {})
    left_hand = groups.get("left_hand", {})
    right_hand = groups.get("right_hand", {})

    if face:
        draw_points(image, face, (120, 120, 120), 1)

    if pose:
        draw_connections(image, pose, POSE_CONNECTIONS, (245, 117, 66), 3)
        draw_points(image, pose, (66, 135, 245), 4)

    if left_hand:
        draw_connections(image, left_hand, HAND_CONNECTIONS, (0, 180, 255), 2)
        draw_points(image, left_hand, (0, 140, 255), 4)

    if right_hand:
        draw_connections(image, right_hand, HAND_CONNECTIONS, (255, 70, 180), 2)
        draw_points(image, right_hand, (255, 40, 140), 4)

    cv2.putText(
        image,
        f"sample {sample}",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return image


def write_image(args: argparse.Namespace, samples: dict[int, dict[str, dict[int, dict[str, float]]]]) -> Path:
    sample_ids = sorted(samples)
    sample = args.sample if args.sample is not None else sample_ids[len(sample_ids) // 2]
    if sample not in samples:
        raise RuntimeError(f"Sample {sample} was not found in {args.csv_path}")

    output = args.output or args.csv_path.parent / f"landmark_sample_{sample:06d}.png"
    background = load_background(
        args.csv_path,
        sample,
        args.width,
        args.height,
        args.blank_background,
    )
    image = draw_sample(background, samples[sample], sample, args.draw_face_contours)
    cv2.imwrite(str(output), image)
    return output


def write_video(args: argparse.Namespace, samples: dict[int, dict[str, dict[int, dict[str, float]]]]) -> Path:
    output = args.output or args.csv_path.parent / "landmark_csv_visualization.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)

    fps = output_fps(args, args.metadata)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")

    try:
        for sample in sorted(samples):
            background = load_background(
                args.csv_path,
                sample,
                args.width,
                args.height,
                args.blank_background,
            )
            writer.write(
                draw_sample(background, samples[sample], sample, args.draw_face_contours)
            )
    finally:
        writer.release()

    return output


def main() -> None:
    args = parse_args()
    args.csv_path = resolve_csv_path(args.input_path)
    args.metadata = read_metadata(args.csv_path)
    args.width = int(args.metadata.get("width") or args.width)
    args.height = int(args.metadata.get("height") or args.height)
    samples = read_csv(args.csv_path)

    if args.image:
        output = write_image(args, samples)
    else:
        output = write_video(args, samples)

    print(f"Wrote visualization to {output}")


if __name__ == "__main__":
    main()
