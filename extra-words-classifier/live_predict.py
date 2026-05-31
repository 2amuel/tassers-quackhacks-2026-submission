from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks.python import BaseOptions, vision

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data import (
    FEATURE_NORMALIZATION,
    INPUT_FEATURES_PER_FRAME,
    model_input_features,
    predictor,
    resample_window,
)
from model import OUTPUT_LABELS, create_model


PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
FALLBACK_CHECKPOINT = MODELS_DIR / "extra_words_left_hand_torso_face_sequences.pt"
HOLISTIC_MODEL_PATH = PROJECT_ROOT / "src" / "models" / "holistic_landmarker.task"


def newest_extra_words_checkpoint() -> Path:
    checkpoints = sorted(
        MODELS_DIR.glob("extra_words_left_hand_torso_face*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else FALLBACK_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the expanded landmark-window classifier on live webcam video."
    )
    parser.add_argument("--checkpoint", type=Path, default=newest_extra_words_checkpoint())
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--smoothing", type=int, default=5, help="Number of recent predictions to vote over.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-body-normalize", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: str) -> tuple[torch.nn.Module, int]:
    if not checkpoint.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {checkpoint}")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    window_frames = int(payload["window_frames"])
    input_dim = int(payload.get("input_features_per_frame", INPUT_FEATURES_PER_FRAME))
    if input_dim != INPUT_FEATURES_PER_FRAME:
        raise RuntimeError(
            f"Checkpoint expects {input_dim} features per frame, "
            f"but live expanded features provide {INPUT_FEATURES_PER_FRAME}."
        )
    checkpoint_feature_normalization = payload.get("feature_normalization")
    if checkpoint_feature_normalization != FEATURE_NORMALIZATION:
        raise RuntimeError(
            "Checkpoint uses a different feature normalization. "
            "Retrain with extra-words-classifier/train.py --reset, or pass a newer checkpoint."
        )

    model = create_model(INPUT_FEATURES_PER_FRAME, window_frames).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, window_frames


def create_holistic_landmarker():
    if not HOLISTIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Could not find MediaPipe model: {HOLISTIC_MODEL_PATH}. "
            "Run a collector once or copy holistic_landmarker.task there."
        )

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HOLISTIC_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def landmarks_to_group(landmarks) -> dict[int, dict[str, float]]:
    return {
        index: {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": getattr(landmark, "visibility", None),
            "presence": getattr(landmark, "presence", None),
        }
        for index, landmark in enumerate(landmarks)
    }


def result_to_model_features(result, normalize_to_body: bool) -> torch.Tensor:
    groups = {
        "pose": landmarks_to_group(result.pose_landmarks),
        "left_hand": landmarks_to_group(result.left_hand_landmarks),
        "right_hand": landmarks_to_group(result.right_hand_landmarks),
        "face": landmarks_to_group(result.face_landmarks),
    }
    features = torch.tensor(
        [predictor.sample_to_features(groups, normalize_to_body=normalize_to_body)],
        dtype=torch.float32,
    )
    return model_input_features(features).squeeze(0)


def draw_prediction(
    frame,
    label: str,
    confidence: float,
    enough_context: bool,
    threshold: float,
) -> None:
    display_label = label if confidence >= threshold and enough_context else "-"
    status = "collecting context" if not enough_context else f"{confidence:.0%}"

    cv2.rectangle(frame, (0, 0), (frame.shape[1], 112), (0, 0, 0), -1)
    cv2.putText(
        frame,
        f"Current label: {display_label}",
        (24, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {status}    q quit",
        (24, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (180, 230, 255),
        2,
        cv2.LINE_AA,
    )


def smoothed_prediction(history: deque[tuple[str, float]]) -> tuple[str, float]:
    if not history:
        return "-", 0.0

    votes: dict[str, list[float]] = {}
    for label, confidence in history:
        votes.setdefault(label, []).append(confidence)

    best_label = max(votes, key=lambda label: (len(votes[label]), sum(votes[label])))
    confidences = votes[best_label]
    return best_label, sum(confidences) / len(confidences)


def main() -> None:
    args = parse_args()
    model, window_frames = load_model(args.checkpoint, args.device)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at camera index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    frame_features: deque[torch.Tensor] = deque(maxlen=window_frames)
    prediction_history: deque[tuple[str, float]] = deque(maxlen=max(args.smoothing, 1))
    started_at = time.monotonic()
    last_timestamp_ms = -1

    try:
        with create_holistic_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed; stopping.")
                    break

                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                timestamp_ms = int((time.monotonic() - started_at) * 1000)
                if timestamp_ms <= last_timestamp_ms:
                    timestamp_ms = last_timestamp_ms + 1
                last_timestamp_ms = timestamp_ms

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(image, timestamp_ms)
                frame_features.append(
                    result_to_model_features(
                        result,
                        normalize_to_body=not args.no_body_normalize,
                    )
                )

                label = "-"
                confidence = 0.0
                enough_context = len(frame_features) >= max(2, window_frames // 3)
                if enough_context:
                    features = torch.stack(list(frame_features))
                    window = resample_window(features, window_frames).unsqueeze(0).to(args.device)
                    with torch.no_grad():
                        probs = torch.softmax(model(window)[0], dim=-1)
                    confidence_tensor, class_id_tensor = torch.max(probs, dim=-1)
                    prediction_history.append(
                        (
                            OUTPUT_LABELS[int(class_id_tensor.item())],
                            float(confidence_tensor.item()),
                        )
                    )
                    label, confidence = smoothed_prediction(prediction_history)

                draw_prediction(
                    frame,
                    label,
                    confidence,
                    enough_context,
                    args.confidence_threshold,
                )
                cv2.imshow("Live Expanded ASL Classifier", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
