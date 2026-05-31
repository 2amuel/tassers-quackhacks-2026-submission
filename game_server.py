from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
import time
import urllib.request
from collections import deque
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python import BaseOptions, vision


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_FILE = PROJECT_ROOT / "letter-stream-game.html"
PREDICTOR_FILE = PROJECT_ROOT / "neural-network" / "predict_from_csv.py"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "asl_transformer_ctc.pt"
DEFAULT_HOLISTIC_MODEL = PROJECT_ROOT / "src" / "models" / "holistic_landmarker.task"
HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)


def load_predictor_module():
    spec = importlib.util.spec_from_file_location("predict_from_csv", PREDICTOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load predictor file: {PREDICTOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


predictor = load_predictor_module()


def landmarks_to_group(landmarks) -> dict[int, dict[str, float]]:
    if not landmarks:
        return {}
    landmark_iterable = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    return {
        index: {
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z,
            "visibility": getattr(landmark, "visibility", None),
            "presence": getattr(landmark, "presence", None),
        }
        for index, landmark in enumerate(landmark_iterable)
    }


class LiveLetterPredictor:
    def __init__(
        self,
        checkpoint: Path | None,
        holistic_model: Path | None,
        device: str,
        normalize_to_body: bool,
        smoothing: int,
    ) -> None:
        self.device = device
        self.normalize_to_body = normalize_to_body
        self.model = predictor.load_model(checkpoint if checkpoint and checkpoint.exists() else None, device)
        self.features: deque[torch.Tensor] = deque(maxlen=predictor.SEQUENCE_LENGTH)
        self.history: deque[str] = deque(maxlen=max(smoothing, 1))
        self.checkpoint = checkpoint if checkpoint and checkpoint.exists() else None
        self.holistic_model = self.ensure_holistic_model(holistic_model)
        self.holistic = self.create_holistic_landmarker(self.holistic_model)
        self.last_timestamp_ms = -1
        self.started_at = time.monotonic()

    def ensure_holistic_model(self, holistic_model: Path | None) -> Path | None:
        if holistic_model is None:
            return None
        if holistic_model.exists():
            return holistic_model

        holistic_model.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe Holistic model to {holistic_model}...")
        try:
            urllib.request.urlretrieve(HOLISTIC_MODEL_URL, holistic_model)
        except OSError as exc:
            print(f"Warning: could not download {HOLISTIC_MODEL_URL}: {exc}")
            return None
        return holistic_model

    def create_holistic_landmarker(self, holistic_model: Path | None):
        if holistic_model is None:
            return None

        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(holistic_model)),
            running_mode=vision.RunningMode.VIDEO,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )
        return vision.HolisticLandmarker.create_from_options(options)

    @property
    def ready(self) -> bool:
        return len(self.features) >= max(8, predictor.SEQUENCE_LENGTH // 3)

    def close(self) -> None:
        if self.holistic is not None:
            self.holistic.close()

    def predict(self, frame: np.ndarray) -> dict[str, Any]:
        if self.holistic is None:
            raise RuntimeError(
                f"Missing MediaPipe holistic model. Expected {DEFAULT_HOLISTIC_MODEL} "
                "or pass --holistic-model."
            )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.monotonic() - self.started_at) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.holistic.detect_for_video(image, timestamp_ms)
        groups = {
            "pose": landmarks_to_group(result.pose_landmarks),
            "left_hand": landmarks_to_group(result.left_hand_landmarks),
            "right_hand": landmarks_to_group(result.right_hand_landmarks),
            "face": landmarks_to_group(result.face_landmarks),
        }

        feature_row = predictor.sample_to_features(groups, self.normalize_to_body)
        self.features.append(torch.tensor(feature_row, dtype=torch.float32))

        if not self.ready:
            return {
                "letter": None,
                "confidence": 0.0,
                "ready": False,
                "frames": len(self.features),
                    "checkpointLoaded": self.checkpoint is not None,
                    "holisticLoaded": self.holistic is not None,
            }

        window = predictor.padded_window(
            torch.stack(tuple(self.features)),
            len(self.features) - 1,
        ).unsqueeze(0)

        with torch.no_grad():
            log_probs = self.model(window.to(self.device))
            newest_frame_log_probs = log_probs[-1, 0, :]
            probs = torch.exp(newest_frame_log_probs)
            confidence_tensor, token_id_tensor = torch.max(probs, dim=-1)

        token_id = int(token_id_tensor.item())
        token = predictor.OUTPUT_TOKENS[token_id]
        letter = None if token_id == 0 else token
        if letter:
            self.history.append(letter)

        smoothed_letter = max(set(self.history), key=self.history.count) if self.history else letter

        return {
            "letter": smoothed_letter,
            "rawLetter": letter,
            "confidence": float(confidence_tensor.item()),
            "ready": True,
            "frames": len(self.features),
            "checkpointLoaded": self.checkpoint is not None,
            "holisticLoaded": self.holistic is not None,
        }


def decode_data_url(data_url: str) -> np.ndarray:
    _, _, encoded = data_url.partition(",")
    if not encoded:
        encoded = data_url
    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image payload.")
    return frame


class GameRequestHandler(SimpleHTTPRequestHandler):
    predictor: LiveLetterPredictor

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_file(FRONTEND_FILE, "text/html; charset=utf-8")
            return
        if path == "/api/status":
            self.send_json(
                {
                    "ok": True,
                    "model": "neural-network/asl_transformer_ctc",
                    "checkpointLoaded": self.predictor.checkpoint is not None,
                    "holisticLoaded": self.predictor.holistic is not None,
                }
            )
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            frame = decode_data_url(str(payload["image"]))
            prediction = self.predictor.predict(frame)
            prediction["ok"] = True
            prediction["serverTime"] = time.time()
            self.send_json(prediction)
        except Exception as error:
            self.send_json(
                {"ok": False, "error": str(error)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def send_file(self, path: Path, content_type: str) -> None:
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the SignNinja game and prediction API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--holistic-model", type=Path, default=DEFAULT_HOLISTIC_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-body-normalize", action="store_true")
    parser.add_argument("--smoothing", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor_instance = LiveLetterPredictor(
        checkpoint=args.checkpoint,
        holistic_model=args.holistic_model,
        device=args.device,
        normalize_to_body=not args.no_body_normalize,
        smoothing=args.smoothing,
    )
    GameRequestHandler.predictor = predictor_instance

    server = ThreadingHTTPServer((args.host, args.port), GameRequestHandler)
    print(f"Serving SignNinja at http://{args.host}:{args.port}")
    if predictor_instance.checkpoint is None:
        print(
            "Warning: no checkpoint found. Predictions will run with random weights "
            f"until you pass --checkpoint or add {DEFAULT_CHECKPOINT}."
        )
    if predictor_instance.holistic is None:
        print(
            "Warning: no MediaPipe holistic task model found. Prediction requests "
            f"will fail until you pass --holistic-model or add {DEFAULT_HOLISTIC_MODEL}."
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        predictor_instance.close()
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Could not start game server: {exc}", file=sys.stderr)
        raise
