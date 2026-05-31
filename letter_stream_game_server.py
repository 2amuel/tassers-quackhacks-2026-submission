from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import traceback
from collections import deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import cv2
import mediapipe as mp
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFIER_DIR = PROJECT_ROOT / "letter-window-classifier"
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from data import resample_window  # noqa: E402
from live_predict import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    create_holistic_landmarker,
    load_model,
    result_to_left_hand_features as live_result_to_left_hand_features,
    smoothed_prediction,
)
from model import LETTERS  # noqa: E402


class StreamingPredictor:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        confidence_threshold: float,
        smoothing: int,
        normalize_to_body: bool,
        mirror: bool,
        frame_width: int,
        frame_height: int,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.normalize_to_body = normalize_to_body
        self.mirror = mirror
        self.frame_size = (frame_width, frame_height)
        self.model = None
        self.window_frames = 48
        self.landmarker = None
        self.frame_features: deque[torch.Tensor] = deque(maxlen=self.window_frames)
        self.prediction_history: deque[tuple[str, float]] = deque(maxlen=max(smoothing, 1))
        self.started_at = time.monotonic()
        self.last_timestamp_ms = -1
        self.error: str | None = None

        try:
            self.model, self.window_frames = load_model(checkpoint, device)
            self.frame_features = deque(maxlen=self.window_frames)
            self.landmarker = create_holistic_landmarker()
        except Exception as exc:
            self.error = str(exc)

    @property
    def ok(self) -> bool:
        return self.model is not None and self.landmarker is not None and self.error is None

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()

    def status(self) -> dict:
        return {
            "ok": self.ok,
            "checkpointLoaded": self.model is not None,
            "windowFrames": self.window_frames,
            "error": self.error,
        }

    def next_timestamp_ms(self) -> int:
        timestamp_ms = int((time.monotonic() - self.started_at) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def predict_data_url(self, image_data_url: str) -> dict:
        if not self.ok:
            return {"ok": False, "error": self.error or "Predictor is not ready."}

        frame = decode_image_data_url(image_data_url)
        frame = cv2.resize(frame, self.frame_size, interpolation=cv2.INTER_AREA)
        if self.mirror:
            frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(image, self.next_timestamp_ms())
        self.frame_features.append(
            result_to_left_hand_features(
                result,
                normalize_to_body=self.normalize_to_body,
            )
        )

        enough_context = len(self.frame_features) >= max(2, self.window_frames // 3)
        if not enough_context:
            return {
                "ok": True,
                "ready": False,
                "letter": "",
                "confidence": 0.0,
                "frames": len(self.frame_features),
            }

        features = torch.stack(list(self.frame_features))
        window = resample_window(features, self.window_frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(window)[0], dim=-1)
        confidence_tensor, class_id_tensor = torch.max(probs, dim=-1)
        raw_letter = LETTERS[int(class_id_tensor.item())]
        raw_confidence = float(confidence_tensor.item())
        self.prediction_history.append((raw_letter, raw_confidence))
        letter, confidence = smoothed_prediction(self.prediction_history)

        return {
            "ok": True,
            "ready": True,
            "letter": letter if confidence >= self.confidence_threshold else "",
            "rawLetter": raw_letter,
            "confidence": confidence,
            "rawConfidence": raw_confidence,
            "frames": len(self.frame_features),
        }


def decode_image_data_url(image_data_url: str) -> np.ndarray:
    if "," in image_data_url:
        _header, encoded = image_data_url.split(",", maxsplit=1)
    else:
        encoded = image_data_url

    image_bytes = base64.b64decode(encoded)
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image data.")
    return frame


def result_to_left_hand_features(result, normalize_to_body: bool) -> torch.Tensor:
    """Use live_predict's feature path, but tolerate absent landmark groups."""
    safe_result = SimpleNamespace(
        pose_landmarks=getattr(result, "pose_landmarks", None) or [],
        left_hand_landmarks=getattr(result, "left_hand_landmarks", None) or [],
        right_hand_landmarks=getattr(result, "right_hand_landmarks", None) or [],
        face_landmarks=getattr(result, "face_landmarks", None) or [],
    )
    return live_result_to_left_hand_features(
        safe_result,
        normalize_to_body=normalize_to_body,
    )


def make_handler(predictor_instance: StreamingPredictor):
    class GameHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PROJECT_ROOT), **kwargs)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/status":
                self.send_json(predictor_instance.status())
                return
            if path == "/":
                self.path = "/letter-stream-game.html"
            super().do_GET()

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path != "/api/predict":
                self.send_error(404)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length))
                image = payload.get("image")
                if not image:
                    raise ValueError("Missing image field.")
                self.send_json(predictor_instance.predict_data_url(image))
            except Exception as exc:
                traceback.print_exc()
                self.send_json(
                    {
                        "ok": False,
                        "error": str(exc),
                        "errorType": type(exc).__name__,
                    },
                    status=200,
                )

        def send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return GameHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve SignNinja with streaming ASL prediction.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--no-body-normalize", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor_instance = StreamingPredictor(
        checkpoint=args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        smoothing=args.smoothing,
        normalize_to_body=not args.no_body_normalize,
        mirror=not args.no_mirror,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(predictor_instance))
    url = f"http://{args.host}:{args.port}/"
    print(f"Serving SignNinja at {url}")
    if predictor_instance.error:
        print(f"Prediction backend is not ready: {predictor_instance.error}")
    try:
        server.serve_forever()
    finally:
        predictor_instance.close()
        server.server_close()


if __name__ == "__main__":
    main()
