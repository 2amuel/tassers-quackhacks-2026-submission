from __future__ import annotations

import argparse
import platform
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import torch

try:
    import pygame
except ImportError as exc:  # pragma: no cover - friendly runtime message
    raise SystemExit(
        "pygame is required for the desktop game. Install it with: pip install pygame"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
CLASSIFIER_DIR = PROJECT_ROOT / "letter-window-classifier"
if str(CLASSIFIER_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_DIR))

from data import resample_window  # noqa: E402
from live_predict import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    create_holistic_landmarker,
    load_model,
    result_to_left_hand_features,
    smoothed_prediction,
)
from model import LETTERS  # noqa: E402


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VISIBLE_COUNT = 5
MAIN_INDEX = 2
MAX_HEALTH = 3
TIMER_DURATION = 10.0


COLORS = {
    "bg": (47, 52, 50),
    "panel": (23, 27, 25),
    "panel_strong": (38, 43, 40),
    "ink": (246, 244, 236),
    "muted": (197, 203, 194),
    "accent": (184, 240, 106),
    "accent_2": (230, 208, 111),
    "danger": (216, 93, 79),
    "camera_bg": (16, 21, 14),
    "border": (83, 91, 85),
}


@dataclass
class StreamEntry:
    letter: str
    cracked_until: float = 0.0
    crack_angle: int = 0


class LiveLetterPredictor:
    def __init__(
        self,
        checkpoint: Path,
        device: str,
        confidence_threshold: float,
        smoothing: int,
        normalize_to_body: bool,
    ) -> None:
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.normalize_to_body = normalize_to_body
        self.model, self.window_frames = load_model(checkpoint, device)
        self.landmarker = create_holistic_landmarker()
        self.frame_features: deque[torch.Tensor] = deque(maxlen=self.window_frames)
        self.prediction_history: deque[tuple[str, float]] = deque(maxlen=max(smoothing, 1))
        self.started_at = time.monotonic()
        self.last_timestamp_ms = -1
        self.raw_letter = "-"
        self.letter = "-"
        self.confidence = 0.0
        self.ready = False

    def close(self) -> None:
        self.landmarker.close()

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int((time.monotonic() - self.started_at) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def update(self, frame_bgr) -> tuple[str, float, bool]:
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(image, self._next_timestamp_ms())
        self.frame_features.append(
            result_to_left_hand_features(
                result,
                normalize_to_body=self.normalize_to_body,
            )
        )

        self.ready = len(self.frame_features) >= max(2, self.window_frames // 3)
        if not self.ready:
            self.letter = "-"
            self.confidence = 0.0
            return self.letter, self.confidence, self.ready

        features = torch.stack(list(self.frame_features))
        window = resample_window(features, self.window_frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(window)[0], dim=-1)
        confidence_tensor, class_id_tensor = torch.max(probs, dim=-1)
        self.raw_letter = LETTERS[int(class_id_tensor.item())]
        raw_confidence = float(confidence_tensor.item())
        self.prediction_history.append((self.raw_letter, raw_confidence))
        self.letter, self.confidence = smoothed_prediction(self.prediction_history)
        return self.letter, self.confidence, self.ready

    @property
    def accepted_letter(self) -> str:
        if self.ready and self.confidence >= self.confidence_threshold:
            return self.letter
        return "-"


class SignNinjaApp:
    def __init__(self, args: argparse.Namespace) -> None:
        pygame.init()
        pygame.display.set_caption("SignNinja")
        self.args = args
        self.screen = pygame.display.set_mode((args.window_width, args.window_height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.fonts = {
            "title": pygame.font.SysFont("segoeui", 42, bold=True),
            "large": pygame.font.SysFont("segoeui", 76, bold=True),
            "score": pygame.font.SysFont("segoeui", 34, bold=True),
            "body": pygame.font.SysFont("segoeui", 22, bold=True),
            "small": pygame.font.SysFont("segoeui", 16, bold=True),
        }

        self.cap = self.open_camera()

        self.predictor = LiveLetterPredictor(
            checkpoint=args.checkpoint,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            smoothing=args.smoothing,
            normalize_to_body=not args.no_body_normalize,
        )

        self.frame_bgr = None
        self.camera_error = ""
        self.prediction_error = ""
        self.stream: list[StreamEntry] = []
        self.score = 0
        self.streak = 0
        self.health = MAX_HEALTH
        self.time_left = TIMER_DURATION
        self.state = "ready"
        self.started_at: float | None = None
        self.message = "Press Enter to begin."
        self.prepare_game()

    def open_camera(self) -> cv2.VideoCapture:
        backends = [cv2.CAP_ANY]
        if platform.system() == "Windows":
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

        last_error = f"Could not open webcam at camera index {self.args.camera}"
        for backend in backends:
            cap = cv2.VideoCapture(self.args.camera, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
            cap.set(cv2.CAP_PROP_FPS, self.args.fps)
            ok, _frame = cap.read()
            if ok:
                return cap

            last_error = f"Camera opened with backend {backend}, but frame reads failed."
            cap.release()

        raise RuntimeError(last_error)

    def close(self) -> None:
        self.predictor.close()
        self.cap.release()
        pygame.quit()

    def random_entry(self) -> StreamEntry:
        return StreamEntry(letter=random.choice(ALPHABET))

    def build_stream(self) -> None:
        self.stream = [self.random_entry() for _ in range(8)]

    def ensure_stream_length(self) -> None:
        while len(self.stream) < VISIBLE_COUNT + 2:
            self.stream.append(self.random_entry())

    def prepare_game(self) -> None:
        self.score = 0
        self.streak = 0
        self.health = MAX_HEALTH
        self.time_left = TIMER_DURATION
        self.state = "ready"
        self.started_at = None
        self.message = "Press Enter to begin."
        self.build_stream()

    def start_game(self) -> None:
        self.score = 0
        self.streak = 0
        self.health = MAX_HEALTH
        self.time_left = TIMER_DURATION
        self.state = "playing"
        self.started_at = time.monotonic()
        self.message = "Go."
        self.build_stream()

    def end_game(self) -> None:
        self.state = "game-over"
        self.message = f"Game over. Final score: {self.score}. Press Enter to play again."

    def current_target(self) -> str:
        self.ensure_stream_length()
        return self.stream[MAIN_INDEX].letter

    def advance_stream(self, manual: bool = False) -> None:
        if self.state != "playing":
            return

        self.score += 1
        self.streak += 1
        self.time_left = TIMER_DURATION
        self.stream[MAIN_INDEX].cracked_until = time.monotonic() + 0.18
        self.stream[MAIN_INDEX].crack_angle = random.randint(-70, 70)
        self.message = "Manual score." if manual else "Nice. Keep the stream moving."
        self.stream.pop(0)
        self.stream.append(self.random_entry())

    def take_damage(self) -> None:
        self.health = max(0, self.health - 1)
        self.streak = 0
        self.time_left = TIMER_DURATION
        if self.health == 0:
            self.end_game()
        else:
            self.message = "Time ran out. One health point lost."

    def update_game(self, dt: float) -> None:
        if self.state != "playing":
            return

        self.time_left = max(0.0, self.time_left - dt)
        if self.time_left <= 0:
            self.take_damage()
            return

        predicted_letter = self.predictor.accepted_letter
        if predicted_letter == self.current_target():
            self.advance_stream()
        elif predicted_letter != "-":
            self.message = f"Prediction sees {predicted_letter}. Target: {self.current_target()}."
        elif not self.predictor.ready:
            self.message = "Collecting sign context..."

    def process_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if event.key in (pygame.K_RETURN, pygame.K_r):
                    self.start_game()
                if event.key == pygame.K_SPACE:
                    self.advance_stream(manual=True)
        return True

    def update_camera_and_prediction(self) -> None:
        ok, frame = self.cap.read()
        if not ok:
            self.camera_error = "Camera frame read failed."
            return

        if not self.args.no_mirror:
            frame = cv2.flip(frame, 1)
        self.frame_bgr = frame
        self.camera_error = ""

        try:
            self.predictor.update(frame)
            self.prediction_error = ""
        except Exception as exc:
            self.prediction_error = str(exc)
            if self.state == "playing":
                self.message = "Prediction failed, but camera is still running."

    def signs_per_minute(self) -> int:
        if not self.started_at or self.score == 0:
            return 0
        elapsed_minutes = max((time.monotonic() - self.started_at) / 60.0, 1.0 / 60.0)
        return round(self.score / elapsed_minutes)

    def draw_text(
        self,
        text: str,
        font_key: str,
        color: tuple[int, int, int],
        pos: tuple[int, int],
        center: bool = False,
    ) -> pygame.Rect:
        surface = self.fonts[font_key].render(text, True, color)
        rect = surface.get_rect()
        if center:
            rect.center = pos
        else:
            rect.topleft = pos
        self.screen.blit(surface, rect)
        return rect

    def draw_scorebox(self, label: str, value: str, rect: pygame.Rect, value_color=None) -> None:
        pygame.draw.rect(self.screen, COLORS["panel"], rect, border_radius=8)
        pygame.draw.rect(self.screen, COLORS["border"], rect, 1, border_radius=8)
        self.draw_text(label.upper(), "small", COLORS["muted"], (rect.x + 14, rect.y + 10))
        self.draw_text(value, "score", value_color or COLORS["ink"], (rect.x + 14, rect.y + 31))

    def draw_header(self, width: int) -> None:
        self.draw_text("SignNinja", "title", COLORS["ink"], (32, 22))
        self.draw_text("Sign the center letter to move forward.", "body", COLORS["muted"], (33, 70))

        box_w = 118
        gap = 10
        total_w = box_w * 5 + gap * 4
        x = max(32, width - total_w - 32)
        y = 22
        backend = self.predictor.accepted_letter
        if backend == "-":
            backend = self.predictor.letter if self.predictor.ready else "-"
        confidence = f"{round(self.predictor.confidence * 100)}%" if self.predictor.ready else ""
        self.draw_scorebox("Score", str(self.score), pygame.Rect(x, y, box_w, 78))
        self.draw_scorebox("Streak", str(self.streak), pygame.Rect(x + (box_w + gap), y, box_w, 78))
        self.draw_scorebox(
            "Timer",
            f"{self.time_left:.1f}",
            pygame.Rect(x + (box_w + gap) * 2, y, box_w, 78),
            COLORS["accent_2"],
        )
        self.draw_scorebox("Signs/min", str(self.signs_per_minute()), pygame.Rect(x + (box_w + gap) * 3, y, box_w, 78))
        self.draw_scorebox("Predict", f"{backend} {confidence}".strip(), pygame.Rect(x + (box_w + gap) * 4, y, box_w, 78))

    def draw_stream(self, width: int) -> None:
        self.ensure_stream_length()
        circle_size = min(112, max(64, width // 10))
        gap = max(12, circle_size // 5)
        total_w = VISIBLE_COUNT * circle_size + (VISIBLE_COUNT - 1) * gap
        start_x = (width - total_w) // 2
        y = 130
        now = time.monotonic()

        track_rect = pygame.Rect(start_x - 20, y - 14, total_w + 40, circle_size + 28)
        pygame.draw.rect(self.screen, (18, 22, 20), track_rect, border_radius=track_rect.height // 2)
        pygame.draw.rect(self.screen, COLORS["border"], track_rect, 1, border_radius=track_rect.height // 2)

        for index, entry in enumerate(self.stream[:VISIBLE_COUNT]):
            center = (start_x + index * (circle_size + gap) + circle_size // 2, y + circle_size // 2)
            is_main = index == MAIN_INDEX
            radius = circle_size // 2
            fill = COLORS["panel_strong"] if is_main else COLORS["panel"]
            border = COLORS["accent"] if is_main else COLORS["border"]
            pygame.draw.circle(self.screen, fill, center, radius)
            pygame.draw.circle(self.screen, border, center, radius, 4 if is_main else 2)
            self.draw_text(entry.letter, "large", COLORS["ink"], center, center=True)
            if entry.cracked_until > now:
                end_x = int(center[0] + radius * 0.88)
                start_line_x = int(center[0] - radius * 0.88)
                pygame.draw.line(
                    self.screen,
                    COLORS["accent"],
                    (start_line_x, center[1]),
                    (end_x, center[1] + entry.crack_angle // 5),
                    5,
                )

    def draw_health(self, rect: pygame.Rect) -> None:
        point_h = (rect.height - 24) // MAX_HEALTH
        for index in range(MAX_HEALTH):
            y = rect.y + index * (point_h + 12)
            point = pygame.Rect(rect.x, y, rect.width, point_h)
            fill = (94, 122, 66) if index < self.health else (34, 45, 31)
            border = COLORS["border"] if index < self.health else COLORS["danger"]
            pygame.draw.rect(self.screen, fill, point, border_radius=8)
            pygame.draw.rect(self.screen, border, point, 2, border_radius=8)

    def draw_camera(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, COLORS["camera_bg"], rect, border_radius=8)
        if self.frame_bgr is None:
            message = self.camera_error or "Waiting for camera..."
            self.draw_text(message, "body", COLORS["muted"], rect.center, center=True)
            return

        frame_rgb = cv2.cvtColor(self.frame_bgr, cv2.COLOR_BGR2RGB)
        frame_h, frame_w = frame_rgb.shape[:2]
        frame_surface = pygame.image.frombuffer(frame_rgb.tobytes(), (frame_w, frame_h), "RGB")

        scale = max(rect.width / frame_w, rect.height / frame_h)
        scaled_size = (int(frame_w * scale), int(frame_h * scale))
        frame_surface = pygame.transform.smoothscale(frame_surface, scaled_size)
        crop = pygame.Rect(
            (scaled_size[0] - rect.width) // 2,
            (scaled_size[1] - rect.height) // 2,
            rect.width,
            rect.height,
        )
        self.screen.blit(frame_surface, rect.topleft, crop)
        pygame.draw.rect(self.screen, COLORS["border"], rect, 1, border_radius=8)

    def draw_play_area(self, width: int, height: int) -> None:
        camera_w = min(680, width - 180)
        camera_h = min(330, max(210, height - 390))
        camera_x = (width - camera_w) // 2 + 35
        camera_y = 280
        health_rect = pygame.Rect(camera_x - 92, camera_y, 62, camera_h)
        self.draw_health(health_rect)
        self.draw_camera(pygame.Rect(camera_x, camera_y, camera_w, camera_h))

        message_y = min(height - 100, camera_y + camera_h + 22)
        self.draw_text(self.message, "body", COLORS["accent_2"], (width // 2, message_y), center=True)
        self.draw_text(
            "Enter/R start or restart    Space manual score    Q/Esc quit",
            "small",
            COLORS["muted"],
            (width // 2, height - 42),
            center=True,
        )

    def draw(self) -> None:
        width, height = self.screen.get_size()
        self.screen.fill(COLORS["bg"])
        self.draw_header(width)
        self.draw_stream(width)
        self.draw_play_area(width, height)
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(self.args.fps) / 1000.0
            running = self.process_events()
            self.update_camera_and_prediction()
            self.update_game(dt)
            self.draw()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SignNinja as a desktop webcam app.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--window-width", type=int, default=1120)
    parser.add_argument("--window-height", type=int, default=780)
    parser.add_argument("--no-body-normalize", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = SignNinjaApp(args)
    try:
        app.run()
    finally:
        app.close()


if __name__ == "__main__":
    main()
