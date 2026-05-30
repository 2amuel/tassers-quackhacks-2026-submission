from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import torch


MODEL_FILE = Path(__file__).with_name("neural-network.py")


def load_model_module():
    spec = importlib.util.spec_from_file_location("asl_neural_network", MODEL_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load model file: {MODEL_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model_module = load_model_module()

SEQUENCE_LENGTH = model_module.SEQUENCE_LENGTH
INPUT_FEATURES_PER_FRAME = model_module.INPUT_FEATURES_PER_FRAME
OUTPUT_TOKENS = model_module.OUTPUT_TOKENS
SELECTED_FACE_LANDMARKS = model_module.SELECTED_FACE_LANDMARKS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ASL transformer over a MediaPipe Holistic landmark CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to holistic_landmarks.csv.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional PyTorch model checkpoint. If omitted, predictions use random weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--no-body-normalize",
        action="store_true",
        help="Do not subtract shoulder center and scale by shoulder width.",
    )
    return parser.parse_args()


def read_landmark_csv(csv_path: Path) -> dict[int, dict[str, dict[int, dict[str, float]]]]:
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
            }

    return samples


def body_normalization(groups: dict[str, dict[int, dict[str, float]]]) -> tuple[float, float, float]:
    pose = groups.get("pose", {})
    left_shoulder = pose.get(11)
    right_shoulder = pose.get(12)

    if left_shoulder is None or right_shoulder is None:
        return 0.0, 0.0, 1.0

    center_x = (left_shoulder["x"] + right_shoulder["x"]) / 2.0
    center_y = (left_shoulder["y"] + right_shoulder["y"]) / 2.0
    dx = left_shoulder["x"] - right_shoulder["x"]
    dy = left_shoulder["y"] - right_shoulder["y"]
    scale = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    return center_x, center_y, scale


def append_landmarks(
    features: list[float],
    landmarks: dict[int, dict[str, float]],
    indices: range | tuple[int, ...],
    center_x: float,
    center_y: float,
    scale: float,
    normalize_to_body: bool,
) -> None:
    for index in indices:
        point = landmarks.get(index)
        if point is None:
            features.extend((0.0, 0.0, 0.0))
            continue

        x = point["x"]
        y = point["y"]
        z = point["z"]
        if normalize_to_body:
            x = (x - center_x) / scale
            y = (y - center_y) / scale
            z = z / scale

        features.extend((x, y, z))


def sample_to_features(
    groups: dict[str, dict[int, dict[str, float]]],
    normalize_to_body: bool,
) -> list[float]:
    center_x, center_y, scale = body_normalization(groups)
    features: list[float] = []

    append_landmarks(
        features,
        groups.get("left_hand", {}),
        range(21),
        center_x,
        center_y,
        scale,
        normalize_to_body,
    )
    append_landmarks(
        features,
        groups.get("right_hand", {}),
        range(21),
        center_x,
        center_y,
        scale,
        normalize_to_body,
    )
    append_landmarks(
        features,
        groups.get("pose", {}),
        range(33),
        center_x,
        center_y,
        scale,
        normalize_to_body,
    )
    append_landmarks(
        features,
        groups.get("face", {}),
        SELECTED_FACE_LANDMARKS,
        center_x,
        center_y,
        scale,
        normalize_to_body,
    )

    if len(features) != INPUT_FEATURES_PER_FRAME:
        raise RuntimeError(
            f"Expected {INPUT_FEATURES_PER_FRAME} features, created {len(features)}."
        )
    return features


def csv_to_feature_tensor(csv_path: Path, normalize_to_body: bool) -> torch.Tensor:
    samples = read_landmark_csv(csv_path)
    if not samples:
        raise RuntimeError(f"No normalized landmarks found in {csv_path}")

    frames = [
        sample_to_features(samples[sample_index], normalize_to_body)
        for sample_index in sorted(samples)
    ]
    return torch.tensor(frames, dtype=torch.float32)


def load_model(checkpoint: Path | None, device: str) -> torch.nn.Module:
    model = model_module.create_model().to(device)
    model.eval()

    if checkpoint is None:
        print("Warning: no checkpoint supplied; predictions will use random weights.")
        return model

    payload = torch.load(checkpoint, map_location=device)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)
    return model


def ctc_collapse(class_ids: list[int]) -> str:
    decoded: list[str] = []
    previous = 0

    for class_id in class_ids:
        if class_id != 0 and class_id != previous:
            decoded.append(OUTPUT_TOKENS[class_id])
        previous = class_id

    return "".join(decoded)


def ctc_greedy_decode(log_probs: torch.Tensor) -> str:
    class_ids = torch.argmax(log_probs, dim=-1).tolist()
    return ctc_collapse(class_ids)


def padded_window(features: torch.Tensor, end_index: int) -> torch.Tensor:
    start_index = max(0, end_index - SEQUENCE_LENGTH + 1)
    window = features[start_index : end_index + 1]

    if window.size(0) < SEQUENCE_LENGTH:
        padding = torch.zeros(
            SEQUENCE_LENGTH - window.size(0),
            INPUT_FEATURES_PER_FRAME,
            dtype=features.dtype,
        )
        window = torch.cat((padding, window), dim=0)

    return window


def predict_each_frame(model: torch.nn.Module, features: torch.Tensor, device: str) -> str:
    predicted_path: list[int] = []

    print("sample,frame_token")
    with torch.no_grad():
        for end_index in range(features.size(0)):
            window = padded_window(features, end_index).unsqueeze(0).to(device)
            log_probs = model(window)

            newest_frame_log_probs = log_probs[-1, 0, :]
            frame_token_id = int(torch.argmax(newest_frame_log_probs).item())
            predicted_path.append(frame_token_id)

            frame_token = OUTPUT_TOKENS[frame_token_id]
            print(f"{end_index},{frame_token}")

    return ctc_collapse(predicted_path)


def main() -> None:
    args = parse_args()
    features = csv_to_feature_tensor(
        args.csv_path,
        normalize_to_body=not args.no_body_normalize,
    )
    model = load_model(args.checkpoint, args.device)
    prediction = predict_each_frame(model, features, args.device)
    print()
    print(f"final_prediction,{prediction}")


if __name__ == "__main__":
    main()
