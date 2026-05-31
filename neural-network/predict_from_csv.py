from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import torch


MODEL_FILE = Path(__file__).with_name("neural-network.py")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "best_asl_ctc_transformer.pt"


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
        default=DEFAULT_CHECKPOINT,
        help=f"PyTorch model checkpoint. Defaults to {DEFAULT_CHECKPOINT}.",
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
    parser.add_argument(
        "--sliding-window",
        action="store_true",
        help="Use 60-frame sliding windows instead of running the full CSV sequence at once.",
    )
    parser.add_argument(
        "--blank-logit-penalty",
        type=float,
        default=0.0,
        help="Subtract this value from the blank class before greedy decoding.",
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
                "visibility": parse_optional_float(row.get("visibility")),
                "presence": parse_optional_float(row.get("presence")),
            }

    return samples


def parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def confidence_feature(point: dict[str, float]) -> float:
    presence = point.get("presence")
    visibility = point.get("visibility")
    if presence is not None:
        return max(0.0, min(float(presence), 1.0))
    if visibility is not None:
        return max(0.0, min(float(visibility), 1.0))
    return 1.0


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
            features.extend((0.0, 0.0, 0.0, 0.0))
            continue

        x = point["x"]
        y = point["y"]
        z = point["z"]
        if normalize_to_body:
            x = (x - center_x) / scale
            y = (y - center_y) / scale
            z = z / scale

        features.extend((x, y, z, confidence_feature(point)))


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


def load_model(checkpoint: Path, device: str) -> torch.nn.Module:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Could not find model checkpoint: {checkpoint}. "
            "Train first with train_ctc.py or pass --checkpoint."
        )

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload

    max_sequence_length = SEQUENCE_LENGTH
    if isinstance(state_dict, dict) and "position.pe" in state_dict:
        position_pe = state_dict["position.pe"]
        if isinstance(position_pe, torch.Tensor):
            max_sequence_length = position_pe.size(1)

    model = model_module.create_model(max_sequence_length=max_sequence_length).to(device)
    model.eval()
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


def apply_blank_logit_penalty(log_probs: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 0.0:
        return log_probs

    adjusted = log_probs.clone()
    adjusted[..., 0] -= penalty
    return adjusted


def ctc_greedy_decode(log_probs: torch.Tensor, blank_logit_penalty: float = 0.0) -> str:
    log_probs = apply_blank_logit_penalty(log_probs, blank_logit_penalty)
    class_ids = torch.argmax(log_probs, dim=-1).tolist()
    return ctc_collapse(class_ids)


def blank_margin_summary(log_probs: torch.Tensor) -> tuple[float, float]:
    blank_scores = log_probs[..., 0]
    best_letter_scores = log_probs[..., 1:].max(dim=-1).values
    margins = blank_scores - best_letter_scores
    return float(margins.mean().item()), float(margins.max().item())


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


def predict_sliding_windows(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    blank_logit_penalty: float,
) -> str:
    predicted_path: list[int] = []

    print("sample,frame_token")
    with torch.no_grad():
        for end_index in range(features.size(0)):
            window = padded_window(features, end_index).unsqueeze(0).to(device)
            log_probs = model(window)

            newest_frame_log_probs = apply_blank_logit_penalty(
                log_probs[-1, 0, :],
                blank_logit_penalty,
            )
            frame_token_id = int(torch.argmax(newest_frame_log_probs).item())
            predicted_path.append(frame_token_id)

            frame_token = OUTPUT_TOKENS[frame_token_id]
            print(f"{end_index},{frame_token}")

    return ctc_collapse(predicted_path)


def predict_full_sequence(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    blank_logit_penalty: float,
) -> str:
    print("sample,frame_token")
    with torch.no_grad():
        batch = features.unsqueeze(0).to(device)
        log_probs = model(batch)[:, 0, :]
        mean_blank_margin, max_blank_margin = blank_margin_summary(log_probs)
        print(f"blank_margin_mean,{mean_blank_margin:.4f}")
        print(f"blank_margin_max,{max_blank_margin:.4f}")
        log_probs = apply_blank_logit_penalty(log_probs, blank_logit_penalty)
        predicted_path = torch.argmax(log_probs, dim=-1).tolist()

    for sample_index, token_id in enumerate(predicted_path):
        print(f"{sample_index},{OUTPUT_TOKENS[token_id]}")

    return ctc_collapse(predicted_path)


def main() -> None:
    args = parse_args()
    features = csv_to_feature_tensor(
        args.csv_path,
        normalize_to_body=not args.no_body_normalize,
    )
    model = load_model(args.checkpoint, args.device)
    if args.sliding_window:
        prediction = predict_sliding_windows(model, features, args.device, args.blank_logit_penalty)
    else:
        prediction = predict_full_sequence(model, features, args.device, args.blank_logit_penalty)
    print()
    print(f"final_prediction,{prediction}")


if __name__ == "__main__":
    main()
