from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import (
    FEATURE_NORMALIZATION,
    INPUT_FEATURES_PER_FRAME,
    csv_to_feature_tensor,
    resample_window,
)
from model import OUTPUT_LABELS, create_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "extra_words_left_hand_torso_face_sequences.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ASL letters with an expanded landmark-window classifier.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--repeat-collapse", type=int, default=2)
    parser.add_argument("--no-body-normalize", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: str) -> tuple[torch.nn.Module, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    window_frames = int(payload["window_frames"])
    input_dim = int(payload.get("input_features_per_frame", INPUT_FEATURES_PER_FRAME))
    if input_dim != INPUT_FEATURES_PER_FRAME:
        raise RuntimeError(
            f"Checkpoint expects {input_dim} features per frame, "
            f"but this expanded classifier provides {INPUT_FEATURES_PER_FRAME}."
        )
    checkpoint_feature_normalization = payload.get("feature_normalization")
    if checkpoint_feature_normalization != FEATURE_NORMALIZATION:
        raise RuntimeError(
            "Checkpoint uses a different feature normalization. "
            "Retrain with extra-words-classifier/train.py --reset or pass a newer checkpoint."
        )

    model = create_model(INPUT_FEATURES_PER_FRAME, window_frames).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, window_frames


def collapse_labels(predictions: list[tuple[str, float]], repeat_collapse: int) -> str:
    if len(predictions) == 1:
        return predictions[0][0]

    output: list[str] = []
    previous = ""
    run_length = 0

    for label, _confidence in predictions:
        if label == previous:
            run_length += 1
        else:
            previous = label
            run_length = 1

        if label and label != (output[-1] if output else "") and run_length >= repeat_collapse:
            output.append(label)

    return " ".join(output)


def main() -> None:
    args = parse_args()
    model, window_frames = load_model(args.checkpoint, args.device)
    features = csv_to_feature_tensor(
        args.csv_path,
        normalize_to_body=not args.no_body_normalize,
    )

    predictions: list[tuple[str, float]] = []
    print("start_frame,end_frame,label,confidence")
    with torch.no_grad():
        for start in range(0, max(features.size(0) - 1, 1), args.stride):
            end = min(start + window_frames, features.size(0))
            if end <= start:
                break
            window = resample_window(features[start:end], window_frames).unsqueeze(0).to(args.device)
            probs = torch.softmax(model(window)[0], dim=-1)
            confidence, class_id = torch.max(probs, dim=-1)
            label = OUTPUT_LABELS[int(class_id.item())]
            confidence_value = float(confidence.item())
            emitted_label = label if confidence_value >= args.confidence_threshold else ""
            predictions.append((emitted_label, confidence_value))
            print(f"{start},{end},{label},{confidence_value:.4f}")

    print()
    print(f"final_prediction,{collapse_labels(predictions, args.repeat_collapse)}")


if __name__ == "__main__":
    main()
