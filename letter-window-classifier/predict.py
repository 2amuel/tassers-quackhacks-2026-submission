from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import INPUT_FEATURES_PER_FRAME, csv_to_feature_tensor, resample_window
from model import LETTERS, create_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict ASL letters with a sliding window classifier.")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "models" / "asl_left_hand_letter_window_transformer_letters.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.45)
    parser.add_argument("--repeat-collapse", type=int, default=2)
    parser.add_argument("--no-body-normalize", action="store_true")
    return parser.parse_args()


def load_model(checkpoint: Path, device: str) -> tuple[torch.nn.Module, int]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    window_frames = int(payload["window_frames"])
    model = create_model(INPUT_FEATURES_PER_FRAME, window_frames).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, window_frames


def collapse_letters(predictions: list[tuple[str, float]], repeat_collapse: int) -> str:
    if len(predictions) == 1:
        return predictions[0][0]

    output: list[str] = []
    previous = ""
    run_length = 0

    for letter, _confidence in predictions:
        if letter == previous:
            run_length += 1
        else:
            previous = letter
            run_length = 1

        if letter and letter != (output[-1] if output else "") and run_length >= repeat_collapse:
            output.append(letter)

    return "".join(output)


def main() -> None:
    args = parse_args()
    model, window_frames = load_model(args.checkpoint, args.device)
    features = csv_to_feature_tensor(
        args.csv_path,
        normalize_to_body=not args.no_body_normalize,
    )

    predictions: list[tuple[str, float]] = []
    print("start_frame,end_frame,letter,confidence")
    with torch.no_grad():
        for start in range(0, max(features.size(0) - 1, 1), args.stride):
            end = min(start + window_frames, features.size(0))
            if end <= start:
                break
            window = resample_window(features[start:end], window_frames).unsqueeze(0).to(args.device)
            probs = torch.softmax(model(window)[0], dim=-1)
            confidence, class_id = torch.max(probs, dim=-1)
            letter = LETTERS[int(class_id.item())]
            confidence_value = float(confidence.item())
            emitted_letter = letter if confidence_value >= args.confidence_threshold else ""
            predictions.append((emitted_letter, confidence_value))
            print(f"{start},{end},{letter},{confidence_value:.4f}")

    print()
    print(f"final_prediction,{collapse_letters(predictions, args.repeat_collapse)}")


if __name__ == "__main__":
    main()
