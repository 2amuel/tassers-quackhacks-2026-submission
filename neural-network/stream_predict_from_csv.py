from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

PREDICTOR_FILE = Path(__file__).with_name("predict_from_csv.py")


def load_predictor_module():
    spec = importlib.util.spec_from_file_location("predict_from_csv", PREDICTOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load predictor file: {PREDICTOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


predictor = load_predictor_module()

SEQUENCE_LENGTH = predictor.SEQUENCE_LENGTH
CENTER_DELAY = SEQUENCE_LENGTH // 2
OUTPUT_TOKENS = predictor.OUTPUT_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream over a MediaPipe Holistic CSV, revise recent frame-token "
            "predictions as future context becomes available, and print the "
            "current CTC-collapsed prediction after each frame."
        )
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
    parser.add_argument(
        "--context-delay",
        type=int,
        default=CENTER_DELAY,
        help=(
            "Number of future frames to wait before a frame is considered finalized. "
            "Defaults to half the live 60-frame context window."
        ),
    )
    return parser.parse_args()


def token_name(token_id: int | None) -> str:
    if token_id is None:
        return "?"
    return OUTPUT_TOKENS[token_id]


def collapsed_prediction(tokens: list[int | None]) -> str:
    return predictor.ctc_collapse([token for token in tokens if token is not None])


def stream_predict(
    model: torch.nn.Module,
    features: torch.Tensor,
    device: str,
    context_delay: int,
) -> None:
    frame_tokens: list[int | None] = [None] * features.size(0)
    finalized_until = -1

    print("sample,updated_frames,finalized_until,current_prediction")
    with torch.no_grad():
        for end_index in range(features.size(0)):
            window_start = max(0, end_index - SEQUENCE_LENGTH + 1)
            window = predictor.padded_window(features, end_index).unsqueeze(0).to(device)
            log_probs = model(window)[:, 0, :]
            token_ids = torch.argmax(log_probs, dim=-1).tolist()

            padding_count = SEQUENCE_LENGTH - (end_index - window_start + 1)
            updates: list[str] = []

            for absolute_frame in range(window_start, end_index + 1):
                window_position = padding_count + (absolute_frame - window_start)
                new_token = int(token_ids[window_position])
                old_token = frame_tokens[absolute_frame]
                frame_tokens[absolute_frame] = new_token

                if old_token != new_token:
                    updates.append(f"{absolute_frame}:{token_name(new_token)}")

            finalized_until = max(finalized_until, end_index - context_delay)
            finalized_tokens = frame_tokens[: finalized_until + 1]
            provisional_tokens = frame_tokens[finalized_until + 1 : end_index + 1]

            current_prediction = collapsed_prediction(finalized_tokens + provisional_tokens)
            updated_frames = " ".join(updates) if updates else "-"
            print(
                f"{end_index},{updated_frames},{finalized_until},{current_prediction}"
            )

    final_prediction = collapsed_prediction(frame_tokens)
    print()
    print("frame,final_token")
    for frame_index, token_id in enumerate(frame_tokens):
        print(f"{frame_index},{token_name(token_id)}")

    print()
    print(f"final_prediction,{final_prediction}")


def main() -> None:
    args = parse_args()
    features = predictor.csv_to_feature_tensor(
        args.csv_path,
        normalize_to_body=not args.no_body_normalize,
    )
    model = predictor.load_model(args.checkpoint, args.device)
    stream_predict(model, features, args.device, args.context_delay)


if __name__ == "__main__":
    main()
