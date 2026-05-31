from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import FEATURE_NORMALIZATION, INPUT_FEATURES_PER_FRAME
from live_predict import DEFAULT_CHECKPOINT, load_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "asl_left_hand_letter_window_transformer_sequences.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the letter-window classifier to ONNX.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, window_frames = load_model(args.checkpoint, args.device)
    model.eval()

    sample = torch.zeros(
        1,
        window_frames,
        INPUT_FEATURES_PER_FRAME,
        dtype=torch.float32,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=["windows"],
        output_names=["logits"],
        dynamic_axes={"windows": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    print(f"Exported {args.output}")
    print(f"window_frames={window_frames}")
    print(f"input_features_per_frame={INPUT_FEATURES_PER_FRAME}")
    print(f"feature_normalization={FEATURE_NORMALIZATION}")


if __name__ == "__main__":
    main()
