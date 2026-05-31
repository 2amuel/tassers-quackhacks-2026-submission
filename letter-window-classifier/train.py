from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import (
    INPUT_FEATURES_PER_FRAME,
    LetterWindowDataset,
    discover_isolated_letter_examples,
    read_sequence_label_examples,
    split_examples,
)
from model import LETTERS, create_model


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sliding-window ASL letter classifier.")
    parser.add_argument("--letter-landmarks-dir", type=Path, default=PROJECT_ROOT / "training" / "letter_landmarks")
    parser.add_argument(
        "--sequence-labels-csv",
        type=Path,
        action="append",
        default=[],
        help="Multi-letter labels.csv to split into approximate per-letter windows. Can be repeated.",
    )
    parser.add_argument(
        "--sam-letter-data-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory like sam-letter-data/the_data containing labels.csv and landmarks/. Can be repeated.",
    )
    parser.add_argument("--window-frames", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--jitter-frames", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to an isolated-letter checkpoint unless --sequence-labels-csv is used.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Start from fresh weights even if --checkpoint already exists.",
    )
    parser.add_argument("--no-body-normalize", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        features = batch["features"].to(device)
        targets = batch["target"].to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(features)
            loss = loss_fn(logits, targets)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((torch.argmax(logits, dim=-1) == targets).sum().item())
        total_examples += batch_size

    return total_loss / max(total_examples, 1), total_correct / max(total_examples, 1)


def load_checkpoint(
    checkpoint: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    window_frames: int,
) -> tuple[int, float]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    checkpoint_window_frames = int(payload.get("window_frames", window_frames))
    if checkpoint_window_frames != window_frames:
        raise RuntimeError(
            f"Checkpoint was trained with --window-frames {checkpoint_window_frames}, "
            f"but this run requested {window_frames}."
        )
    checkpoint_input_dim = int(payload.get("input_features_per_frame", INPUT_FEATURES_PER_FRAME))
    if checkpoint_input_dim != INPUT_FEATURES_PER_FRAME:
        raise RuntimeError(
            f"Checkpoint was trained with {checkpoint_input_dim} input features per frame, "
            f"but right-hand-only training uses {INPUT_FEATURES_PER_FRAME}. "
            "Use --reset or a different --checkpoint."
        )

    model.load_state_dict(payload["model_state_dict"])
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    start_epoch = int(payload.get("epoch", 0))
    best_val_acc = float(payload.get("best_val_acc", -1.0))
    return start_epoch, best_val_acc


def save_checkpoint(
    checkpoint: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "window_frames": args.window_frames,
            "input_features_per_frame": INPUT_FEATURES_PER_FRAME,
            "letters": LETTERS,
            "args": vars(args),
        },
        checkpoint,
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        uses_extra_manifest = bool(args.sequence_labels_csv or args.sam_letter_data_dir)
        checkpoint_name = (
            "asl_left_hand_letter_window_transformer_sequences.pt"
            if uses_extra_manifest
            else "asl_left_hand_letter_window_transformer_letters.pt"
        )
        args.checkpoint = PROJECT_ROOT / "models" / checkpoint_name

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    examples = discover_isolated_letter_examples(args.letter_landmarks_dir)
    print(f"Loaded {len(examples)} isolated letter examples from {args.letter_landmarks_dir}")

    for labels_csv in args.sequence_labels_csv:
        sequence_examples = read_sequence_label_examples(labels_csv)
        print(f"Loaded {len(sequence_examples)} split sequence examples from {labels_csv}")
        examples.extend(sequence_examples)

    for sam_letter_data_dir in args.sam_letter_data_dir:
        labels_csv = sam_letter_data_dir / "labels.csv"
        sequence_examples = read_sequence_label_examples(labels_csv)
        print(f"Loaded {len(sequence_examples)} Sam letter-data examples from {labels_csv}")
        examples.extend(sequence_examples)

    if not examples:
        raise RuntimeError("No letter training examples found.")

    train_examples, val_examples = split_examples(examples, args.val_split, args.seed)
    train_dataset = LetterWindowDataset(
        train_examples,
        window_frames=args.window_frames,
        normalize_to_body=not args.no_body_normalize,
        jitter_frames=args.jitter_frames,
    )
    val_dataset = LetterWindowDataset(
        val_examples,
        window_frames=args.window_frames,
        normalize_to_body=not args.no_body_normalize,
        jitter_frames=0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: {
            "features": torch.stack([item["features"] for item in batch]),
            "target": torch.stack([item["target"] for item in batch]),
        },
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: {
            "features": torch.stack([item["features"] for item in batch]),
            "target": torch.stack([item["target"] for item in batch]),
        },
    )

    model = create_model(INPUT_FEATURES_PER_FRAME, args.window_frames).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    start_epoch = 0
    if args.checkpoint.exists() and not args.reset:
        print(f"Loading checkpoint {args.checkpoint} before training.")
        start_epoch, best_val_acc = load_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            device,
            args.window_frames,
        )
        print(f"Resumed from epoch {start_epoch} with best_val_acc={best_val_acc:.1%}.")
    elif args.reset and args.checkpoint.exists():
        print(f"--reset set; ignoring existing checkpoint {args.checkpoint}.")

    end_epoch = start_epoch + args.epochs
    for epoch in range(start_epoch + 1, end_epoch + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, loss_fn, device)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(model, val_loader, None, loss_fn, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(args.checkpoint, model, optimizer, epoch, best_val_acc, args)

        print(
            f"epoch {epoch:03d} ({epoch - start_epoch}/{args.epochs}) "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.1%} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.1%}"
        )

    print(f"Saved best checkpoint to {args.checkpoint}")


if __name__ == "__main__":
    main()
