from __future__ import annotations

import argparse
import csv
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PREDICTOR_FILE = SCRIPT_DIR / "predict_from_csv.py"


def load_predictor_module():
    spec = importlib.util.spec_from_file_location("predict_from_csv", PREDICTOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load predictor file: {PREDICTOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


predictor = load_predictor_module()
OUTPUT_TOKENS = predictor.OUTPUT_TOKENS
LETTER_TO_CLASS = {letter: index for index, letter in enumerate(OUTPUT_TOKENS)}


@dataclass(frozen=True)
class Example:
    clip_id: str
    landmark_csv_path: Path
    expected_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ASL CTC transformer on data/labels.csv.")
    parser.add_argument("--labels-csv", type=Path, default=PROJECT_ROOT / "data" / "labels.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--checkpoint-name", default="asl_ctc_transformer.pt")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--no-body-normalize",
        action="store_true",
        help="Do not subtract shoulder center and scale by shoulder width.",
    )
    return parser.parse_args()


def resolve_path(path_text: str, labels_csv: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate

    candidate = labels_csv.parent / path
    if candidate.exists():
        return candidate

    if path.parts and path.parts[0].lower() == labels_csv.parent.name.lower():
        candidate = labels_csv.parent.parent / path
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / path


def read_examples(labels_csv: Path) -> list[Example]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Could not find labels CSV: {labels_csv}")

    examples: list[Example] = []
    with labels_csv.open(newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        for row in reader:
            expected_text = row["expected_text"].strip().upper()
            if not expected_text.isalpha():
                raise ValueError(
                    f"Label for {row.get('clip_id', '<unknown>')} contains non-letters: "
                    f"{expected_text!r}"
                )

            landmark_csv_path = resolve_path(row["landmark_csv_path"], labels_csv)
            if not landmark_csv_path.exists():
                print(f"Skipping {row.get('clip_id', '<unknown>')}: missing {landmark_csv_path}")
                continue

            examples.append(
                Example(
                    clip_id=row["clip_id"],
                    landmark_csv_path=landmark_csv_path,
                    expected_text=expected_text,
                )
            )

    if not examples:
        raise RuntimeError(f"No training examples found in {labels_csv}")
    return examples


class FingerspellingDataset(Dataset):
    def __init__(self, examples: list[Example], normalize_to_body: bool):
        self.examples = examples
        self.normalize_to_body = normalize_to_body

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        features = predictor.csv_to_feature_tensor(
            example.landmark_csv_path,
            normalize_to_body=self.normalize_to_body,
        )
        target = torch.tensor(
            [LETTER_TO_CLASS[letter] for letter in example.expected_text],
            dtype=torch.long,
        )
        return {
            "clip_id": example.clip_id,
            "features": features,
            "target": target,
            "target_text": example.expected_text,
        }


def collate_batch(batch: list[dict]) -> dict:
    features = [item["features"] for item in batch]
    targets = [item["target"] for item in batch]
    input_lengths = torch.tensor([feature.size(0) for feature in features], dtype=torch.long)
    target_lengths = torch.tensor([target.size(0) for target in targets], dtype=torch.long)

    padded_features = pad_sequence(features, batch_first=True)
    max_frames = padded_features.size(1)
    padding_mask = torch.arange(max_frames).unsqueeze(0) >= input_lengths.unsqueeze(1)

    return {
        "clip_ids": [item["clip_id"] for item in batch],
        "features": padded_features,
        "padding_mask": padding_mask,
        "input_lengths": input_lengths,
        "targets": torch.cat(targets),
        "target_lengths": target_lengths,
        "target_texts": [item["target_text"] for item in batch],
    }


def split_examples(examples: list[Example], val_split: float, seed: int) -> tuple[list[Example], list[Example]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_split))
    if len(shuffled) > 1:
        val_count = max(1, min(val_count, len(shuffled) - 1))
    else:
        val_count = 0
    return shuffled[val_count:], shuffled[:val_count]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.CTCLoss,
    device: str,
    grad_clip: float,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0

    for batch in loader:
        features = batch["features"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        targets = batch["targets"].to(device)
        input_lengths = batch["input_lengths"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        with torch.set_grad_enabled(is_training):
            log_probs = model(features, padding_mask=padding_mask)
            loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float | None,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "output_tokens": OUTPUT_TOKENS,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    examples = read_examples(args.labels_csv)
    train_examples, val_examples = split_examples(examples, args.val_split, args.seed)

    train_dataset = FingerspellingDataset(
        train_examples,
        normalize_to_body=not args.no_body_normalize,
    )
    val_dataset = FingerspellingDataset(
        val_examples,
        normalize_to_body=not args.no_body_normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    model = predictor.model_module.create_model().to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    best_val_loss: float | None = None
    best_path = args.checkpoint_dir / f"best_{args.checkpoint_name}"
    final_path = args.checkpoint_dir / args.checkpoint_name

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Device: {args.device}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            args.device,
            args.grad_clip,
        )

        val_loss = None
        if val_loader:
            with torch.no_grad():
                val_loss = run_epoch(
                    model,
                    val_loader,
                    None,
                    loss_fn,
                    args.device,
                    args.grad_clip,
                )

        save_checkpoint(final_path, model, optimizer, epoch, train_loss, val_loss, args)

        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, train_loss, val_loss, args)

        val_text = "n/a" if val_loss is None else f"{val_loss:.4f}"
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_text}"
        )

    print(f"Saved final model to {final_path}")
    if best_val_loss is not None:
        print(f"Saved best validation model to {best_path}")


if __name__ == "__main__":
    main()
