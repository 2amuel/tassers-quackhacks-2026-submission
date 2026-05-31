from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
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


@dataclass(frozen=True)
class EpochStats:
    loss: float
    nonblank_frame_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ASL CTC transformer on landmark CSV training data.")
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        default=PROJECT_ROOT / "training-data",
        help="Directory containing per-signer folders, each with data/labels.csv.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Train from one labels CSV instead of aggregating --training-data-dir.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train in this run.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--blank-logit-penalty",
        type=float,
        default=0.0,
        help="Subtract this value from the blank class before CTC loss to discourage all-blank collapse.",
    )
    parser.add_argument(
        "--frame-ce-weight",
        type=float,
        default=0.0,
        help="Weight for an auxiliary per-frame letter loss that helps prevent early blank collapse.",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=1.0,
        help="Weight for the CTC loss.",
    )
    parser.add_argument(
        "--frame-ce-warmup-epochs",
        type=int,
        default=0,
        help="Train with only the auxiliary frame loss for this many initial epochs.",
    )
    parser.add_argument(
        "--max-frames-per-letter",
        type=float,
        default=0.0,
        help="Uniformly downsample each clip to at most this many frames per target letter. Use 0 to disable.",
    )
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--checkpoint-name", default="asl_ctc_transformer.pt")
    parser.add_argument(
        "--no-checkpoint-load",
        action="store_true",
        help="Start from freshly initialized weights instead of loading an existing checkpoint.",
    )
    parser.add_argument(
        "--reset-output-layer",
        action="store_true",
        help="Reinitialize the final CTC classifier after loading a checkpoint.",
    )
    parser.add_argument(
        "--blank-bias",
        type=float,
        default=0.0,
        help="Blank-token bias to use with --reset-output-layer.",
    )
    parser.add_argument(
        "--letter-bias",
        type=float,
        default=0.0,
        help="Letter-token bias to use with --reset-output-layer.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "tensor-cache",
        help="Directory for cached feature tensors created from landmark CSVs.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute cached feature tensors even when a valid cache file exists.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. Use 'auto' to prefer cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA automatic mixed precision to speed up GPU training.",
    )
    parser.add_argument(
        "--no-body-normalize",
        action="store_true",
        help="Do not subtract shoulder center and scale by shoulder width.",
    )
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a CUDA GPU. "
            "Install a CUDA-enabled PyTorch build or use --device cpu."
        )
    return device


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"{device} ({torch.cuda.get_device_name(index)})"
    return str(device)


def discover_labels_csvs(training_data_dir: Path) -> list[Path]:
    if not training_data_dir.exists():
        raise FileNotFoundError(f"Could not find training data directory: {training_data_dir}")

    labels_csvs: list[Path] = []
    root_labels_csv = training_data_dir / "labels.csv"
    if root_labels_csv.exists():
        labels_csvs.append(root_labels_csv)

    for data_folder in sorted(path for path in training_data_dir.iterdir() if path.is_dir()):
        labels_csv = data_folder / "data" / "labels.csv"
        if labels_csv.exists():
            labels_csvs.append(labels_csv)
            continue

        labels_csv = data_folder / "labels.csv"
        if labels_csv.exists():
            labels_csvs.append(labels_csv)

    generated_labels_csv = PROJECT_ROOT / "training" / "data_output" / "labels.csv"
    if generated_labels_csv.exists() and generated_labels_csv not in labels_csvs:
        labels_csvs.append(generated_labels_csv)

    if not labels_csvs:
        raise RuntimeError(f"No labels.csv files found under {training_data_dir}")
    return labels_csvs


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


def read_examples_from_csv(labels_csv: Path) -> list[Example]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Could not find labels CSV: {labels_csv}")

    examples: list[Example] = []
    dataset_name = labels_csv.parent.parent.name if labels_csv.parent.name == "data" else labels_csv.parent.name
    with labels_csv.open(newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        for row in reader:
            expected_text = row["expected_text"].strip().upper()
            if not expected_text.isalpha():
                raise ValueError(
                    f"Label for {row.get('clip_id', '<unknown>')} contains non-letters: "
                    f"{expected_text!r}"
                )
            num_frames = parse_optional_int(row.get("num_frames"))
            min_timesteps = min_ctc_timesteps(expected_text)
            if num_frames is not None and num_frames < min_timesteps:
                print(
                    f"Skipping {row.get('clip_id', '<unknown>')}: "
                    f"{num_frames} frames is too short for {expected_text!r} "
                    f"(needs at least {min_timesteps})"
                )
                continue

            landmark_csv_path = resolve_path(row["landmark_csv_path"], labels_csv)
            if not landmark_csv_path.exists():
                print(f"Skipping {row.get('clip_id', '<unknown>')}: missing {landmark_csv_path}")
                continue

            examples.append(
                Example(
                    clip_id=f"{dataset_name}/{row['clip_id']}",
                    landmark_csv_path=landmark_csv_path,
                    expected_text=expected_text,
                )
            )

    if not examples:
        raise RuntimeError(f"No training examples found in {labels_csv}")
    return examples


def read_examples(labels_csvs: list[Path]) -> list[Example]:
    examples: list[Example] = []
    for labels_csv in labels_csvs:
        csv_examples = read_examples_from_csv(labels_csv)
        print(f"Loaded {len(csv_examples)} examples from {labels_csv}")
        examples.extend(csv_examples)

    if not examples:
        raise RuntimeError("No training examples found")
    return examples


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def min_ctc_timesteps(text: str) -> int:
    repeated_neighbors = sum(1 for previous, current in zip(text, text[1:]) if previous == current)
    return len(text) + repeated_neighbors


def cache_key(csv_path: Path, normalize_to_body: bool) -> str:
    resolved = str(csv_path.resolve()).lower()
    mode = "body" if normalize_to_body else "raw"
    digest = hashlib.sha1(f"{resolved}|{mode}".encode("utf-8")).hexdigest()
    return digest[:16]


def cache_metadata(csv_path: Path, normalize_to_body: bool) -> dict:
    stat = csv_path.stat()
    return {
        "source_path": str(csv_path.resolve()),
        "source_mtime_ns": stat.st_mtime_ns,
        "source_size": stat.st_size,
        "normalize_to_body": normalize_to_body,
        "input_features_per_frame": predictor.INPUT_FEATURES_PER_FRAME,
    }


def is_valid_cache(payload: dict, expected_metadata: dict) -> bool:
    return all(payload.get(key) == value for key, value in expected_metadata.items())


def load_or_create_features(
    csv_path: Path,
    normalize_to_body: bool,
    cache_dir: Path | None,
    rebuild_cache: bool,
) -> torch.Tensor:
    if cache_dir is None:
        return predictor.csv_to_feature_tensor(csv_path, normalize_to_body=normalize_to_body)

    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata = cache_metadata(csv_path, normalize_to_body)
    cache_path = cache_dir / f"{cache_key(csv_path, normalize_to_body)}.pt"

    if cache_path.exists() and not rebuild_cache:
        try:
            payload = torch.load(cache_path, map_location="cpu")
            if isinstance(payload, dict) and is_valid_cache(payload.get("metadata", {}), metadata):
                return payload["features"]
        except (OSError, RuntimeError, KeyError, TypeError, ValueError):
            pass

    features = predictor.csv_to_feature_tensor(csv_path, normalize_to_body=normalize_to_body)
    torch.save({"metadata": metadata, "features": features}, cache_path)
    return features


def resample_features(features: torch.Tensor, max_frames: int) -> torch.Tensor:
    if max_frames <= 0 or features.size(0) <= max_frames:
        return features

    indices = torch.linspace(0, features.size(0) - 1, steps=max_frames)
    return features[indices.round().long()]


class FingerspellingDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        normalize_to_body: bool,
        cache_dir: Path | None,
        rebuild_cache: bool,
        max_frames_per_letter: float,
    ):
        self.examples = examples
        self.normalize_to_body = normalize_to_body
        self.cache_dir = cache_dir
        self.rebuild_cache = rebuild_cache
        self.max_frames_per_letter = max_frames_per_letter

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        features = load_or_create_features(
            example.landmark_csv_path,
            self.normalize_to_body,
            self.cache_dir,
            self.rebuild_cache,
        )
        max_frames = 0
        if self.max_frames_per_letter > 0.0:
            max_frames = max(
                min_ctc_timesteps(example.expected_text),
                int(round(len(example.expected_text) * self.max_frames_per_letter)),
            )
        features = resample_features(features, max_frames)
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


def apply_blank_logit_penalty(log_probs: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 0.0:
        return log_probs

    adjusted = log_probs.clone()
    adjusted[:, :, 0] -= penalty
    return adjusted - torch.logsumexp(adjusted, dim=-1, keepdim=True)


def nonblank_frame_rate(log_probs: torch.Tensor, input_lengths: torch.Tensor) -> float:
    predictions = torch.argmax(log_probs.detach(), dim=-1)
    max_frames = predictions.size(0)
    frame_mask = torch.arange(max_frames, device=predictions.device).unsqueeze(1) < input_lengths.unsqueeze(0)
    valid_frames = int(frame_mask.sum().item())
    if valid_frames == 0:
        return 0.0

    nonblank_frames = int(((predictions != 0) & frame_mask).sum().item())
    return nonblank_frames / valid_frames


def frame_letter_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> torch.Tensor:
    batch_log_probs = log_probs.transpose(0, 1)
    frame_log_probs: list[torch.Tensor] = []
    frame_targets: list[torch.Tensor] = []
    target_offset = 0

    for batch_index, input_length_tensor in enumerate(input_lengths):
        input_length = int(input_length_tensor.item())
        target_length = int(target_lengths[batch_index].item())
        sequence_targets = targets[target_offset : target_offset + target_length]
        target_offset += target_length

        if input_length == 0 or target_length == 0:
            continue

        positions = torch.arange(input_length, device=log_probs.device)
        target_indices = torch.div(
            positions * target_length,
            input_length,
            rounding_mode="floor",
        ).clamp(max=target_length - 1)
        frame_log_probs.append(batch_log_probs[batch_index, :input_length, :])
        frame_targets.append(sequence_targets[target_indices])

    if not frame_log_probs:
        return log_probs.new_zeros(())

    return F.nll_loss(torch.cat(frame_log_probs, dim=0), torch.cat(frame_targets, dim=0))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    loss_fn: nn.CTCLoss,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    blank_logit_penalty: float,
    frame_ce_weight: float,
    ctc_weight: float,
) -> EpochStats:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_examples = 0
    total_nonblank_frame_rate = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and is_training)

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        autocast_context = torch.cuda.amp.autocast(enabled=True) if use_amp else contextlib.nullcontext()
        with torch.set_grad_enabled(is_training):
            with autocast_context:
                log_probs = model(features, padding_mask=padding_mask)
                log_probs_for_loss = apply_blank_logit_penalty(log_probs, blank_logit_penalty)
                loss = ctc_weight * loss_fn(log_probs_for_loss, targets, input_lengths, target_lengths)
                if frame_ce_weight > 0.0:
                    loss = loss + frame_ce_weight * frame_letter_loss(
                        log_probs_for_loss,
                        targets,
                        input_lengths,
                        target_lengths,
                    )

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()

        batch_size = features.size(0)
        total_loss += float(loss.item()) * batch_size
        total_nonblank_frame_rate += nonblank_frame_rate(log_probs, input_lengths) * batch_size
        total_examples += batch_size

    return EpochStats(
        loss=total_loss / max(total_examples, 1),
        nonblank_frame_rate=total_nonblank_frame_rate / max(total_examples, 1),
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model.load_state_dict(state_dict)

    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    start_epoch = int(payload.get("epoch", 0))
    val_loss = payload.get("val_loss")
    return start_epoch, val_loss


def checkpoint_sequence_length(path: Path, default: int) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    if isinstance(state_dict, dict) and "position.pe" in state_dict:
        position_pe = state_dict["position.pe"]
        if isinstance(position_pe, torch.Tensor):
            return position_pe.size(1)
    return default


def reset_output_layer(model: nn.Module, blank_bias: float, letter_bias: float) -> None:
    classifier = model.output_layer[-1]
    nn.init.xavier_uniform_(classifier.weight)
    nn.init.constant_(classifier.bias, letter_bias)
    classifier.bias.data[0] = blank_bias


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
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")

    device = resolve_device(args.device)
    use_cuda = device.type == "cuda"
    use_amp = args.amp and use_cuda

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.amp:
        print("AMP requested, but it is only enabled for CUDA devices. Continuing without AMP.")

    labels_csvs = [args.labels_csv] if args.labels_csv is not None else discover_labels_csvs(args.training_data_dir)
    examples = read_examples(labels_csvs)
    train_examples, val_examples = split_examples(examples, args.val_split, args.seed)

    train_dataset = FingerspellingDataset(
        train_examples,
        normalize_to_body=not args.no_body_normalize,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        max_frames_per_letter=args.max_frames_per_letter,
    )
    val_dataset = FingerspellingDataset(
        val_examples,
        normalize_to_body=not args.no_body_normalize,
        cache_dir=args.cache_dir,
        rebuild_cache=args.rebuild_cache,
        max_frames_per_letter=args.max_frames_per_letter,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_batch,
    )

    best_val_loss: float | None = None
    best_path = args.checkpoint_dir / f"best_{args.checkpoint_name}"
    final_path = args.checkpoint_dir / args.checkpoint_name

    initial_checkpoint_path = None
    if not args.no_checkpoint_load:
        if best_path.exists():
            initial_checkpoint_path = best_path
        elif final_path.exists():
            initial_checkpoint_path = final_path

    max_sequence_length = predictor.SEQUENCE_LENGTH
    if initial_checkpoint_path is not None:
        max_sequence_length = checkpoint_sequence_length(
            initial_checkpoint_path,
            default=predictor.SEQUENCE_LENGTH,
        )

    model = predictor.model_module.create_model(max_sequence_length=max_sequence_length).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    if initial_checkpoint_path is not None:
        print(f"Loading checkpoint {initial_checkpoint_path} before training.")
        start_epoch, resumed_val_loss = load_checkpoint(initial_checkpoint_path, model, optimizer, device)
        if resumed_val_loss is not None:
            best_val_loss = resumed_val_loss
        print(f"Starting from checkpoint epoch {start_epoch}.")
    else:
        start_epoch = 0

    reset_fresh_output_layer = initial_checkpoint_path is None and not args.reset_output_layer
    if reset_fresh_output_layer:
        reset_output_layer(model, blank_bias=0.0, letter_bias=0.0)
        print("Using neutral final CTC output layer for fresh training.")

    if args.reset_output_layer:
        reset_output_layer(model, args.blank_bias, args.letter_bias)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        best_val_loss = None
        print(
            "Reset final CTC output layer "
            f"(blank_bias={args.blank_bias}, letter_bias={args.letter_bias}) "
            "and reset optimizer state."
        )

    end_epoch = start_epoch + args.epochs

    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Device: {describe_device(device)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")
    print(f"Tensor cache: {args.cache_dir}")
    print(f"Max frames per letter: {args.max_frames_per_letter or 'disabled'}")
    print(f"Frame CE warmup epochs: {args.frame_ce_warmup_epochs}")
    print(f"Epochs this run: {args.epochs}")

    for epoch in range(start_epoch + 1, end_epoch + 1):
        run_epoch_index = epoch - start_epoch
        if run_epoch_index <= args.frame_ce_warmup_epochs:
            epoch_ctc_weight = 0.0
            epoch_frame_ce_weight = max(args.frame_ce_weight, 1.0)
        else:
            epoch_ctc_weight = args.ctc_weight
            epoch_frame_ce_weight = args.frame_ce_weight

        train_stats = run_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            args.grad_clip,
            use_amp,
            args.blank_logit_penalty,
            epoch_frame_ce_weight,
            epoch_ctc_weight,
        )

        val_stats = None
        if val_loader:
            with torch.no_grad():
                val_stats = run_epoch(
                    model,
                    val_loader,
                    None,
                    loss_fn,
                    device,
                    args.grad_clip,
                    use_amp=False,
                    blank_logit_penalty=args.blank_logit_penalty,
                    frame_ce_weight=epoch_frame_ce_weight,
                    ctc_weight=epoch_ctc_weight,
                )

        val_loss = None if val_stats is None else val_stats.loss
        save_checkpoint(final_path, model, optimizer, epoch, train_stats.loss, val_loss, args)

        if val_loss is not None and (best_val_loss is None or val_loss < best_val_loss):
            best_val_loss = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, train_stats.loss, val_loss, args)

        val_text = "n/a" if val_loss is None else f"{val_loss:.4f}"
        val_nonblank_text = "n/a" if val_stats is None else f"{val_stats.nonblank_frame_rate:.1%}"
        print(
            f"epoch {epoch:03d} ({epoch - start_epoch}/{args.epochs}) "
            f"train_loss={train_stats.loss:.4f} val_loss={val_text} "
            f"train_nonblank={train_stats.nonblank_frame_rate:.1%} "
            f"val_nonblank={val_nonblank_text} "
            f"ctc_weight={epoch_ctc_weight:.2f} frame_ce_weight={epoch_frame_ce_weight:.2f}"
        )

    print(f"Saved final model to {final_path}")
    if best_val_loss is not None:
        print(f"Saved best validation model to {best_path}")


if __name__ == "__main__":
    main()
