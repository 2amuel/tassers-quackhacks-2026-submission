from __future__ import annotations

import csv
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from model import LETTERS


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PREDICTOR_FILE = PROJECT_ROOT / "neural-network" / "predict_from_csv.py"


def load_predictor_module():
    spec = importlib.util.spec_from_file_location("predict_from_csv", PREDICTOR_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load predictor file: {PREDICTOR_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


predictor = load_predictor_module()
HAND_FEATURES_PER_FRAME = predictor.model_module.HAND_FEATURES
LEFT_HAND_START = 0
LEFT_HAND_END = HAND_FEATURES_PER_FRAME
INPUT_FEATURES_PER_FRAME = HAND_FEATURES_PER_FRAME
LETTER_TO_CLASS = {letter: index for index, letter in enumerate(LETTERS)}


@dataclass(frozen=True)
class LetterExample:
    clip_id: str
    csv_path: Path
    letter: str
    start_frame: int | None = None
    end_frame: int | None = None


def resolve_path(path_text: str, labels_csv: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path

    for candidate in (
        PROJECT_ROOT / path,
        labels_csv.parent / path,
        labels_csv.parent.parent / path,
    ):
        if candidate.exists():
            return candidate

    if path.parts and path.parts[0].lower() == "data":
        candidate = labels_csv.parent / Path(*path.parts[1:])
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / path


def discover_isolated_letter_examples(letter_landmarks_dir: Path) -> list[LetterExample]:
    if not letter_landmarks_dir.exists():
        return []

    examples: list[LetterExample] = []
    for folder in sorted(path for path in letter_landmarks_dir.iterdir() if path.is_dir()):
        letter = folder.name[:1].upper()
        csv_path = folder / "holistic_landmarks.csv"
        if letter in LETTER_TO_CLASS and csv_path.exists():
            examples.append(LetterExample(folder.name, csv_path, letter))
    return examples


def read_sequence_label_examples(labels_csv: Path) -> list[LetterExample]:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Could not find labels CSV: {labels_csv}")

    examples: list[LetterExample] = []
    dataset_name = labels_csv.parent.parent.name if labels_csv.parent.name == "data" else labels_csv.parent.name
    with labels_csv.open(newline="", encoding="utf-8") as labels_file:
        reader = csv.DictReader(labels_file)
        for row in reader:
            expected_text = row["expected_text"].strip().upper()
            if not expected_text or any(letter not in LETTER_TO_CLASS for letter in expected_text):
                continue

            csv_path = resolve_path(row["landmark_csv_path"], labels_csv)
            if not csv_path.exists():
                print(f"Skipping {row.get('clip_id', '<unknown>')}: missing {csv_path}")
                continue

            num_frames = parse_optional_int(row.get("num_frames"))
            if num_frames is None:
                features = predictor.csv_to_feature_tensor(csv_path, normalize_to_body=True)
                num_frames = features.size(0)

            clip_id = f"{dataset_name}/{row.get('clip_id', csv_path.stem)}"
            for index, letter in enumerate(expected_text):
                start_frame = round(index * num_frames / len(expected_text))
                end_frame = round((index + 1) * num_frames / len(expected_text))
                examples.append(
                    LetterExample(
                        clip_id=f"{clip_id}:{index}:{letter}",
                        csv_path=csv_path,
                        letter=letter,
                        start_frame=start_frame,
                        end_frame=max(start_frame + 1, end_frame),
                    )
                )
    return examples


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def resample_window(features: torch.Tensor, window_frames: int) -> torch.Tensor:
    if features.size(0) == window_frames:
        return features
    if features.size(0) == 0:
        return torch.zeros(window_frames, INPUT_FEATURES_PER_FRAME, dtype=torch.float32)

    indices = torch.linspace(0, features.size(0) - 1, steps=window_frames)
    return features[indices.round().long()]


def left_hand_features(features: torch.Tensor) -> torch.Tensor:
    return features[:, LEFT_HAND_START:LEFT_HAND_END]


def csv_to_feature_tensor(csv_path: Path, normalize_to_body: bool) -> torch.Tensor:
    features = predictor.csv_to_feature_tensor(csv_path, normalize_to_body=normalize_to_body)
    return left_hand_features(features)


def split_examples(
    examples: list[LetterExample],
    val_split: float,
    seed: int,
) -> tuple[list[LetterExample], list[LetterExample]]:
    shuffled = examples[:]
    random.Random(seed).shuffle(shuffled)
    val_count = int(round(len(shuffled) * val_split))
    if len(shuffled) > 1:
        val_count = max(1, min(val_count, len(shuffled) - 1))
    else:
        val_count = 0
    return shuffled[val_count:], shuffled[:val_count]


class LetterWindowDataset(Dataset):
    def __init__(
        self,
        examples: list[LetterExample],
        window_frames: int,
        normalize_to_body: bool,
        jitter_frames: int,
    ):
        self.examples = examples
        self.window_frames = window_frames
        self.normalize_to_body = normalize_to_body
        self.jitter_frames = jitter_frames
        self._feature_cache: dict[Path, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def _load_features(self, csv_path: Path) -> torch.Tensor:
        if csv_path not in self._feature_cache:
            self._feature_cache[csv_path] = csv_to_feature_tensor(
                csv_path,
                normalize_to_body=self.normalize_to_body,
            )
        return self._feature_cache[csv_path]

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        features = self._load_features(example.csv_path)

        start_frame = 0 if example.start_frame is None else example.start_frame
        end_frame = features.size(0) if example.end_frame is None else example.end_frame
        if self.jitter_frames > 0:
            shift = random.randint(-self.jitter_frames, self.jitter_frames)
            start_frame += shift
            end_frame += shift

        start_frame = max(0, min(start_frame, features.size(0) - 1))
        end_frame = max(start_frame + 1, min(end_frame, features.size(0)))
        window = resample_window(features[start_frame:end_frame], self.window_frames)

        return {
            "clip_id": example.clip_id,
            "features": window,
            "target": torch.tensor(LETTER_TO_CLASS[example.letter], dtype=torch.long),
            "letter": example.letter,
        }
