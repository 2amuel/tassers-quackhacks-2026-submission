# Letter Window Classifier

This is a non-CTC ASL fingerspelling experiment. It trains a Transformer to classify
short left-hand landmark-video windows as one letter (`A-Z`), then predicts full
videos by sliding the window over time and collapsing repeated letters.

## Train on isolated letter clips

```powershell
python .\letter-window-classifier\train.py --epochs 100
```

Training on isolated letters resumes from
`models/asl_left_hand_letter_window_transformer_letters.pt` by default when that
checkpoint exists. Use `--reset` to start over with fresh weights:

```powershell
python .\letter-window-classifier\train.py --epochs 100 --reset
```

By default this reads:

```text
training/letter_landmarks/*/holistic_landmarks.csv
```

The first character of each folder name is used as the label, so `a1` is `A`,
`z3` is `Z`, and so on.

## Train with multi-letter videos too

Multi-letter clips are split evenly by their `expected_text` in `labels.csv`.
This is approximate, but it is much easier to train than CTC.

```powershell
python .\letter-window-classifier\train.py `
  --sequence-labels-csv .\training-data\sam-data\data\labels.csv `
  --epochs 100
```

You can repeat `--sequence-labels-csv` for more datasets.

When `--sequence-labels-csv` is used without an explicit `--checkpoint`, training
uses `models/asl_left_hand_letter_window_transformer_sequences.pt` so noisy
sequence-split runs do not overwrite the isolated-letter model.

## Train With Sam Letter Data

The `sam-letter-data/the_data` folder contains a `labels.csv` plus `landmarks/`.
Use:

```powershell
python .\letter-window-classifier\train.py `
  --sam-letter-data-dir .\sam-letter-data\the_data `
  --epochs 100
```

## Predict

```powershell
python .\letter-window-classifier\predict.py .\training-data\sam-data\data\landmarks\clip_000019.csv
```

Useful prediction knobs:

```powershell
--stride 8
--confidence-threshold 0.45
--repeat-collapse 2
```

Lower the confidence threshold if the output is empty. Increase it if the output
has too many extra letters.
