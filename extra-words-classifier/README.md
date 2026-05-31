# Extra Words Classifier

This is a copy of `letter-window-classifier` with an expanded per-frame landmark
input for signs where body and head position may matter more than isolated
letter handshape.

It predicts the 26 alphabet labels plus:

```text
HELLO, MY, NAME, YOU, YOUR, HOW, GOOD, BAD, HAPPY, SAD, NOT
```

`I` is included through the alphabet label set, so clips labeled `I` train the
same output class.

Each frame uses:

```text
left hand: 21 landmarks * xyz/confidence = 84 features
torso:     pose landmarks 11,12,13,14,23,24 * xyz/confidence = 24 features
face:      face landmarks 1,10,152,234,454 * xyz/confidence = 20 features
total:     128 features per frame
```

The left hand is still wrist-centered and hand-scale-normalized. Torso and face
anchors come from the existing body-normalized holistic feature tensor, so they
preserve signer-relative position without pulling in expression-heavy face
features.

## Train

```powershell
python .\extra-words-classifier\train.py --epochs 100 --reset
```

By default this trains from all four local sources when they exist:

```text
data/labels.csv
sam-letter-data/the_data/labels.csv
aaryan-letter-data/data/labels.csv
words-data/labels.csv
```

Rows from `words-data/labels.csv` are treated as whole-label examples for the
word vocabulary. Existing letter datasets still train the alphabet labels.

Default checkpoints are separate from `letter-window-classifier`:

```text
models/extra_words_left_hand_torso_face_letters.pt
models/extra_words_left_hand_torso_face_sequences.pt
```

## Predict

```powershell
python .\extra-words-classifier\predict.py .\training-data\sam-data\data\landmarks\clip_000019.csv
```

## Live Predict

```powershell
python .\extra-words-classifier\live_predict.py
```
