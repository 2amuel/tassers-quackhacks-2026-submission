import string
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


CTC_BLANK_TOKEN = "<blank>"
LETTERS = tuple(string.ascii_uppercase)
OUTPUT_TOKENS = (CTC_BLANK_TOKEN,) + LETTERS

NUM_HANDS = 2
HAND_LANDMARKS_PER_HAND = 21
POSE_LANDMARKS = 33

# A compact face subset for ASL-relevant expression cues: eyebrows, eyes, nose,
# lips/mouth, chin, and face orientation anchors. These are MediaPipe Face Mesh
# landmark indices.
SELECTED_FACE_LANDMARKS = (
    10,
    152,
    234,
    454,
    1,
    4,
    33,
    133,
    159,
    145,
    362,
    263,
    386,
    374,
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    61,
    291,
    0,
    17,
    13,
    14,
    78,
    308,
    82,
    312,
    87,
    317,
    178,
    402,
    95,
    324,
)

COORDS_PER_LANDMARK = 3
SEQUENCE_LENGTH = 60

HAND_FEATURES = NUM_HANDS * HAND_LANDMARKS_PER_HAND * COORDS_PER_LANDMARK
POSE_FEATURES = POSE_LANDMARKS * COORDS_PER_LANDMARK
FACE_FEATURES = len(SELECTED_FACE_LANDMARKS) * COORDS_PER_LANDMARK
INPUT_FEATURES_PER_FRAME = HAND_FEATURES + POSE_FEATURES + FACE_FEATURES


@dataclass(frozen=True)
class LandmarkLayout:
    sequence_length: int = SEQUENCE_LENGTH
    hands: int = NUM_HANDS
    hand_landmarks_per_hand: int = HAND_LANDMARKS_PER_HAND
    pose_landmarks: int = POSE_LANDMARKS
    selected_face_landmarks: tuple[int, ...] = SELECTED_FACE_LANDMARKS
    coords_per_landmark: int = COORDS_PER_LANDMARK
    input_features_per_frame: int = INPUT_FEATURES_PER_FRAME
    output_tokens: tuple[str, ...] = OUTPUT_TOKENS


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = SEQUENCE_LENGTH):
        super().__init__()
        self.embed_dim = embed_dim
        self.register_buffer("pe", self._build(max_len))

    def _build(self, max_len: int) -> torch.Tensor:
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )

        pe = torch.zeros(max_len, self.embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            self.pe = self._build(x.size(1)).to(device=x.device, dtype=x.dtype)
        return x + self.pe[:, : x.size(1)]


class ASLTransformerCTC(nn.Module):
    """Transformer model for landmark-based ASL fingerspelling recognition.

    Input shape:
        (batch, frames, 345)

    Per-frame features:
        - 2 hands * 21 landmarks * xyz = 126
        - 33 pose landmarks * xyz = 99
        - 40 selected face landmarks * xyz = 120

    Output shape:
        (frames, batch, 27)

    The output is log-probabilities for PyTorch's CTCLoss. Class 0 is the CTC
    blank token. Classes 1-26 are A-Z.
    """

    def __init__(
        self,
        input_dim: int = INPUT_FEATURES_PER_FRAME,
        num_classes: int = len(OUTPUT_TOKENS),
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_sequence_length: int = SEQUENCE_LENGTH,
    ):
        super().__init__()
        self.layout = LandmarkLayout()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position = PositionalEncoding(embed_dim, max_sequence_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        landmarks: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if landmarks.ndim != 3:
            raise ValueError("Expected landmarks shape (batch, frames, features).")
        if landmarks.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features per frame, "
                f"received {landmarks.size(-1)}."
            )

        x = self.input_projection(landmarks)
        x = self.position(x)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        logits = self.output_layer(x)

        # CTCLoss expects (time, batch, classes).
        return F.log_softmax(logits, dim=-1).transpose(0, 1)


def create_model() -> ASLTransformerCTC:
    return ASLTransformerCTC()


def example_ctc_loss() -> torch.Tensor:
    model = create_model()
    batch_size = 2
    frames = SEQUENCE_LENGTH

    landmarks = torch.randn(batch_size, frames, INPUT_FEATURES_PER_FRAME)
    log_probs = model(landmarks)

    # Example targets: "ASL" and "CAT". Blank is class 0, so A=1, B=2, etc.
    targets = torch.tensor([1, 19, 12, 3, 1, 20], dtype=torch.long)
    target_lengths = torch.tensor([3, 3], dtype=torch.long)
    input_lengths = torch.full((batch_size,), frames, dtype=torch.long)

    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)
    return loss_fn(log_probs, targets, input_lengths, target_lengths)


if __name__ == "__main__":
    model = create_model()
    sample = torch.randn(4, 81, INPUT_FEATURES_PER_FRAME)
    output = model(sample)

    print(f"Input shape:  {tuple(sample.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Tokens:       {OUTPUT_TOKENS}")
