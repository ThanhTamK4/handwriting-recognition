"""Train an mltu CRNN+CTC word recognizer on the local IAM_Words dataset.

Ported from mltu Tutorial 03 (pythonlessons/mltu). Skips the remote dataset
download and instead reads the IAM_Words files already extracted under
`data/IAM_Words/`. On best-val-loss the model is auto-exported to
`models/mltu/model.onnx` (with `configs.yaml` alongside) via the `Model2onnx`
callback, so the main venv can run inference with `onnxruntime` only.

Run inside `.venv-train` (TF 2.10 + mltu):
    python training/train_mltu.py
"""
from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
)
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Bidirectional,
    Conv2D,
    Dense,
    Dropout,
    Input,
    LSTM,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence

import cv2
import numpy as np

from mltu.annotations.images import CVImage
from mltu.dataProvider import DataProvider
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding


# ---------- paths ----------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "IAM_Words"
WORDS_TXT = DATA_DIR / "words.txt"
WORDS_DIR = DATA_DIR / "words"
OUT_DIR = PROJECT_ROOT / "models" / "mltu"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- dataset parsing ----------

def load_samples():
    if not WORDS_TXT.exists():
        raise FileNotFoundError(f"{WORDS_TXT} not found. Extract IAM_Words first.")
    samples = []
    max_len = 0
    vocab = set()
    with open(WORDS_TXT, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split(" ")
            if len(parts) < 9 or parts[1] != "ok":
                continue
            word_id = parts[0]
            label = " ".join(parts[8:])
            a, b, *_ = word_id.split("-")
            img_path = WORDS_DIR / a / f"{a}-{b}" / f"{word_id}.png"
            if not img_path.exists() or img_path.stat().st_size == 0:
                continue
            samples.append([str(img_path), label])
            vocab.update(label)
            max_len = max(max_len, len(label))
    return samples, sorted(vocab), max_len


# ---------- model ----------

def residual_block(x, filters, strides=1):
    shortcut = x
    x = Conv2D(filters, (3, 3), strides=strides, padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=strides, padding="same", use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def build_model(input_dim, output_dim):
    inputs = Input(shape=input_dim, name="input")
    x = tf.keras.layers.Lambda(lambda img: img / 255.0)(inputs)

    x = residual_block(x, 16)
    x = residual_block(x, 16)
    x = residual_block(x, 32, strides=2)
    x = residual_block(x, 32)
    x = residual_block(x, 64, strides=2)
    x = residual_block(x, 64)

    # Collapse height axis -> sequence along width. Transpose so width becomes
    # the time axis: (B, H, W, C) -> (B, W, H, C) -> (B, W, H*C).
    x = tf.transpose(x, perm=[0, 2, 1, 3])
    shape = x.shape
    x = Reshape((shape[1], shape[2] * shape[3]))(x)

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.25)(x)

    # +1 for CTC blank.
    x = Dense(output_dim + 1, activation="softmax", name="output")(x)
    return Model(inputs=inputs, outputs=x)


# ---------- unicode-safe image reader ----------

class UnicodeImageReader:
    """Drop-in replacement for mltu.preprocessors.ImageReader that handles
    non-ASCII file paths on Windows (cv2.imread fails on those)."""

    def __call__(self, image_path, label):
        data = np.fromfile(image_path, dtype=np.uint8)
        if data.size == 0:
            raise FileNotFoundError(f"Empty or unreadable file: {image_path}")
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to decode image: {image_path}")
        return CVImage(img), label


# ---------- keras Sequence adapter ----------

class KerasSequenceProvider(Sequence):
    """Wraps an mltu DataProvider so Keras 2.10 recognizes it as a Sequence."""

    def __init__(self, provider):
        self.provider = provider

    def __len__(self):
        return len(self.provider)

    def __getitem__(self, index):
        return self.provider[index]

    def on_epoch_end(self):
        if hasattr(self.provider, "on_epoch_end"):
            self.provider.on_epoch_end()


# ---------- main ----------

def main():
    samples, vocab_list, max_word_length = load_samples()
    print(f"Loaded {len(samples)} samples, vocab size={len(vocab_list)}, max_word_length={max_word_length}")

    vocab = "".join(vocab_list)
    height = 32
    width = 256  # 256 -> 64 timesteps after two stride-2 blocks; must be >= 2*max_label-1
    batch_size = 64
    learning_rate = 1e-3
    train_epochs = 30
    train_workers = 4

    # Write configs.yaml manually (inference only needs vocab/height/width).
    with open(OUT_DIR / "configs.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "vocab": vocab,
                "height": height,
                "width": width,
                "max_text_length": max_word_length,
            },
            fh,
        )

    data_provider = DataProvider(
        dataset=samples,
        skip_validation=True,
        batch_size=batch_size,
        data_preprocessors=[UnicodeImageReader()],
        transformers=[
            ImageResizer(width, height, keep_aspect_ratio=False),
            LabelIndexer(vocab),
            LabelPadding(max_word_length=max_word_length, padding_value=len(vocab)),
        ],
    )

    train_provider, val_provider = data_provider.split(split=0.9)
    train_seq = KerasSequenceProvider(train_provider)
    val_seq = KerasSequenceProvider(val_provider)

    model = build_model(
        input_dim=(height, width, 3),
        output_dim=len(vocab),
    )
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=CTCloss(),
        metrics=[
            CERMetric(vocabulary=vocab),
            WERMetric(vocabulary=vocab),
        ],
        run_eagerly=False,
    )
    model.summary(line_length=110)

    callbacks = [
        EarlyStopping(monitor="val_CER", patience=10, verbose=1, mode="min"),
        ModelCheckpoint(
            filepath=str(OUT_DIR / "model.h5"),
            monitor="val_CER",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=5, verbose=1, mode="min"),
        TensorBoard(log_dir=str(OUT_DIR / "logs"), update_freq="epoch"),
        TrainLogger(str(OUT_DIR)),
        Model2onnx(
            saved_model_path=str(OUT_DIR / "model.h5"),
            metadata={"vocab": vocab},
        ),
    ]

    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=train_epochs,
        callbacks=callbacks,
        workers=train_workers,
    )

    print(f"\nDone. ONNX model: {OUT_DIR / 'model.onnx'}")
    print(f"Configs: {OUT_DIR / 'configs.yaml'}")


if __name__ == "__main__":
    main()
