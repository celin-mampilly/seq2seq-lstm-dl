"""
train.py
Loads the preprocessed .npy sequence arrays + vocab.pkl produced by
data/prepare_sequences.py, builds the Seq2Seq LSTM model from model.py,
and trains it with teacher forcing.

Run from the project root:
    python train.py
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

from model import build_training_model

# ---------------------------------------------------------------------------
# Paths (match your existing project layout)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SEQ_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "sequences")
VOCAB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "vocab.pkl")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
LATENT_DIM = 128
BATCH_SIZE = 64
EPOCHS = 50
VALIDATION_SPLIT_FROM_FILES = True  # we have a real val set on disk


def load_split(split_name):
    """Loads encoder_input, decoder_input, decoder_target for one split."""
    enc_in = np.load(os.path.join(SEQ_DIR, f"{split_name}_encoder_input.npy"))
    dec_in = np.load(os.path.join(SEQ_DIR, f"{split_name}_decoder_input.npy"))
    dec_tgt = np.load(os.path.join(SEQ_DIR, f"{split_name}_decoder_target.npy"))

    # decoder_target arrives as (N, T, 1) from the prep step; sparse
    # categorical crossentropy expects integer targets shaped (N, T).
    if dec_tgt.ndim == 3 and dec_tgt.shape[-1] == 1:
        dec_tgt = np.squeeze(dec_tgt, axis=-1)

    return enc_in, dec_in, dec_tgt


def main():
    # ---- Load vocab / config ----
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)

    # Adjust key names here if your vocab.pkl stores them differently.
    tokenizer = vocab_data["tokenizer"]
    vocab_size = len(tokenizer.word_index) + 1  # +1 for the 0 padding index

    print(f"Vocab size: {vocab_size}")

    # ---- Load data ----
    train_enc_in, train_dec_in, train_dec_tgt = load_split("train")
    val_enc_in, val_dec_in, val_dec_tgt = load_split("val")

    # Derive sequence lengths from the actual arrays rather than vocab.pkl —
    # vocab.pkl's stored max_encoder_len/max_decoder_len can drift out of
    # sync with what prepare_sequences.py actually padded to.
    max_encoder_len = train_enc_in.shape[1]
    max_decoder_len = train_dec_in.shape[1]
    print(f"max_encoder_len={max_encoder_len}, max_decoder_len={max_decoder_len}")

    print("train_encoder_input:", train_enc_in.shape)
    print("train_decoder_input:", train_dec_in.shape)
    print("train_decoder_target:", train_dec_tgt.shape)

    # Sanity check: val split must share the same padded lengths as train,
    # since they're fed into the same fixed-shape Input layers.
    assert val_enc_in.shape[1] == max_encoder_len, (
        f"val encoder length {val_enc_in.shape[1]} != train {max_encoder_len}"
    )
    assert val_dec_in.shape[1] == max_decoder_len, (
        f"val decoder length {val_dec_in.shape[1]} != train {max_decoder_len}"
    )

    # ---- Build model ----
    model, layers = build_training_model(
        vocab_size=vocab_size,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        embedding_dim=EMBEDDING_DIM,
        latent_dim=LATENT_DIM,
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # ---- Callbacks ----
    checkpoint_path = os.path.join(MODEL_DIR, "seq2seq_best.keras")
    callbacks = [
        ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, verbose=1
        ),
        EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        CSVLogger(os.path.join(MODEL_DIR, "training_log.csv")),
    ]

    # ---- Train ----
    history = model.fit(
        [train_enc_in, train_dec_in],
        train_dec_tgt,
        validation_data=([val_enc_in, val_dec_in], val_dec_tgt),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # ---- Save final model (best weights already restored by EarlyStopping) ----
    final_path = os.path.join(MODEL_DIR, "seq2seq_final.keras")
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    print(f"Best checkpoint saved to {checkpoint_path}")

    return model, layers, history


if __name__ == "__main__":
    main()