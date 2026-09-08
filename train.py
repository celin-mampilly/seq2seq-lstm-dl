"""
train.py
Loads the preprocessed .npy sequence arrays + vocab.pkl produced by
data/prepare_sequences.py, builds the Seq2Seq LSTM model with attention
from model.py, and trains it with teacher forcing.

Run from the project root:
    python train.py
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
)

from model import build_training_model


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SEQ_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "sequences",
)

VOCAB_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "vocab.pkl",
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models",
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 64
LATENT_DIM = 128

BATCH_SIZE = 64
EPOCHS = 50

VALIDATION_SPLIT_FROM_FILES = True


# ---------------------------------------------------------------------------
# Load one dataset split
# ---------------------------------------------------------------------------
def load_split(split_name):
    """
    Loads encoder_input, decoder_input, decoder_target
    for one split.
    """

    enc_in = np.load(
        os.path.join(
            SEQ_DIR,
            f"{split_name}_encoder_input.npy"
        )
    )

    dec_in = np.load(
        os.path.join(
            SEQ_DIR,
            f"{split_name}_decoder_input.npy"
        )
    )

    dec_tgt = np.load(
        os.path.join(
            SEQ_DIR,
            f"{split_name}_decoder_target.npy"
        )
    )

    # decoder_target may be stored as:
    # (N, T, 1)
    #
    # Sparse categorical crossentropy expects:
    # (N, T)
    if dec_tgt.ndim == 3 and dec_tgt.shape[-1] == 1:
        dec_tgt = np.squeeze(
            dec_tgt,
            axis=-1
        )

    return enc_in, dec_in, dec_tgt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():

    print("=" * 70)
    print("TRAINING SEQ2SEQ LSTM WITH ADDITIVE ATTENTION")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # Load vocabulary
    # -----------------------------------------------------------------------
    with open(VOCAB_PATH, "rb") as f:
        vocab_data = pickle.load(f)

    tokenizer = vocab_data["tokenizer"]

    vocab_size = len(
        tokenizer.word_index
    ) + 1

    print(f"Vocab size: {vocab_size}")

    # -----------------------------------------------------------------------
    # Load training and validation data
    # -----------------------------------------------------------------------
    train_enc_in, train_dec_in, train_dec_tgt = load_split("train")

    val_enc_in, val_dec_in, val_dec_tgt = load_split("val")

    # -----------------------------------------------------------------------
    # Get actual sequence lengths from arrays
    # -----------------------------------------------------------------------
    max_encoder_len = train_enc_in.shape[1]
    max_decoder_len = train_dec_in.shape[1]

    print(
        f"max_encoder_len = {max_encoder_len}"
    )

    print(
        f"max_decoder_len = {max_decoder_len}"
    )

    # -----------------------------------------------------------------------
    # Display dataset shapes
    # -----------------------------------------------------------------------
    print()
    print("Training data:")
    print(
        "train_encoder_input:",
        train_enc_in.shape
    )

    print(
        "train_decoder_input:",
        train_dec_in.shape
    )

    print(
        "train_decoder_target:",
        train_dec_tgt.shape
    )

    print()
    print("Validation data:")
    print(
        "val_encoder_input:",
        val_enc_in.shape
    )

    print(
        "val_decoder_input:",
        val_dec_in.shape
    )

    print(
        "val_decoder_target:",
        val_dec_tgt.shape
    )

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------

    assert val_enc_in.shape[1] == max_encoder_len, (
        f"Validation encoder length "
        f"{val_enc_in.shape[1]} != train "
        f"{max_encoder_len}"
    )

    assert val_dec_in.shape[1] == max_decoder_len, (
        f"Validation decoder length "
        f"{val_dec_in.shape[1]} != train "
        f"{max_decoder_len}"
    )

    assert train_dec_tgt.shape == train_dec_in.shape, (
        "Training decoder input and target "
        "must have the same shape."
    )

    assert val_dec_tgt.shape == val_dec_in.shape, (
        "Validation decoder input and target "
        "must have the same shape."
    )

    # -----------------------------------------------------------------------
    # Build attention model
    # -----------------------------------------------------------------------
    print()
    print("Building attention model...")

    model, layers = build_training_model(
        vocab_size=vocab_size,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        embedding_dim=EMBEDDING_DIM,
        latent_dim=LATENT_DIM,
    )

    # model.py already compiles the model, but compile again explicitly
    # so the training configuration is clear here.
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print()
    model.summary()

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------

    # IMPORTANT:
    # Do NOT overwrite the baseline model.
    checkpoint_path = os.path.join(
        MODEL_DIR,
        "seq2seq_attention_best.keras"
    )

    log_path = os.path.join(
        MODEL_DIR,
        "attention_training_log.csv"
    )

    callbacks = [

        ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),

        CSVLogger(
            log_path
        ),
    ]

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        [train_enc_in, train_dec_in],
        train_dec_tgt,

        validation_data=(
            [val_enc_in, val_dec_in],
            val_dec_tgt,
        ),

        batch_size=BATCH_SIZE,
        epochs=EPOCHS,

        callbacks=callbacks,

        verbose=1,
    )

    # -----------------------------------------------------------------------
    # Save final attention model
    # -----------------------------------------------------------------------

    final_path = os.path.join(
        MODEL_DIR,
        "seq2seq_attention_final.keras"
    )

    model.save(final_path)

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Saved final attention model to:\n"
        f"{final_path}"
    )

    print()

    print(
        f"Best attention checkpoint saved to:\n"
        f"{checkpoint_path}"
    )

    print()

    print(
        f"Training log saved to:\n"
        f"{log_path}"
    )

    return model, layers, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()