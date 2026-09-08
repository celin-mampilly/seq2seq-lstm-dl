import os
import numpy as np
import pickle
import tensorflow as tf

from model_question_aware import build_training_model


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed/question_aware"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open("data/processed/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

vocab_size = vocab["vocab_size"]


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("Loading data...")

train_encoder = np.load(
    os.path.join(DATA_DIR, "train_encoder_input.npy")
)

train_decoder_input = np.load(
    os.path.join(DATA_DIR, "train_decoder_input.npy")
)

train_decoder_target = np.load(
    os.path.join(DATA_DIR, "train_decoder_target.npy")
)

val_encoder = np.load(
    os.path.join(DATA_DIR, "val_encoder_input.npy")
)

val_decoder_input = np.load(
    os.path.join(DATA_DIR, "val_decoder_input.npy")
)

val_decoder_target = np.load(
    os.path.join(DATA_DIR, "val_decoder_target.npy")
)


# ============================================================
# PRINT SHAPES
# ============================================================

print("\nData shapes:")

print(
    "Train encoder:",
    train_encoder.shape
)

print(
    "Train decoder input:",
    train_decoder_input.shape
)

print(
    "Train decoder target:",
    train_decoder_target.shape
)

print(
    "Validation encoder:",
    val_encoder.shape
)

print(
    "Validation decoder input:",
    val_decoder_input.shape
)

print(
    "Validation decoder target:",
    val_decoder_target.shape
)


# ============================================================
# DETERMINE SEQUENCE LENGTHS
# ============================================================

encoder_len = train_encoder.shape[1]
decoder_len = train_decoder_input.shape[1]


print("\nModel configuration:")
print("Vocabulary size:", vocab_size)
print("Encoder length:", encoder_len)
print("Decoder length:", decoder_len)


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding question-aware Seq2Seq model...")

model = build_training_model(
    vocab_size=vocab_size,
    encoder_len=encoder_len,
    decoder_len=decoder_len,
    embedding_dim=64,
    latent_dim=128
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

best_model_path = os.path.join(
    MODEL_DIR,
    "seq2seq_question_aware_best.keras"
)

final_model_path = os.path.join(
    MODEL_DIR,
    "seq2seq_question_aware_final.keras"
)

log_path = os.path.join(
    MODEL_DIR,
    "question_aware_training_log.csv"
)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)


csv_logger = tf.keras.callbacks.CSVLogger(
    log_path
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting training...\n")

history = model.fit(
    [train_encoder, train_decoder_input],
    train_decoder_target,
    validation_data=(
        [val_encoder, val_decoder_input],
        val_decoder_target
    ),
    batch_size=64,
    epochs=50,
    callbacks=[
        checkpoint,
        early_stopping,
        csv_logger
    ],
    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(final_model_path)


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("QUESTION-AWARE TRAINING COMPLETED")
print("=" * 60)

print("\nBest model:")
print(best_model_path)

print("\nFinal model:")
print(final_model_path)

print("\nTraining log:")
print(log_path)

print("\nBest validation loss:")

best_epoch = np.argmin(
    history.history["val_loss"]
)

print(
    "Epoch:",
    best_epoch + 1
)

print(
    "Validation loss:",
    history.history["val_loss"][best_epoch]
)

print(
    "Validation accuracy:",
    history.history["val_accuracy"][best_epoch]
)