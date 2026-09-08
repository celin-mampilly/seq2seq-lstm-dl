import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed"
CSV_DIR = "data/processed"
OUTPUT_DIR = "data/processed/question_aware"

# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD TOKENIZER
# ============================================================

vocab_path = os.path.join(DATA_DIR, "vocab.pkl")

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

tokenizer = vocab["tokenizer"]

print("Tokenizer loaded.")
print("Vocabulary size:", vocab["vocab_size"])


# ============================================================
# FUNCTION TO PREPARE A DATASET
# ============================================================

def prepare_split(csv_filename):

    csv_path = os.path.join(CSV_DIR, csv_filename)

    print(f"\nReading: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check that required columns exist
    required_columns = ["context", "question", "answer"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found in {csv_filename}. "
                f"Available columns: {list(df.columns)}"
            )

    # --------------------------------------------------------
    # Combine CONTEXT + QUESTION
    #
    # Example:
    #
    # Context:
    # John travelled to the hallway.
    # Mary journeyed to the bathroom.
    #
    # Question:
    # Where is John?
    #
    # Becomes:
    #
    # John travelled to the hallway. Mary journeyed to the
    # bathroom. Where is John?
    # --------------------------------------------------------

    combined_text = (
        df["context"].astype(str)
        + " "
        + df["question"].astype(str)
    )

    # Convert text to token IDs
    sequences = tokenizer.texts_to_sequences(combined_text)

    print("Number of samples:", len(sequences))

    return sequences, df


# ============================================================
# PREPARE TRAIN / VALIDATION / TEST
# ============================================================

train_sequences, train_df = prepare_split("train.csv")
val_sequences, val_df = prepare_split("validation.csv")
test_sequences, test_df = prepare_split("test.csv")


# ============================================================
# FIND MAXIMUM SEQUENCE LENGTH
# ============================================================

train_max = max(len(seq) for seq in train_sequences)
val_max = max(len(seq) for seq in val_sequences)
test_max = max(len(seq) for seq in test_sequences)

max_encoder_len = max(
    train_max,
    val_max,
    test_max
)

print("\nMaximum sequence lengths:")
print("Train:", train_max)
print("Validation:", val_max)
print("Test:", test_max)
print("Final encoder length:", max_encoder_len)


# ============================================================
# PAD SEQUENCES
# ============================================================

train_encoder = pad_sequences(
    train_sequences,
    maxlen=max_encoder_len,
    padding="post",
    truncating="post"
)

val_encoder = pad_sequences(
    val_sequences,
    maxlen=max_encoder_len,
    padding="post",
    truncating="post"
)

test_encoder = pad_sequences(
    test_sequences,
    maxlen=max_encoder_len,
    padding="post",
    truncating="post"
)


# ============================================================
# SAVE QUESTION-AWARE ENCODER INPUTS
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "train_encoder_input.npy"
    ),
    train_encoder
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "val_encoder_input.npy"
    ),
    val_encoder
)

np.save(
    os.path.join(
        OUTPUT_DIR,
        "test_encoder_input.npy"
    ),
    test_encoder
)


# ============================================================
# COPY DECODER INPUTS AND TARGETS
#
# The answer does NOT change.
# Only the encoder input changes.
# ============================================================

sequence_dir = os.path.join(
    DATA_DIR,
    "sequences"
)

decoder_files = [
    "train_decoder_input.npy",
    "train_decoder_target.npy",
    "val_decoder_input.npy",
    "val_decoder_target.npy",
    "test_decoder_input.npy",
    "test_decoder_target.npy"
]

for filename in decoder_files:

    source_path = os.path.join(
        sequence_dir,
        filename
    )

    destination_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    if os.path.exists(source_path):

        data = np.load(source_path)

        np.save(
            destination_path,
            data
        )

        print(f"Copied: {filename} {data.shape}")

    else:

        print(
            f"WARNING: {filename} not found at "
            f"{source_path}"
        )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("QUESTION-AWARE DATA PREPARATION COMPLETED")
print("=" * 60)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nEncoder shapes:")
print("Train:", train_encoder.shape)
print("Validation:", val_encoder.shape)
print("Test:", test_encoder.shape)

print("\nSaved files:")
for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", filename)

print("\nDone!")