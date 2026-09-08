import os
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf

from model_question_aware import build_inference_models


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "data/processed/question_aware"

MODEL_PATH = "models/seq2seq_question_aware_best.keras"

VOCAB_PATH = "data/processed/vocab.pkl"

TEST_CSV_PATH = "data/processed/test.csv"


# ============================================================
# LOAD VOCABULARY
# ============================================================

print("Loading vocabulary...")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

tokenizer = vocab["tokenizer"]

vocab_size = vocab["vocab_size"]


# ============================================================
# SPECIAL TOKEN IDs
# ============================================================
# vocab.pkl stores the NAMES of the special tokens:
#
# "<sos>"
# "<eos>"
# "<pad>"
#
# The model needs their INTEGER IDs.
# ============================================================

sos_token = tokenizer.word_index[
    vocab["sos_token"]
]

eos_token = tokenizer.word_index[
    vocab["eos_token"]
]

# Padding ID is 0 because mask_zero=True
pad_token = 0


print("Vocabulary size:", vocab_size)
print("SOS ID:", sos_token)
print("EOS ID:", eos_token)
print("PAD ID:", pad_token)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

encoder_input = np.load(
    os.path.join(
        DATA_DIR,
        "test_encoder_input.npy"
    )
)

decoder_target = np.load(
    os.path.join(
        DATA_DIR,
        "test_decoder_target.npy"
    )
)


print(
    "Encoder input shape:",
    encoder_input.shape
)

print(
    "Decoder target shape:",
    decoder_target.shape
)


# ============================================================
# LOAD TEST CSV
# ============================================================

test_df = pd.read_csv(
    TEST_CSV_PATH
)


# Actual answers from CSV
answers = (
    test_df["answer"]
    .astype(str)
    .str.lower()
    .str.strip()
    .tolist()
)


# ============================================================
# CHECK DATA ALIGNMENT
# ============================================================

if len(encoder_input) != len(answers):

    raise ValueError(
        "Mismatch between encoder inputs "
        f"({len(encoder_input)}) and answers "
        f"({len(answers)})"
    )


print(
    "Number of test samples:",
    len(answers)
)


# ============================================================
# DETERMINE MODEL DIMENSIONS
# ============================================================

encoder_len = encoder_input.shape[1]

decoder_len = decoder_target.shape[1]


print("\nModel dimensions:")
print(
    "Encoder length:",
    encoder_len
)

print(
    "Decoder length:",
    decoder_len
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print(
    "\nLoading trained question-aware model..."
)

training_model = tf.keras.models.load_model(
    MODEL_PATH
)

print(
    "Training model loaded successfully."
)


# ============================================================
# BUILD INFERENCE MODELS
# ============================================================

print(
    "\nBuilding inference models..."
)

encoder_model, decoder_model = (
    build_inference_models(
        vocab_size=vocab_size,
        encoder_len=encoder_len,
        embedding_dim=64,
        latent_dim=128
    )
)


# ============================================================
# TRANSFER WEIGHTS BY LAYER NAME
# ============================================================

print(
    "\nTransferring weights..."
)

transferred_layers = []


# ------------------------------------------------------------
# Encoder weights
# ------------------------------------------------------------

for inference_layer in encoder_model.layers:

    try:

        training_layer = (
            training_model.get_layer(
                inference_layer.name
            )
        )

        if inference_layer.get_weights():

            inference_layer.set_weights(
                training_layer.get_weights()
            )

            transferred_layers.append(
                inference_layer.name
            )

    except ValueError:

        pass


# ------------------------------------------------------------
# Decoder weights
# ------------------------------------------------------------

for inference_layer in decoder_model.layers:

    try:

        training_layer = (
            training_model.get_layer(
                inference_layer.name
            )
        )

        if inference_layer.get_weights():

            inference_layer.set_weights(
                training_layer.get_weights()
            )

            if inference_layer.name not in transferred_layers:

                transferred_layers.append(
                    inference_layer.name
                )

    except ValueError:

        pass


print(
    "Transferred layers:",
    transferred_layers
)


# ============================================================
# VERIFY REQUIRED WEIGHTS
# ============================================================

required_layers = [
    "encoder_embedding",
    "encoder_lstm",
    "decoder_embedding",
    "decoder_lstm",
    "output"
]


missing_layers = [
    layer
    for layer in required_layers
    if layer not in transferred_layers
]


if missing_layers:

    raise RuntimeError(
        "The following model layers did not receive "
        f"weights: {missing_layers}"
    )


print(
    "All required model weights transferred successfully."
)


# ============================================================
# ID -> WORD MAPPING
# ============================================================

id_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}


# ============================================================
# GREEDY DECODING
# ============================================================

def decode_batch(
    encoder_inputs,
    batch_size=128
):
    """
    Generate answers using greedy decoding.

    The encoder receives:
        Context + Question

    The decoder generates:
        Answer
    """

    all_predictions = []

    total_samples = len(
        encoder_inputs
    )


    # ========================================================
    # PROCESS TEST SET IN BATCHES
    # ========================================================

    for start in range(
        0,
        total_samples,
        batch_size
    ):

        end = min(
            start + batch_size,
            total_samples
        )

        batch = encoder_inputs[
            start:end
        ]


        # ----------------------------------------------------
        # ENCODE CONTEXT + QUESTION
        # ----------------------------------------------------

        state_h, state_c = (
            encoder_model.predict(
                batch,
                verbose=0
            )
        )


        current_batch_size = (
            batch.shape[0]
        )


        # ----------------------------------------------------
        # START WITH <SOS>
        # ----------------------------------------------------

        current_token = np.full(
            (
                current_batch_size,
                1
            ),
            sos_token,
            dtype=np.int32
        )


        # Keep track of sequences that already generated <EOS>
        finished = np.zeros(
            current_batch_size,
            dtype=bool
        )


        # Store generated token IDs
        batch_predictions = [
            []
            for _ in range(
                current_batch_size
            )
        ]


        # ----------------------------------------------------
        # GENERATE ANSWER TOKENS
        # ----------------------------------------------------
        #
        # Decoder target length is 5.
        # We allow a few extra steps to safely reach <EOS>.
        # ----------------------------------------------------

        max_decode_steps = (
            decoder_len + 5
        )


        for step in range(
            max_decode_steps
        ):

            output_tokens, state_h, state_c = (
                decoder_model.predict(
                    [
                        current_token,
                        state_h,
                        state_c
                    ],
                    verbose=0
                )
            )


            # ------------------------------------------------
            # Select most probable token
            # ------------------------------------------------

            next_token = np.argmax(
                output_tokens[:, -1, :],
                axis=-1
            )


            # ------------------------------------------------
            # Process each sample
            # ------------------------------------------------

            for i, token_id in enumerate(
                next_token
            ):

                # Already finished
                if finished[i]:

                    continue


                token_id = int(
                    token_id
                )


                # <EOS>
                if token_id == eos_token:

                    finished[i] = True


                # Ignore padding
                elif token_id != pad_token:

                    batch_predictions[i].append(
                        token_id
                    )


            # ------------------------------------------------
            # Current prediction becomes next decoder input
            # ------------------------------------------------

            current_token = (
                next_token.reshape(
                    -1,
                    1
                )
            )


            # ------------------------------------------------
            # Stop if every sequence finished
            # ------------------------------------------------

            if np.all(
                finished
            ):

                break


        # ====================================================
        # CONVERT TOKEN IDs TO WORDS
        # ====================================================

        for token_ids in batch_predictions:

            words = []

            for token_id in token_ids:

                word = id_to_word.get(
                    int(token_id),
                    ""
                )

                if word:

                    words.append(
                        word
                    )


            prediction = (
                " ".join(words)
                .strip()
                .lower()
            )


            all_predictions.append(
                prediction
            )


    return all_predictions


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)

predictions = decode_batch(
    encoder_input
)


# ============================================================
# CHECK PREDICTION COUNT
# ============================================================

if len(predictions) != len(answers):

    raise ValueError(
        "Prediction count does not match "
        f"answer count: "
        f"{len(predictions)} vs {len(answers)}"
    )


# ============================================================
# EXACT-MATCH EVALUATION
# ============================================================

correct = 0

for predicted, actual in zip(
    predictions,
    answers
):

    if predicted == actual:

        correct += 1


total = len(
    answers
)

incorrect = (
    total - correct
)

accuracy = (
    correct / total
) * 100


# ============================================================
# MAIN RESULTS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "QUESTION-AWARE MODEL RESULTS"
)

print(
    "=" * 60
)

print(
    f"Correct: {correct}/{total}"
)

print(
    f"Incorrect: {incorrect}/{total}"
)

print(
    f"Exact-match accuracy: {accuracy:.2f}%"
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "SAMPLE PREDICTIONS"
)

print(
    "=" * 60
)

for i in range(
    min(20, total)
):

    if predictions[i] == answers[i]:

        status = "CORRECT"

    else:

        status = "WRONG"


    print(
        f"{i:4d} | "
        f"Predicted: {predictions[i]:12s} | "
        f"Actual: {answers[i]:12s} | "
        f"{status}"
    )


# ============================================================
# PER-ANSWER ACCURACY
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PER-ANSWER ACCURACY"
)

print(
    "=" * 60
)


unique_answers = sorted(
    set(answers)
)


for answer in unique_answers:

    indices = [
        i
        for i, actual in enumerate(
            answers
        )
        if actual == answer
    ]


    class_correct = sum(
        predictions[i] == answers[i]
        for i in indices
    )


    class_total = len(
        indices
    )


    class_accuracy = (
        class_correct /
        class_total
    ) * 100


    print(
        f"{answer:10s}: "
        f"{class_correct}/{class_total} "
        f"({class_accuracy:.2f}%)"
    )


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "PREDICTION DISTRIBUTION"
)

print(
    "=" * 60
)


from collections import Counter


prediction_counts = Counter(
    predictions
)


for answer, count in sorted(
    prediction_counts.items()
):

    print(
        f"{answer:12s}: {count}"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n" + "=" * 60
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 60
)

print(
    "Original Seq2Seq LSTM:       52.50%"
)

print(
    "Seq2Seq + Attention:        51.40%"
)

print(
    f"Question-Aware Seq2Seq:     {accuracy:.2f}%"
)

print(
    "=" * 60
)

print(
    "\nEvaluation complete."
)