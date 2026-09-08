import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    AdditiveAttention,
    Concatenate
)


# ============================================================
# PATHS
# ============================================================

SEQ_DIR = "data/processed/sequences"
VOCAB_PATH = "data/processed/vocab.pkl"

BASELINE_WEIGHTS = "models/seq2seq_best.keras"
ATTENTION_WEIGHTS = "models/seq2seq_attention_best.keras"

BATCH_SIZE = 64


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

X_test = np.load(
    os.path.join(SEQ_DIR, "test_encoder_input.npy")
)

Y_test = np.load(
    os.path.join(SEQ_DIR, "test_decoder_target.npy")
)

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

tokenizer = vocab["tokenizer"]

vocab_size = vocab["vocab_size"]

encoder_len = X_test.shape[1]
decoder_len = Y_test.shape[1]

sos_id = tokenizer.word_index[vocab["sos_token"]]
eos_id = tokenizer.word_index[vocab["eos_token"]]
pad_id = tokenizer.word_index.get(
    vocab["pad_token"],
    0
)

print("Test encoder shape:", X_test.shape)
print("Test decoder shape:", Y_test.shape)
print("Vocabulary size:", vocab_size)
print("Encoder length:", encoder_len)
print("Decoder length:", decoder_len)

print("SOS ID:", sos_id)
print("EOS ID:", eos_id)
print("PAD ID:", pad_id)


# ============================================================
# BASELINE MODEL
# ============================================================

def build_baseline_models(
    vocab_size,
    encoder_len,
    decoder_len,
    embedding_dim=64,
    latent_dim=128
):

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoder_inputs = Input(
        shape=(encoder_len,),
        name="encoder_input"
    )

    encoder_embedding = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="encoder_embedding"
    )(encoder_inputs)

    encoder_lstm = LSTM(
        latent_dim,
        return_state=True,
        name="encoder_lstm"
    )

    _, state_h, state_c = encoder_lstm(
        encoder_embedding
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder_inputs = Input(
        shape=(decoder_len,),
        name="decoder_input"
    )

    decoder_embedding = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=[state_h, state_c]
    )

    decoder_dense = Dense(
        vocab_size,
        activation="softmax",
        name="decoder_dense"
    )

    outputs = decoder_dense(
        decoder_outputs
    )

    training_model = Model(
        [encoder_inputs, decoder_inputs],
        outputs
    )

    # --------------------------------------------------------
    # Inference Encoder
    # --------------------------------------------------------

    encoder_model = Model(
        encoder_inputs,
        [state_h, state_c]
    )

    # --------------------------------------------------------
    # Inference Decoder
    # --------------------------------------------------------

    decoder_token_input = Input(
        shape=(1,),
        name="decoder_token_input"
    )

    decoder_state_h = Input(
        shape=(latent_dim,),
        name="decoder_state_h"
    )

    decoder_state_c = Input(
        shape=(latent_dim,),
        name="decoder_state_c"
    )

    decoder_token_embedding = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="inference_decoder_embedding"
    )(decoder_token_input)

    decoder_output, new_h, new_c = decoder_lstm(
        decoder_token_embedding,
        initial_state=[
            decoder_state_h,
            decoder_state_c
        ]
    )

    decoder_probs = decoder_dense(
        decoder_output
    )

    decoder_model = Model(
        [
            decoder_token_input,
            decoder_state_h,
            decoder_state_c
        ],
        [
            decoder_probs,
            new_h,
            new_c
        ]
    )

    return (
        training_model,
        encoder_model,
        decoder_model
    )


# ============================================================
# ATTENTION MODEL
# ============================================================

def build_attention_models(
    vocab_size,
    encoder_len,
    decoder_len,
    embedding_dim=64,
    latent_dim=128
):

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoder_inputs = Input(
        shape=(encoder_len,),
        name="encoder_input"
    )

    encoder_embedding = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="encoder_embedding"
    )(encoder_inputs)

    encoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm"
    )

    encoder_outputs, state_h, state_c = encoder_lstm(
        encoder_embedding
    )

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    decoder_inputs = Input(
        shape=(decoder_len,),
        name="decoder_input"
    )

    decoder_embedding = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="decoder_embedding"
    )(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )

    decoder_outputs, _, _ = decoder_lstm(
        decoder_embedding,
        initial_state=[
            state_h,
            state_c
        ]
    )

    # --------------------------------------------------------
    # Attention
    # --------------------------------------------------------

    attention_layer = AdditiveAttention(
        name="attention"
    )

    attention_context = attention_layer(
        [
            decoder_outputs,
            encoder_outputs
        ]
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = Concatenate(
        axis=-1,
        name="concatenate"
    )(
        [
            decoder_outputs,
            attention_context
        ]
    )

    decoder_dense = Dense(
        vocab_size,
        activation="softmax",
        name="decoder_dense"
    )

    outputs = decoder_dense(
        combined
    )

    training_model = Model(
        [
            encoder_inputs,
            decoder_inputs
        ],
        outputs
    )

    # --------------------------------------------------------
    # Inference Encoder
    # --------------------------------------------------------

    encoder_model = Model(
        encoder_inputs,
        [
            encoder_outputs,
            state_h,
            state_c
        ]
    )

    # --------------------------------------------------------
    # Inference Decoder
    # --------------------------------------------------------

    decoder_token_input = Input(
        shape=(1,),
        name="decoder_token_input"
    )

    decoder_state_h = Input(
        shape=(latent_dim,),
        name="decoder_state_h"
    )

    decoder_state_c = Input(
        shape=(latent_dim,),
        name="decoder_state_c"
    )

    encoder_outputs_input = Input(
        shape=(encoder_len, latent_dim),
        name="encoder_outputs_input"
    )

    decoder_token_embedding = Embedding(
        vocab_size,
        embedding_dim,
        mask_zero=True,
        name="inference_decoder_embedding"
    )(decoder_token_input)

    decoder_output, new_h, new_c = decoder_lstm(
        decoder_token_embedding,
        initial_state=[
            decoder_state_h,
            decoder_state_c
        ]
    )

    attention_context = attention_layer(
        [
            decoder_output,
            encoder_outputs_input
        ]
    )

    combined = Concatenate(
        axis=-1
    )(
        [
            decoder_output,
            attention_context
        ]
    )

    decoder_probs = decoder_dense(
        combined
    )

    decoder_model = Model(
        [
            decoder_token_input,
            decoder_state_h,
            decoder_state_c,
            encoder_outputs_input
        ],
        [
            decoder_probs,
            new_h,
            new_c
        ]
    )

    return (
        training_model,
        encoder_model,
        decoder_model
    )


# ============================================================
# BASELINE GREEDY DECODING
# ============================================================

def predict_baseline(
    encoder_model,
    decoder_model,
    X
):

    predictions = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        batch = X[
            start:start + BATCH_SIZE
        ]

        state_h, state_c = encoder_model.predict(
            batch,
            verbose=0
        )

        target = np.full(
            (len(batch), 1),
            sos_id,
            dtype=np.int32
        )

        batch_predictions = np.zeros(
            (len(batch), decoder_len),
            dtype=np.int32
        )

        for t in range(decoder_len):

            probs, state_h, state_c = decoder_model.predict(
                [
                    target,
                    state_h,
                    state_c
                ],
                verbose=0
            )

            next_token = np.argmax(
                probs[:, -1, :],
                axis=-1
            )

            batch_predictions[:, t] = next_token

            target[:, 0] = next_token

        predictions.append(
            batch_predictions
        )

    return np.concatenate(
        predictions,
        axis=0
    )


# ============================================================
# ATTENTION GREEDY DECODING
# ============================================================

def predict_attention(
    encoder_model,
    decoder_model,
    X
):

    predictions = []

    for start in range(
        0,
        len(X),
        BATCH_SIZE
    ):

        batch = X[
            start:start + BATCH_SIZE
        ]

        (
            encoder_outputs,
            state_h,
            state_c
        ) = encoder_model.predict(
            batch,
            verbose=0
        )

        target = np.full(
            (len(batch), 1),
            sos_id,
            dtype=np.int32
        )

        batch_predictions = np.zeros(
            (len(batch), decoder_len),
            dtype=np.int32
        )

        for t in range(decoder_len):

            (
                probs,
                state_h,
                state_c
            ) = decoder_model.predict(
                [
                    target,
                    state_h,
                    state_c,
                    encoder_outputs
                ],
                verbose=0
            )

            next_token = np.argmax(
                probs[:, -1, :],
                axis=-1
            )

            batch_predictions[:, t] = next_token

            target[:, 0] = next_token

        predictions.append(
            batch_predictions
        )

    return np.concatenate(
        predictions,
        axis=0
    )


# ============================================================
# EXACT MATCH
# ============================================================

def get_exact_match(
    predictions,
    targets
):

    correct = []

    for pred, true in zip(
        predictions,
        targets
    ):

        pred_tokens = [
            int(x)
            for x in pred
            if x not in (
                pad_id,
                sos_id,
                eos_id
            )
        ]

        true_tokens = [
            int(x)
            for x in true
            if x not in (
                pad_id,
                sos_id,
                eos_id
            )
        ]

        correct.append(
            pred_tokens == true_tokens
        )

    return np.array(correct)


# ============================================================
# BUILD BASELINE
# ============================================================

print("\nBuilding baseline model...")

(
    baseline_training,
    baseline_encoder,
    baseline_decoder
) = build_baseline_models(
    vocab_size,
    encoder_len,
    decoder_len
)

print("Loading baseline weights...")

baseline_training.load_weights(
    BASELINE_WEIGHTS
)

print("Baseline loaded successfully.")


# ============================================================
# BUILD ATTENTION
# ============================================================

print("\nBuilding attention model...")

(
    attention_training,
    attention_encoder,
    attention_decoder
) = build_attention_models(
    vocab_size,
    encoder_len,
    decoder_len
)

print("Loading attention weights...")

attention_training.load_weights(
    ATTENTION_WEIGHTS
)

print("Attention model loaded successfully.")


# ============================================================
# RUN BASELINE
# ============================================================

print("\nRunning baseline predictions...")

baseline_predictions = predict_baseline(
    baseline_encoder,
    baseline_decoder,
    X_test
)

print("Baseline predictions complete.")


# ============================================================
# RUN ATTENTION
# ============================================================

print("\nRunning attention predictions...")

attention_predictions = predict_attention(
    attention_encoder,
    attention_decoder,
    X_test
)

print("Attention predictions complete.")


# ============================================================
# CALCULATE CORRECTNESS
# ============================================================

baseline_correct = get_exact_match(
    baseline_predictions,
    Y_test
)

attention_correct = get_exact_match(
    attention_predictions,
    Y_test
)


# ============================================================
# COMPARISON
# ============================================================

both_correct = np.sum(
    baseline_correct &
    attention_correct
)

baseline_only = np.sum(
    baseline_correct &
    ~attention_correct
)

attention_only = np.sum(
    ~baseline_correct &
    attention_correct
)

both_wrong = np.sum(
    ~baseline_correct &
    ~attention_correct
)


baseline_accuracy = (
    np.mean(baseline_correct) * 100
)

attention_accuracy = (
    np.mean(attention_correct) * 100
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("BASELINE vs ATTENTION")
print("=" * 60)

print(
    f"Baseline correct:        "
    f"{np.sum(baseline_correct)}/1000 "
    f"({baseline_accuracy:.2f}%)"
)

print(
    f"Attention correct:       "
    f"{np.sum(attention_correct)}/1000 "
    f"({attention_accuracy:.2f}%)"
)

print()

print(
    f"Both correct:            "
    f"{both_correct}"
)

print(
    f"Baseline only correct:   "
    f"{baseline_only}"
)

print(
    f"Attention only correct:  "
    f"{attention_only}"
)

print(
    f"Both wrong:              "
    f"{both_wrong}"
)

print("=" * 60)


# ============================================================
# CHECK TOTAL
# ============================================================

total = (
    both_correct +
    baseline_only +
    attention_only +
    both_wrong
)

print(
    f"\nCheck: {total}/1000 examples accounted for."
)


# ============================================================
# ATTENTION FIXED BASELINE ERRORS
# ============================================================

print("\n")
print("=" * 60)
print("EXAMPLES WHERE ATTENTION FIXED BASELINE")
print("=" * 60)

count = 0

for i in range(len(X_test)):

    if (
        not baseline_correct[i]
        and attention_correct[i]
    ):

        print(f"\nExample {i}")

        print(
            "Baseline prediction IDs:",
            baseline_predictions[i].tolist()
        )

        print(
            "Attention prediction IDs:",
            attention_predictions[i].tolist()
        )

        print(
            "Target IDs:",
            Y_test[i].tolist()
        )

        count += 1

        if count >= 10:
            break


# ============================================================
# BASELINE CORRECT / ATTENTION WRONG
# ============================================================

print("\n")
print("=" * 60)
print("EXAMPLES WHERE BASELINE WAS CORRECT")
print("BUT ATTENTION FAILED")
print("=" * 60)

count = 0

for i in range(len(X_test)):

    if (
        baseline_correct[i]
        and not attention_correct[i]
    ):

        print(f"\nExample {i}")

        print(
            "Baseline prediction IDs:",
            baseline_predictions[i].tolist()
        )

        print(
            "Attention prediction IDs:",
            attention_predictions[i].tolist()
        )

        print(
            "Target IDs:",
            Y_test[i].tolist()
        )

        count += 1

        if count >= 10:
            break


print("\nComparison complete.")