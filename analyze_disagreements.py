import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense,
    AdditiveAttention, Concatenate
)

# ============================================================
# PATHS
# ============================================================

SEQ_DIR = "data/processed/sequences"
VOCAB_PATH = "data/processed/vocab.pkl"
TEST_CSV = "data/processed/test.csv"

BASELINE_WEIGHTS = "models/seq2seq_best.keras"
ATTENTION_WEIGHTS = "models/seq2seq_attention_best.keras"

BATCH_SIZE = 64
NUM_EXAMPLES = 10


# ============================================================
# LOAD DATA
# ============================================================

print("Loading data...")

X_test = np.load(
    f"{SEQ_DIR}/test_encoder_input.npy"
)

Y_test = np.load(
    f"{SEQ_DIR}/test_decoder_target.npy"
)

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

tokenizer = vocab["tokenizer"]

vocab_size = vocab["vocab_size"]
encoder_len = X_test.shape[1]
decoder_len = Y_test.shape[1]

sos_id = tokenizer.word_index[vocab["sos_token"]]
eos_id = tokenizer.word_index[vocab["eos_token"]]
pad_id = 0

print(f"Test samples: {len(X_test)}")
print(f"Vocabulary size: {vocab_size}")
print(f"Encoder length: {encoder_len}")
print(f"Decoder length: {decoder_len}")
print(f"SOS ID: {sos_id}")
print(f"EOS ID: {eos_id}")
print(f"PAD ID: {pad_id}")


# ============================================================
# ID → WORD
# ============================================================

id_to_word = {
    idx: word
    for word, idx in tokenizer.word_index.items()
}

id_to_word[pad_id] = "<pad>"


def ids_to_answer(ids):

    words = []

    for token_id in np.asarray(ids).reshape(-1):

        token_id = int(token_id)

        if token_id == pad_id:
            break

        if token_id == eos_id:
            break

        if token_id == sos_id:
            continue

        words.append(
            id_to_word.get(token_id, "<unknown>")
        )

    return " ".join(words)


def true_answer(target):

    return ids_to_answer(target)


# ============================================================
# BASELINE MODEL
# ============================================================

def build_baseline():

    # Encoder
    encoder_input = Input(
        shape=(encoder_len,),
        name="encoder_input"
    )

    encoder_embedding_layer = Embedding(
        vocab_size,
        64,
        mask_zero=True,
        name="encoder_embedding"
    )

    encoder_lstm_layer = LSTM(
        128,
        return_state=True,
        name="encoder_lstm"
    )

    encoder_embedded = encoder_embedding_layer(
        encoder_input
    )

    _, state_h, state_c = encoder_lstm_layer(
        encoder_embedded
    )

    # Decoder
    decoder_input = Input(
        shape=(decoder_len,),
        name="decoder_input"
    )

    decoder_embedding_layer = Embedding(
        vocab_size,
        64,
        mask_zero=True,
        name="decoder_embedding"
    )

    decoder_lstm_layer = LSTM(
        128,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )

    decoder_dense_layer = Dense(
        vocab_size,
        activation="softmax",
        name="decoder_dense"
    )

    decoder_embedded = decoder_embedding_layer(
        decoder_input
    )

    decoder_output, _, _ = decoder_lstm_layer(
        decoder_embedded,
        initial_state=[state_h, state_c]
    )

    decoder_output = decoder_dense_layer(
        decoder_output
    )

    training_model = Model(
        [encoder_input, decoder_input],
        decoder_output
    )

    # Inference encoder
    encoder_model = Model(
        encoder_input,
        [state_h, state_c]
    )

    # Inference decoder
    single_decoder_input = Input(
        shape=(1,),
        name="single_decoder_input"
    )

    decoder_state_h = Input(
        shape=(128,),
        name="decoder_state_h"
    )

    decoder_state_c = Input(
        shape=(128,),
        name="decoder_state_c"
    )

    single_embedded = decoder_embedding_layer(
        single_decoder_input
    )

    single_output, new_h, new_c = decoder_lstm_layer(
        single_embedded,
        initial_state=[
            decoder_state_h,
            decoder_state_c
        ]
    )

    single_output = decoder_dense_layer(
        single_output
    )

    decoder_model = Model(
        [
            single_decoder_input,
            decoder_state_h,
            decoder_state_c
        ],
        [
            single_output,
            new_h,
            new_c
        ]
    )

    return training_model, encoder_model, decoder_model


# ============================================================
# ATTENTION MODEL
# ============================================================

def build_attention():

    # Encoder
    encoder_input = Input(
        shape=(encoder_len,),
        name="encoder_input"
    )

    encoder_embedding_layer = Embedding(
        vocab_size,
        64,
        mask_zero=True,
        name="encoder_embedding"
    )

    encoder_lstm_layer = LSTM(
        128,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm"
    )

    encoder_embedded = encoder_embedding_layer(
        encoder_input
    )

    encoder_outputs, state_h, state_c = (
        encoder_lstm_layer(encoder_embedded)
    )

    # Decoder
    decoder_input = Input(
        shape=(decoder_len,),
        name="decoder_input"
    )

    decoder_embedding_layer = Embedding(
        vocab_size,
        64,
        mask_zero=True,
        name="decoder_embedding"
    )

    decoder_lstm_layer = LSTM(
        128,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )

    decoder_embedded = decoder_embedding_layer(
        decoder_input
    )

    decoder_outputs, _, _ = decoder_lstm_layer(
        decoder_embedded,
        initial_state=[state_h, state_c]
    )

    attention_layer = AdditiveAttention(
        name="attention"
    )

    attention_context = attention_layer(
        [
            decoder_outputs,
            encoder_outputs
        ]
    )

    concatenated = Concatenate(
        axis=-1,
        name="concat"
    )([
        decoder_outputs,
        attention_context
    ])

    decoder_dense_layer = Dense(
        vocab_size,
        activation="softmax",
        name="dense"
    )

    final_output = decoder_dense_layer(
        concatenated
    )

    training_model = Model(
        [encoder_input, decoder_input],
        final_output
    )

    # Inference encoder
    encoder_model = Model(
        encoder_input,
        [
            encoder_outputs,
            state_h,
            state_c
        ]
    )

    # Inference decoder
    single_decoder_input = Input(
        shape=(1,),
        name="decoder_single_token_input"
    )

    decoder_state_h = Input(
        shape=(128,),
        name="decoder_state_h_input"
    )

    decoder_state_c = Input(
        shape=(128,),
        name="decoder_state_c_input"
    )

    encoder_outputs_input = Input(
        shape=(encoder_len, 128),
        name="encoder_outputs_input"
    )

    single_embedded = decoder_embedding_layer(
        single_decoder_input
    )

    single_output, new_h, new_c = decoder_lstm_layer(
        single_embedded,
        initial_state=[
            decoder_state_h,
            decoder_state_c
        ]
    )

    single_attention = attention_layer(
        [
            single_output,
            encoder_outputs_input
        ]
    )

    single_concat = Concatenate(
        axis=-1
    )([
        single_output,
        single_attention
    ])

    single_probabilities = decoder_dense_layer(
        single_concat
    )

    decoder_model = Model(
        [
            single_decoder_input,
            decoder_state_h,
            decoder_state_c,
            encoder_outputs_input
        ],
        [
            single_probabilities,
            new_h,
            new_c
        ]
    )

    return training_model, encoder_model, decoder_model


# ============================================================
# BASELINE BATCHED PREDICTION
# ============================================================

def predict_baseline(
    encoder_model,
    decoder_model,
    X
):

    print("Encoding baseline test data...")

    state_h, state_c = encoder_model.predict(
        X,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # Current token for every example
    target_token = np.full(
        (len(X), 1),
        sos_id,
        dtype=np.int32
    )

    predictions = np.full(
        (len(X), decoder_len),
        pad_id,
        dtype=np.int32
    )

    for step in range(decoder_len):

        output_tokens, state_h, state_c = (
            decoder_model.predict(
                [
                    target_token,
                    state_h,
                    state_c
                ],
                batch_size=BATCH_SIZE,
                verbose=0
            )
        )

        sampled_tokens = np.argmax(
            output_tokens[:, -1, :],
            axis=-1
        )

        predictions[:, step] = sampled_tokens

        target_token = sampled_tokens.reshape(
            -1, 1
        )

    return predictions


# ============================================================
# ATTENTION BATCHED PREDICTION
# ============================================================

def predict_attention(
    encoder_model,
    decoder_model,
    X
):

    print("Encoding attention test data...")

    encoder_outputs, state_h, state_c = (
        encoder_model.predict(
            X,
            batch_size=BATCH_SIZE,
            verbose=0
        )
    )

    target_token = np.full(
        (len(X), 1),
        sos_id,
        dtype=np.int32
    )

    predictions = np.full(
        (len(X), decoder_len),
        pad_id,
        dtype=np.int32
    )

    for step in range(decoder_len):

        output_tokens, state_h, state_c = (
            decoder_model.predict(
                [
                    target_token,
                    state_h,
                    state_c,
                    encoder_outputs
                ],
                batch_size=BATCH_SIZE,
                verbose=0
            )
        )

        sampled_tokens = np.argmax(
            output_tokens[:, -1, :],
            axis=-1
        )

        predictions[:, step] = sampled_tokens

        target_token = sampled_tokens.reshape(
            -1, 1
        )

    return predictions


# ============================================================
# BUILD + LOAD
# ============================================================

print("\nBuilding baseline model...")

baseline_training, baseline_encoder, baseline_decoder = (
    build_baseline()
)

baseline_training.load_weights(
    BASELINE_WEIGHTS
)

print("Baseline weights loaded.")


print("\nBuilding attention model...")

attention_training, attention_encoder, attention_decoder = (
    build_attention()
)

attention_training.load_weights(
    ATTENTION_WEIGHTS
)

print("Attention weights loaded.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating baseline predictions...")

baseline_predictions = predict_baseline(
    baseline_encoder,
    baseline_decoder,
    X_test
)

print("Baseline predictions complete.")


print("\nGenerating attention predictions...")

attention_predictions = predict_attention(
    attention_encoder,
    attention_decoder,
    X_test
)

print("Attention predictions complete.")


# ============================================================
# LOAD CSV
# ============================================================

test_df = pd.read_csv(TEST_CSV)


# ============================================================
# COMPARE
# ============================================================

baseline_correct = 0
attention_correct = 0

both_correct = []
baseline_only = []
attention_only = []
both_wrong = []


for i in range(len(X_test)):

    correct = true_answer(Y_test[i])

    baseline_answer = ids_to_answer(
        baseline_predictions[i]
    )

    attention_answer = ids_to_answer(
        attention_predictions[i]
    )

    b_correct = baseline_answer == correct
    a_correct = attention_answer == correct

    if b_correct:
        baseline_correct += 1

    if a_correct:
        attention_correct += 1

    if b_correct and a_correct:
        both_correct.append(i)

    elif b_correct and not a_correct:
        baseline_only.append(i)

    elif not b_correct and a_correct:
        attention_only.append(i)

    else:
        both_wrong.append(i)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(
    f"Baseline correct:       "
    f"{baseline_correct}/1000 "
    f"({baseline_correct / 10:.2f}%)"
)

print(
    f"Attention correct:      "
    f"{attention_correct}/1000 "
    f"({attention_correct / 10:.2f}%)"
)

print(f"\nBoth correct:           {len(both_correct)}")
print(f"Baseline only correct:  {len(baseline_only)}")
print(f"Attention only correct: {len(attention_only)}")
print(f"Both wrong:             {len(both_wrong)}")


# ============================================================
# SHOW EXAMPLES
# ============================================================

def show_examples(title, indices):

    print("\n")
    print("=" * 70)
    print(title)
    print("=" * 70)

    for count, idx in enumerate(
        indices[:NUM_EXAMPLES],
        start=1
    ):

        row = test_df.iloc[idx]

        correct = true_answer(
            Y_test[idx]
        )

        baseline_answer = ids_to_answer(
            baseline_predictions[idx]
        )

        attention_answer = ids_to_answer(
            attention_predictions[idx]
        )

        print(
            f"\n--- Example {count} "
            f"(test index {idx}) ---"
        )

        print("\nContext:")
        print(row["context"])

        print("\nQuestion:")
        print(row["question"])

        print("\nCorrect:")
        print(correct)

        print("\nBaseline:")
        print(
            f"{baseline_answer} "
            f"{'✓' if baseline_answer == correct else '✗'}"
        )

        print("\nAttention:")
        print(
            f"{attention_answer} "
            f"{'✓' if attention_answer == correct else '✗'}"
        )


show_examples(
    "ATTENTION FIXED BASELINE",
    attention_only
)

show_examples(
    "BASELINE CORRECT BUT ATTENTION FAILED",
    baseline_only
)

print("\nAnalysis complete.")