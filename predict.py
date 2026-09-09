"""
predict.py
Evaluation + detailed error analysis for Seq2Seq LSTM with
Additive Attention on bAbI QA Task 1.

UPDATED: now points at the bidirectional attention model
(embedding_dim=128, latent_dim=256) and saves per-example
predictions to results/predictions.csv for visualize_results.py.

Run from the project root:
    python predict.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from model import build_training_model, build_inference_models


# ============================================================
# CONFIGURATION
# ============================================================

SEQ_DIR = "data/processed/sequences"
VOCAB_DIR = "data/processed"
TEST_CSV = "data/processed/test.csv"

# IMPORTANT: bidirectional attention model (updated architecture)
MODEL_PATH = "models/seq2seq_attention_bidir_best.keras"

# Must match model.py / train.py for the new bidirectional model.
EMBEDDING_DIM = 128
LATENT_DIM = 256

BATCH_SIZE = 256

NUM_EXAMPLES_TO_SHOW = 15
NUM_WRONG_EXAMPLES_TO_SHOW = 10


# ============================================================
# LOAD VOCABULARY
# ============================================================

def load_vocab():

    with open(
        f"{VOCAB_DIR}/vocab.pkl",
        "rb"
    ) as f:
        vocab = pickle.load(f)

    tokenizer = vocab["tokenizer"]

    sos_token = vocab["sos_token"]
    eos_token = vocab["eos_token"]

    sos_id = tokenizer.word_index[sos_token]
    eos_id = tokenizer.word_index[eos_token]

    # Padding token ID
    pad_id = 0

    # ID -> word mapping
    index_word = dict(tokenizer.index_word)

    # Add padding representation
    index_word[pad_id] = ""

    return (
        tokenizer,
        index_word,
        sos_id,
        eos_id,
        pad_id
    )


# ============================================================
# CONVERT TOKEN IDS TO WORDS
# ============================================================

def ids_to_words(
    id_seq,
    index_word,
    eos_id,
    pad_id
):

    words = []

    for tok_id in id_seq:

        tok_id = int(tok_id)

        # Stop at EOS or padding
        if tok_id == eos_id or tok_id == pad_id:
            break

        words.append(
            index_word.get(
                tok_id,
                f"<unk:{tok_id}>"
            )
        )

    return words


# ============================================================
# GREEDY DECODING WITH ATTENTION
# ============================================================

def greedy_decode_batch(
    encoder_input,
    encoder_model,
    decoder_model,
    sos_id,
    max_decoder_len
):

    n = encoder_input.shape[0]

    # --------------------------------------------------------
    # Encode the input
    # --------------------------------------------------------
    #
    # Attention model encoder returns:
    #
    #   encoder_outputs
    #   state_h
    #   state_c
    #
    encoder_outputs, state_h, state_c = encoder_model.predict(
        encoder_input,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    # --------------------------------------------------------
    # Start decoder with <sos>
    # --------------------------------------------------------

    target_tok = np.full(
        (n, 1),
        sos_id,
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Store predicted token IDs
    # --------------------------------------------------------

    decoded_ids = np.zeros(
        (n, max_decoder_len),
        dtype=np.int32
    )

    # --------------------------------------------------------
    # Generate answer one token at a time
    # --------------------------------------------------------

    for t in range(max_decoder_len):

        output_probs, state_h, state_c = decoder_model.predict(
            [
                target_tok,
                state_h,
                state_c,
                encoder_outputs
            ],
            batch_size=BATCH_SIZE,
            verbose=0
        )

        # Select highest-probability token
        sampled_ids = np.argmax(
            output_probs[:, -1, :],
            axis=-1
        )

        decoded_ids[:, t] = sampled_ids

        # Feed prediction back into decoder
        target_tok = sampled_ids.reshape(
            -1,
            1
        )

    return decoded_ids


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # --------------------------------------------------------
    # Load vocabulary
    # --------------------------------------------------------

    (
        tokenizer,
        index_word,
        sos_id,
        eos_id,
        pad_id
    ) = load_vocab()

    # --------------------------------------------------------
    # Load encoded test data
    # --------------------------------------------------------

    test_encoder_input = np.load(
        f"{SEQ_DIR}/test_encoder_input.npy"
    )

    test_decoder_target = np.load(
        f"{SEQ_DIR}/test_decoder_target.npy"
    )

    # Convert (N, T, 1) -> (N, T)
    if test_decoder_target.ndim == 3:
        test_decoder_target = test_decoder_target.squeeze(-1)

    # --------------------------------------------------------
    # Load human-readable CSV
    # --------------------------------------------------------

    test_df = pd.read_csv(TEST_CSV)

    required_columns = [
        "context",
        "question",
        "answer"
    ]

    for column in required_columns:

        if column not in test_df.columns:

            raise ValueError(
                f"Missing required column '{column}' "
                f"in {TEST_CSV}"
            )

    # --------------------------------------------------------
    # Check number of samples
    # --------------------------------------------------------

    num_encoded_samples = test_encoder_input.shape[0]
    num_target_samples = test_decoder_target.shape[0]
    num_csv_samples = len(test_df)

    print(
        f"Encoded test samples : {num_encoded_samples}"
    )

    print(
        f"Target test samples  : {num_target_samples}"
    )

    print(
        f"CSV test samples     : {num_csv_samples}"
    )

    if not (
        num_encoded_samples
        == num_target_samples
        == num_csv_samples
    ):

        raise ValueError(
            "Number of rows in .npy files "
            "and CSV do not match!"
        )

    # --------------------------------------------------------
    # Sequence lengths
    # --------------------------------------------------------

    max_encoder_len = test_encoder_input.shape[1]
    max_decoder_len = test_decoder_target.shape[1]

    print(
        f"max_encoder_len      : {max_encoder_len}"
    )

    print(
        f"max_decoder_len      : {max_decoder_len}"
    )

    # ========================================================
    # VERIFY CSV / ENCODED TARGET ALIGNMENT
    # ========================================================

    print()
    print("=" * 60)
    print("VERIFYING CSV / .NPY ALIGNMENT")
    print("=" * 60)

    csv_answers = (
        test_df["answer"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    decoded_true_answers = []

    alignment_mismatches = []

    for i in range(num_target_samples):

        true_words = ids_to_words(
            test_decoder_target[i],
            index_word,
            eos_id,
            pad_id
        )

        decoded_answer = " ".join(true_words)

        decoded_true_answers.append(
            decoded_answer
        )

        if decoded_answer != csv_answers[i]:

            alignment_mismatches.append(
                (
                    i,
                    csv_answers[i],
                    decoded_answer
                )
            )

    if len(alignment_mismatches) == 0:

        print(
            "CSV/.npy answer alignment: OK"
        )

    else:

        print(
            f"WARNING: Found "
            f"{len(alignment_mismatches)} "
            f"alignment mismatches."
        )

        print()
        print("First mismatches:")

        for item in alignment_mismatches[:10]:

            i, csv_answer, decoded_answer = item

            print(
                f"Row {i}: "
                f"CSV='{csv_answer}' | "
                f".npy='{decoded_answer}'"
            )

        raise ValueError(
            "CSV and .npy answer data are not aligned. "
            "Stop before evaluating predictions."
        )

    # ========================================================
    # REBUILD ATTENTION MODEL (BIDIRECTIONAL)
    # ========================================================

    print()
    print("=" * 60)
    print("REBUILDING BIDIRECTIONAL ATTENTION MODEL")
    print("=" * 60)

    vocab_size = len(
        tokenizer.word_index
    ) + 1

    print(
        f"Vocabulary size: {vocab_size}"
    )

    print(
        f"Encoder length : {max_encoder_len}"
    )

    print(
        f"Decoder length : {max_decoder_len}"
    )

    print(
        f"embedding_dim  : {EMBEDDING_DIM}"
    )

    print(
        f"latent_dim     : {LATENT_DIM}"
    )

    training_model, layers = build_training_model(
        vocab_size=vocab_size,
        max_encoder_len=max_encoder_len,
        max_decoder_len=max_decoder_len,
        embedding_dim=EMBEDDING_DIM,
        latent_dim=LATENT_DIM
    )

    # --------------------------------------------------------
    # Load trained attention weights
    # --------------------------------------------------------

    print()
    print(
        "Loading attention model weights from:"
    )

    print(
        MODEL_PATH
    )

    training_model.load_weights(
        MODEL_PATH
    )

    print(
        "Weights loaded successfully."
    )

    # --------------------------------------------------------
    # Build attention inference models
    # --------------------------------------------------------

    latent_dim = LATENT_DIM

    encoder_model, decoder_model = build_inference_models(
        layers,
        latent_dim=latent_dim
    )

    print(
        "Attention inference models "
        "created successfully."
    )

    # ========================================================
    # RUN PREDICTIONS
    # ========================================================

    print()
    print("=" * 60)
    print("RUNNING PREDICTIONS")
    print("=" * 60)

    print(
        f"Running greedy decoding on "
        f"{num_encoded_samples} test examples..."
    )

    predicted_ids = greedy_decode_batch(
        test_encoder_input,
        encoder_model,
        decoder_model,
        sos_id,
        max_decoder_len
    )

    print(
        "Prediction complete."
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    print()
    print("=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    correct = 0

    actual_answers = []
    predicted_answers = []

    sample_examples = []
    wrong_examples = []

    # --------------------------------------------------------
    # Process every test example
    # --------------------------------------------------------

    for i in range(num_encoded_samples):

        # Ground truth
        true_words = ids_to_words(
            test_decoder_target[i],
            index_word,
            eos_id,
            pad_id
        )

        # Prediction
        pred_words = ids_to_words(
            predicted_ids[i],
            index_word,
            eos_id,
            pad_id
        )

        true_answer = " ".join(
            true_words
        )

        predicted_answer = " ".join(
            pred_words
        )

        actual_answers.append(
            true_answer
        )

        predicted_answers.append(
            predicted_answer
        )

        # Exact match
        is_match = (
            true_words == pred_words
        )

        if is_match:
            correct += 1

        # ----------------------------------------------------
        # Save sample predictions
        # ----------------------------------------------------

        if len(sample_examples) < NUM_EXAMPLES_TO_SHOW:

            sample_examples.append(
                (
                    i,
                    true_answer,
                    predicted_answer,
                    is_match
                )
            )

        # ----------------------------------------------------
        # Save wrong examples
        # ----------------------------------------------------

        if not is_match:

            row = test_df.iloc[i]

            wrong_examples.append(
                {
                    "index": i,
                    "context": str(row["context"]),
                    "question": str(row["question"]),
                    "ground_truth": true_answer,
                    "prediction": predicted_answer
                }
            )

    # ========================================================
    # SAVE PER-EXAMPLE RESULTS (for visualize_results.py)
    # ========================================================

    results_df = pd.DataFrame({
        "index": range(num_encoded_samples),
        "context": test_df["context"].astype(str).tolist(),
        "question": test_df["question"].astype(str).tolist(),
        "ground_truth": actual_answers,
        "prediction": predicted_answers,
        "correct": [
            a == p for a, p in zip(actual_answers, predicted_answers)
        ],
    })

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/predictions.csv", index=False)

    print()
    print(f"Saved per-example predictions to results/predictions.csv")

    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    accuracy = (
        correct / num_encoded_samples
    )

    print()
    print("=" * 60)
    print("SAMPLE PREDICTIONS")
    print("=" * 60)

    for (
        index,
        true_answer,
        predicted_answer,
        is_match
    ) in sample_examples:

        if is_match:
            marker = "correct"
        else:
            marker = "WRONG"

        print(
            f"[{marker:<7}] "
            f"'{true_answer}' -> "
            f"'{predicted_answer}'"
        )

    print()
    print("=" * 60)
    print("OVERALL ACCURACY")
    print("=" * 60)

    print(
        f"Exact-match accuracy: "
        f"{correct}/{num_encoded_samples} "
        f"= {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    # ========================================================
    # GROUND-TRUTH ANSWER DISTRIBUTION
    # ========================================================

    print()
    print("=" * 60)
    print("GROUND-TRUTH ANSWER DISTRIBUTION")
    print("=" * 60)

    answer_counter = Counter(
        actual_answers
    )

    for answer, count in sorted(
        answer_counter.items()
    ):

        print(
            f"{answer:<12}: {count}"
        )

    # ========================================================
    # PER-ANSWER ACCURACY
    # ========================================================

    print()
    print("=" * 60)
    print("PER-ANSWER ACCURACY")
    print("=" * 60)

    per_class_total = Counter()
    per_class_correct = Counter()

    for actual, predicted in zip(
        actual_answers,
        predicted_answers
    ):

        per_class_total[actual] += 1

        if actual == predicted:
            per_class_correct[actual] += 1

    for answer in sorted(
        per_class_total
    ):

        total = per_class_total[answer]
        class_correct = per_class_correct[answer]

        class_accuracy = (
            class_correct / total
        )

        print(
            f"{answer:<12}: "
            f"{class_correct}/{total} "
            f"= {class_accuracy:.4f} "
            f"({class_accuracy * 100:.2f}%)"
        )

    # ========================================================
    # CONFUSION COUNTS
    # ========================================================

    print()
    print("=" * 60)
    print("WRONG PREDICTION / CONFUSION COUNTS")
    print("=" * 60)

    confusion = Counter()

    for actual, predicted in zip(
        actual_answers,
        predicted_answers
    ):

        if actual != predicted:

            confusion[
                (actual, predicted)
            ] += 1

    for (
        (actual, predicted),
        count
    ) in confusion.most_common():

        print(
            f"{actual:<12} -> "
            f"{predicted:<12}: "
            f"{count}"
        )

    # ========================================================
    # MODEL PREDICTION DISTRIBUTION
    # ========================================================

    print()
    print("=" * 60)
    print("MODEL PREDICTION DISTRIBUTION")
    print("=" * 60)

    prediction_counter = Counter(
        predicted_answers
    )

    for answer, count in sorted(
        prediction_counter.items()
    ):

        print(
            f"{answer:<12}: {count}"
        )

    # ========================================================
    # DETAILED WRONG EXAMPLES
    # ========================================================

    print()
    print("=" * 60)
    print("DETAILED WRONG PREDICTIONS")
    print("=" * 60)

    print(
        f"Showing first "
        f"{min(NUM_WRONG_EXAMPLES_TO_SHOW, len(wrong_examples))} "
        f"wrong examples."
    )

    print()

    for example in wrong_examples[
        :NUM_WRONG_EXAMPLES_TO_SHOW
    ]:

        print("-" * 60)

        print(
            f"TEST INDEX    : "
            f"{example['index']}"
        )

        print(
            f"CONTEXT       : "
            f"{example['context']}"
        )

        print(
            f"QUESTION      : "
            f"{example['question']}"
        )

        print(
            f"GROUND TRUTH  : "
            f"{example['ground_truth']}"
        )

        print(
            f"PREDICTION    : "
            f"{example['prediction']}"
        )

    print("-" * 60)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)

    print(
        f"Test samples     : "
        f"{num_encoded_samples}"
    )

    print(
        f"Correct           : "
        f"{correct}"
    )

    print(
        f"Incorrect         : "
        f"{num_encoded_samples - correct}"
    )

    print(
        f"Exact-match       : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "Alignment check   : PASSED"
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()