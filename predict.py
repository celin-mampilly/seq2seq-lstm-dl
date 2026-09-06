"""
predict.py
Greedy decoding + true exact-match evaluation for the Seq2Seq LSTM bAbI QA model.

Unlike the teacher-forced training accuracy (which is inflated by the
padding tokens baked into the fixed-length decoder target), this script:
    1. Encodes each test context+question ONCE.
    2. Runs the decoder autoregressively, one token at a time, starting
       from <sos>, feeding each predicted token back in as the next
       input, until <eos> is produced or max_decoder_len is reached.
    3. Converts predicted and ground-truth token ids back to words.
    4. Computes exact-match accuracy on the decoded answer text.

Run from the project root:
    python predict.py
"""

import pickle

import numpy as np
from tensorflow.keras.models import load_model

from model import build_inference_models

DATA_DIR = "data/processed/sequences"
MODEL_PATH = "models/seq2seq_best.keras"
VOCAB_PATH = f"{DATA_DIR}/vocab.pkl"

BATCH_SIZE = 256
NUM_EXAMPLES_TO_SHOW = 15


def load_vocab():
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    tokenizer = vocab["tokenizer"]
    sos_id = tokenizer.word_index[vocab["sos_token"]]
    eos_id = tokenizer.word_index[vocab["eos_token"]]
    pad_id = 0  # Keras Tokenizer reserves 0 for padding; not in word_index

    index_word = dict(tokenizer.index_word)
    index_word[pad_id] = ""  # so pad tokens render as nothing, not KeyError

    return tokenizer, index_word, sos_id, eos_id, pad_id


def ids_to_words(id_seq, index_word, eos_id, pad_id):
    """Stop at the first <eos> (or pad), drop special tokens, return word list."""
    words = []
    for tok_id in id_seq:
        tok_id = int(tok_id)
        if tok_id == eos_id or tok_id == pad_id:
            break
        words.append(index_word.get(tok_id, f"<unk:{tok_id}>"))
    return words


def greedy_decode_batch(encoder_input, encoder_model, decoder_model,
                         sos_id, max_decoder_len):
    """
    Vectorized greedy decoding for a whole batch at once.
    Returns an array of shape (N, max_decoder_len) of predicted token ids
    (may run past the true <eos> for a given row; ids_to_words truncates).
    """
    n = encoder_input.shape[0]
    state_h, state_c = encoder_model.predict(encoder_input, batch_size=BATCH_SIZE, verbose=0)

    target_tok = np.full((n, 1), sos_id, dtype=np.int32)
    decoded_ids = np.zeros((n, max_decoder_len), dtype=np.int32)

    for t in range(max_decoder_len):
        output_probs, state_h, state_c = decoder_model.predict(
            [target_tok, state_h, state_c], batch_size=BATCH_SIZE, verbose=0
        )
        sampled_ids = np.argmax(output_probs[:, -1, :], axis=-1)  # (N,)
        decoded_ids[:, t] = sampled_ids
        target_tok = sampled_ids.reshape(-1, 1)

    return decoded_ids


def main():
    tokenizer, index_word, sos_id, eos_id, pad_id = load_vocab()

    test_encoder_input = np.load(f"{DATA_DIR}/test_encoder_input.npy")
    test_decoder_target = np.load(f"{DATA_DIR}/test_decoder_target.npy")
    if test_decoder_target.ndim == 3:
        test_decoder_target = test_decoder_target.squeeze(-1)

    # Derive lengths from the actual arrays, not vocab.pkl (known to be stale).
    max_encoder_len = test_encoder_input.shape[1]
    max_decoder_len = test_decoder_target.shape[1]
    print(f"max_encoder_len={max_encoder_len}, max_decoder_len={max_decoder_len}")

    print(f"Loading trained model from {MODEL_PATH} ...")
    trained_model = load_model(MODEL_PATH)

    latent_dim = trained_model.get_layer("encoder_lstm").units
    layers = {
        "encoder_inputs": trained_model.get_layer("encoder_input").output,
        "encoder_embedding": trained_model.get_layer("encoder_embedding"),
        "encoder_lstm": trained_model.get_layer("encoder_lstm"),
        "decoder_embedding": trained_model.get_layer("decoder_embedding"),
        "decoder_lstm": trained_model.get_layer("decoder_lstm"),
        "decoder_dense": trained_model.get_layer("decoder_dense"),
    }
    encoder_model, decoder_model = build_inference_models(layers, latent_dim=latent_dim)

    print(f"Running greedy decoding on {test_encoder_input.shape[0]} test examples ...")
    predicted_ids = greedy_decode_batch(
        test_encoder_input, encoder_model, decoder_model, sos_id, max_decoder_len
    )

    n = test_encoder_input.shape[0]
    correct = 0
    examples = []
    for i in range(n):
        pred_words = ids_to_words(predicted_ids[i], index_word, eos_id, pad_id)
        true_words = ids_to_words(test_decoder_target[i], index_word, eos_id, pad_id)

        is_match = pred_words == true_words
        correct += int(is_match)

        if i < NUM_EXAMPLES_TO_SHOW:
            examples.append((true_words, pred_words, is_match))

    accuracy = correct / n

    print("\nSample predictions (ground truth -> prediction):")
    for true_words, pred_words, is_match in examples:
        marker = "correct" if is_match else "WRONG"
        print(f"  [{marker}] {' '.join(true_words)!r} -> {' '.join(pred_words)!r}")

    print(f"\nExact-match accuracy on test set: {correct}/{n} = {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()