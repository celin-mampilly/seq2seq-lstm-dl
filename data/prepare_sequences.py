"""
data/prepare_sequences.py

Stage 2 of the Seq2Seq LSTM QA project:

    CSV Dataset
        ↓
    Tokenization
        ↓
    Vocabulary
        ↓
    Integer sequences
        ↓
    Padding
        ↓
    Encoder/Decoder input  (saved as .npy + vocab as .pkl)

Reads:  data/processed/train.csv, validation.csv, test.csv
Writes: data/processed/sequences/*.npy
        data/processed/vocab.pkl

Framework assumption: TensorFlow / Keras (tf.keras.preprocessing.text.Tokenizer
+ pad_sequences). If you're actually using PyTorch, say so and I'll rewrite
this with a manual vocab + torch tensors instead.
"""

import os
import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
DATA_DIR = "data/processed"
OUT_DIR = os.path.join(DATA_DIR, "sequences")
os.makedirs(OUT_DIR, exist_ok=True)

# Special tokens. We prepend these manually so we control their ids exactly.
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"   # start-of-sequence, needed to kick off the decoder
EOS_TOKEN = "<eos>"   # end-of-sequence, tells the decoder when to stop
OOV_TOKEN = "<unk>"   # out-of-vocabulary / unknown word

# bAbI answers are usually single words (e.g. "kitchen"), contexts/questions
# are short, so we don't need a huge max length. Tune these after checking
# the printed length stats below.
MAX_ENCODER_LEN = 60   # covers context + question combined
MAX_DECODER_LEN = 6    # answer length + <sos>/<eos>


# ---------------------------------------------------------------------------
# 1. Load the CSVs produced in Stage 1
# ---------------------------------------------------------------------------
def load_split(name):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df = pd.read_csv(path)
    df = df.dropna(subset=["context", "question", "answer"])
    return df


train_df = load_split("train")
val_df = load_split("validation")
test_df = load_split("test")

print(f"Loaded train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")


# ---------------------------------------------------------------------------
# 2. Build encoder input text = context + " " + question
#    Decoder target text = <sos> + answer + <eos>
# ---------------------------------------------------------------------------
def build_encoder_text(df):
    return (df["context"].astype(str) + " " + df["question"].astype(str)).tolist()


def build_decoder_text(df):
    # Keep <sos>/<eos> as separate space-delimited tokens so the tokenizer
    # treats them as single vocab entries, not broken apart.
    return (SOS_TOKEN + " " + df["answer"].astype(str) + " " + EOS_TOKEN).tolist()


train_enc_text = build_encoder_text(train_df)
train_dec_text = build_decoder_text(train_df)

val_enc_text = build_encoder_text(val_df)
val_dec_text = build_decoder_text(val_df)

test_enc_text = build_encoder_text(test_df)
test_dec_text = build_decoder_text(test_df)


# ---------------------------------------------------------------------------
# 3. Tokenization + Vocabulary
#    IMPORTANT: fit the tokenizer ONLY on the training set. Validation/test
#    must reuse the training vocabulary — otherwise you leak information and
#    the model will see word ids at eval time it never learned during
#    training (this is a very common student-project bug).
# ---------------------------------------------------------------------------
# One shared vocabulary for encoder + decoder keeps things simple for a
# small dataset like bAbI (answers are just words that also appear in
# contexts, so a shared vocab is natural here).
tokenizer = Tokenizer(
    filters="",       # bAbI text is already clean lowercase tokens; don't strip punctuation ourselves
    lower=True,
    oov_token=OOV_TOKEN,
)

# Fit on encoder + decoder training text together so both sides share ids.
tokenizer.fit_on_texts(train_enc_text + train_dec_text)

# Keras reserves index 0 implicitly for padding already, and oov_token gets
# index 1 automatically. We rely on that: pad_sequences() below pads with 0.
vocab_size = len(tokenizer.word_index) + 1  # +1 because Keras indices start at 1
print(f"Vocabulary size: {vocab_size}")
print(f"Sample word->id: { {k: tokenizer.word_index[k] for k in list(tokenizer.word_index)[:10]} }")


# ---------------------------------------------------------------------------
# 4. Convert text -> integer sequences
# ---------------------------------------------------------------------------
def to_sequences(texts):
    return tokenizer.texts_to_sequences(texts)


train_enc_seq = to_sequences(train_enc_text)
train_dec_seq = to_sequences(train_dec_text)
val_enc_seq = to_sequences(val_enc_text)
val_dec_seq = to_sequences(val_dec_text)
test_enc_seq = to_sequences(test_enc_text)
test_dec_seq = to_sequences(test_dec_text)

# --- sanity check: print length stats so you can justify MAX_*_LEN above ---
enc_lens = [len(s) for s in train_enc_seq]
dec_lens = [len(s) for s in train_dec_seq]
print(f"Encoder seq length: max={max(enc_lens)}, mean={np.mean(enc_lens):.1f}")
print(f"Decoder seq length: max={max(dec_lens)}, mean={np.mean(dec_lens):.1f}")


# ---------------------------------------------------------------------------
# 5. Padding
# ---------------------------------------------------------------------------
def pad(seqs, maxlen):
    return pad_sequences(seqs, maxlen=maxlen, padding="post", truncating="post")


train_enc_pad = pad(train_enc_seq, MAX_ENCODER_LEN)
train_dec_pad = pad(train_dec_seq, MAX_DECODER_LEN)
val_enc_pad = pad(val_enc_seq, MAX_ENCODER_LEN)
val_dec_pad = pad(val_dec_seq, MAX_DECODER_LEN)
test_enc_pad = pad(test_enc_seq, MAX_ENCODER_LEN)
test_dec_pad = pad(test_dec_seq, MAX_DECODER_LEN)


# ---------------------------------------------------------------------------
# 6. Build encoder/decoder INPUT/TARGET arrays for teacher forcing
#
#    decoder_input  = <sos> w1 w2 ... wn        (fed into decoder at each step)
#    decoder_target = w1 w2 ... wn <eos>        (what the decoder should predict)
#
#    i.e. decoder_target is decoder_input shifted left by one position.
#    This is the standard "teacher forcing" setup for seq2seq training.
# ---------------------------------------------------------------------------
def make_decoder_input_target(dec_pad):
    decoder_input = dec_pad[:, :-1]   # drop last token
    decoder_target = dec_pad[:, 1:]   # drop first token (<sos>)
    return decoder_input, decoder_target


train_dec_input, train_dec_target = make_decoder_input_target(train_dec_pad)
val_dec_input, val_dec_target = make_decoder_input_target(val_dec_pad)
test_dec_input, test_dec_target = make_decoder_input_target(test_dec_pad)

# decoder_target needs a trailing dim of 1 for sparse_categorical_crossentropy
train_dec_target = np.expand_dims(train_dec_target, -1)
val_dec_target = np.expand_dims(val_dec_target, -1)
test_dec_target = np.expand_dims(test_dec_target, -1)


# ---------------------------------------------------------------------------
# 7. Save everything for the training script
# ---------------------------------------------------------------------------
np.save(os.path.join(OUT_DIR, "train_encoder_input.npy"), train_enc_pad)
np.save(os.path.join(OUT_DIR, "train_decoder_input.npy"), train_dec_input)
np.save(os.path.join(OUT_DIR, "train_decoder_target.npy"), train_dec_target)

np.save(os.path.join(OUT_DIR, "val_encoder_input.npy"), val_enc_pad)
np.save(os.path.join(OUT_DIR, "val_decoder_input.npy"), val_dec_input)
np.save(os.path.join(OUT_DIR, "val_decoder_target.npy"), val_dec_target)

np.save(os.path.join(OUT_DIR, "test_encoder_input.npy"), test_enc_pad)
np.save(os.path.join(OUT_DIR, "test_decoder_input.npy"), test_dec_input)
np.save(os.path.join(OUT_DIR, "test_decoder_target.npy"), test_dec_target)

with open(os.path.join(DATA_DIR, "vocab.pkl"), "wb") as f:
    pickle.dump(
        {
            "tokenizer": tokenizer,
            "vocab_size": vocab_size,
            "max_encoder_len": MAX_ENCODER_LEN,
            "max_decoder_len": MAX_DECODER_LEN,
            "sos_token": SOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "pad_token": PAD_TOKEN,
            "oov_token": OOV_TOKEN,
        },
        f,
    )

print("\nSaved:")
print(f"  {OUT_DIR}/train_*.npy, val_*.npy, test_*.npy")
print(f"  {DATA_DIR}/vocab.pkl")
print("\nShapes:")
print("  train_encoder_input :", train_enc_pad.shape)
print("  train_decoder_input :", train_dec_input.shape)
print("  train_decoder_target:", train_dec_target.shape)