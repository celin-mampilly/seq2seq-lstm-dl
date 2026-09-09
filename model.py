"""
model.py
Seq2Seq LSTM with Additive Attention for bAbI QA Task 1.
UPDATED VERSION: bidirectional encoder + dropout + larger capacity.

Architecture:

    Encoder:
        Embedding -> Bidirectional(LSTM)
        |
        +--> all encoder outputs, PROJECTED to latent_dim (for attention)
        |
        +--> final hidden/cell state, PROJECTED to latent_dim
        |    (forward + backward concatenated, then Dense-projected down)

    Decoder:
        Embedding -> LSTM (unidirectional, initialized from encoder state)
                 |
                 v
            Additive Attention (query=decoder, key/value=encoder_outputs)
                 |
                 v
        Concatenate decoder output + attention context
                 |
                 v
        Dense(vocab_size, softmax)

IMPORTANT: This is a different graph than the previous unidirectional
version, so old .keras weight files are NOT compatible. You must
retrain from scratch after swapping this in (train.py doesn't need
to change, since it just calls build_training_model(...) and fits).

Two builders are provided, same names/signatures as before so
predict.py and train.py don't need interface changes:

    - build_training_model(...)
    - build_inference_models(...)
"""

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    AdditiveAttention,
    Concatenate,
)


# ============================================================
# TRAINING MODEL
# ============================================================

def build_training_model(
    vocab_size,
    max_encoder_len,
    max_decoder_len,
    embedding_dim=128,      # increased from 64
    latent_dim=256,         # increased from 128
    dropout=0.2,
    recurrent_dropout=0.0,  # keep 0.0 unless you have GPU; CPU recurrent_dropout is very slow
):
    """
    Builds the Seq2Seq LSTM model with Additive Attention,
    bidirectional encoder, and dropout regularization.

    Returns:
        model:  Keras training model.
        layers: dict of shared layer objects for build_inference_models.
    """

    # ========================================================
    # ENCODER
    # ========================================================

    encoder_inputs = Input(
        shape=(max_encoder_len,),
        name="encoder_input",
    )

    encoder_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="encoder_embedding",
    )

    encoder_emb_out = encoder_embedding(encoder_inputs)

    # --------------------------------------------------------
    # Bidirectional Encoder LSTM
    #
    # Reading the context in both directions means the
    # representation for an early sentence isn't only
    # informed by what came before it, but also by what
    # comes after - useful since the "relevant" sentence
    # for a question can appear anywhere in a long context.
    # --------------------------------------------------------

    encoder_bilstm = Bidirectional(
        LSTM(
            latent_dim,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            name="encoder_lstm",
        ),
        name="encoder_bidirectional",
    )

    (
        encoder_outputs_raw,   # (batch, T_enc, 2*latent_dim)
        forward_h, forward_c,
        backward_h, backward_c,
    ) = encoder_bilstm(encoder_emb_out)

    # --------------------------------------------------------
    # Merge forward/backward states, then project back down
    # to latent_dim so the (unidirectional) decoder LSTM can
    # be initialized with them directly.
    # --------------------------------------------------------

    state_h_concat = Concatenate(name="state_h_concat")(
        [forward_h, backward_h]
    )
    state_c_concat = Concatenate(name="state_c_concat")(
        [forward_c, backward_c]
    )

    state_h_dense = Dense(latent_dim, activation="tanh", name="state_h_proj")
    state_c_dense = Dense(latent_dim, activation="tanh", name="state_c_proj")

    state_h = state_h_dense(state_h_concat)
    state_c = state_c_dense(state_c_concat)

    encoder_states = [state_h, state_c]

    # --------------------------------------------------------
    # Also project encoder_outputs (2*latent_dim) down to
    # latent_dim, so attention's key/value dimension matches
    # the decoder's query dimension.
    # --------------------------------------------------------

    encoder_outputs_proj = Dense(
        latent_dim, activation="tanh", name="encoder_outputs_proj"
    )
    encoder_outputs = encoder_outputs_proj(encoder_outputs_raw)

    # ========================================================
    # DECODER
    # ========================================================

    decoder_inputs = Input(
        shape=(max_decoder_len,),
        name="decoder_input",
    )

    decoder_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="decoder_embedding",
    )

    decoder_emb_out = decoder_embedding(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        name="decoder_lstm",
    )

    decoder_seq_out, _, _ = decoder_lstm(
        decoder_emb_out,
        initial_state=encoder_states,
    )

    # ========================================================
    # ADDITIVE ATTENTION
    # ========================================================

    attention = AdditiveAttention(name="attention")

    attention_output = attention(
        [decoder_seq_out, encoder_outputs]
    )

    # ========================================================
    # COMBINE DECODER + ATTENTION
    # ========================================================

    combined_output = Concatenate(
        axis=-1, name="decoder_attention_concat"
    )([decoder_seq_out, attention_output])

    decoder_dense = Dense(
        vocab_size, activation="softmax", name="decoder_dense"
    )

    decoder_outputs = decoder_dense(combined_output)

    # ========================================================
    # COMPLETE MODEL
    # ========================================================

    model = Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs,
        name="seq2seq_babi_attention_bidir",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    layers = {
        "encoder_inputs": encoder_inputs,
        "encoder_embedding": encoder_embedding,
        "encoder_bilstm": encoder_bilstm,
        "state_h_dense": state_h_dense,
        "state_c_dense": state_c_dense,
        "encoder_outputs_proj": encoder_outputs_proj,
        "decoder_embedding": decoder_embedding,
        "decoder_lstm": decoder_lstm,
        "attention": attention,
        "decoder_dense": decoder_dense,
    }

    return model, layers


# ============================================================
# INFERENCE MODELS
# ============================================================

def build_inference_models(
    layers,
    latent_dim=256,   # must match the value used in build_training_model
):
    """
    Builds the encoder and decoder models used during inference.
    Same shapes as before from predict.py's point of view:

        encoder_model:  encoder_input -> [encoder_outputs, state_h, state_c]
        decoder_model:  [token, state_h, state_c, encoder_outputs]
                            -> [token_probs, state_h, state_c]

    predict.py does NOT need to change - it calls this the same way.
    """

    encoder_inputs = layers["encoder_inputs"]
    encoder_embedding = layers["encoder_embedding"]
    encoder_bilstm = layers["encoder_bilstm"]
    state_h_dense = layers["state_h_dense"]
    state_c_dense = layers["state_c_dense"]
    encoder_outputs_proj = layers["encoder_outputs_proj"]

    decoder_embedding = layers["decoder_embedding"]
    decoder_lstm = layers["decoder_lstm"]
    attention = layers["attention"]
    decoder_dense = layers["decoder_dense"]

    # ========================================================
    # ENCODER INFERENCE MODEL
    # ========================================================

    encoder_emb_out = encoder_embedding(encoder_inputs)

    (
        encoder_outputs_raw,
        forward_h, forward_c,
        backward_h, backward_c,
    ) = encoder_bilstm(encoder_emb_out)

    state_h_concat = Concatenate()([forward_h, backward_h])
    state_c_concat = Concatenate()([forward_c, backward_c])

    state_h = state_h_dense(state_h_concat)
    state_c = state_c_dense(state_c_concat)

    encoder_outputs = encoder_outputs_proj(encoder_outputs_raw)

    encoder_model = Model(
        encoder_inputs,
        [encoder_outputs, state_h, state_c],
        name="encoder_inference",
    )

    # ========================================================
    # DECODER INFERENCE MODEL
    # ========================================================

    decoder_single_input = Input(shape=(1,), name="decoder_single_token_input")
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_h_input")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_c_input")

    # NOTE: shape is (None, latent_dim) since encoder_outputs is
    # already projected down to latent_dim above.
    encoder_outputs_input = Input(
        shape=(None, latent_dim), name="encoder_outputs_input"
    )

    decoder_emb = decoder_embedding(decoder_single_input)

    decoder_seq_out, decoder_state_h, decoder_state_c = decoder_lstm(
        decoder_emb,
        initial_state=[decoder_state_input_h, decoder_state_input_c],
    )

    attention_output = attention(
        [decoder_seq_out, encoder_outputs_input]
    )

    combined_output = Concatenate(
        axis=-1, name="inference_attention_concat"
    )([decoder_seq_out, attention_output])

    decoder_token_probs = decoder_dense(combined_output)

    decoder_model = Model(
        [
            decoder_single_input,
            decoder_state_input_h,
            decoder_state_input_c,
            encoder_outputs_input,
        ],
        [decoder_token_probs, decoder_state_h, decoder_state_c],
        name="decoder_inference",
    )

    return encoder_model, decoder_model