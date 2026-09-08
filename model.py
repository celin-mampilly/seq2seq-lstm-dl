"""
model.py
Seq2Seq LSTM with Additive Attention for bAbI QA Task 1.

Architecture:

    Encoder:
        Embedding -> LSTM
        |
        +--> all encoder outputs (for attention)
        |
        +--> final hidden state + cell state

    Decoder:
        Embedding -> LSTM
                 |
                 v
            Additive Attention
                 |
                 v
        Concatenate decoder output
        + attention context
                 |
                 v
        Dense(vocab_size, softmax)

Two builders are provided:

    - build_training_model(...)
        Model used during training with teacher forcing.

    - build_inference_models(...)
        Encoder + decoder models used for token-by-token
        greedy decoding during prediction.

The SAME layer objects are shared between the training
and inference models so that trained weights are reused.
"""

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
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
    embedding_dim=64,
    latent_dim=128,
):
    """
    Builds the Seq2Seq LSTM model with Additive Attention.

    Inputs:
        encoder_input:
            Padded context + question sequence.

        decoder_input:
            Decoder sequence shifted right, beginning with <sos>.

    Output:
        Softmax probability distribution over the vocabulary
        for every decoder timestep.

    Returns:
        model:
            Keras training model.

        layers:
            Dictionary containing shared layer objects used
            later to construct inference models.
    """

    # ========================================================
    # ENCODER
    # ========================================================

    encoder_inputs = Input(
        shape=(max_encoder_len,),
        name="encoder_input",
    )

    # --------------------------------------------------------
    # Encoder Embedding
    # --------------------------------------------------------

    encoder_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="encoder_embedding",
    )

    encoder_emb_out = encoder_embedding(
        encoder_inputs
    )

    # --------------------------------------------------------
    # Encoder LSTM
    #
    # IMPORTANT:
    # return_sequences=True
    #
    # We need every hidden state for attention.
    # --------------------------------------------------------

    encoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm",
    )

    encoder_outputs, state_h, state_c = encoder_lstm(
        encoder_emb_out
    )

    encoder_states = [
        state_h,
        state_c,
    ]

    # ========================================================
    # DECODER
    # ========================================================

    decoder_inputs = Input(
        shape=(max_decoder_len,),
        name="decoder_input",
    )

    # --------------------------------------------------------
    # Decoder Embedding
    # --------------------------------------------------------

    decoder_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="decoder_embedding",
    )

    decoder_emb_out = decoder_embedding(
        decoder_inputs
    )

    # --------------------------------------------------------
    # Decoder LSTM
    # --------------------------------------------------------

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm",
    )

    decoder_seq_out, _, _ = decoder_lstm(
        decoder_emb_out,
        initial_state=encoder_states,
    )

    # ========================================================
    # ADDITIVE ATTENTION
    # ========================================================

    attention = AdditiveAttention(
        name="attention",
    )

    # --------------------------------------------------------
    # Attention:
    #
    # Query:
    #   decoder hidden states
    #
    # Key/Value:
    #   encoder hidden states
    #
    # Shape:
    #
    # decoder_seq_out:
    #     (batch, decoder_length, latent_dim)
    #
    # encoder_outputs:
    #     (batch, encoder_length, latent_dim)
    #
    # attention_output:
    #     (batch, decoder_length, latent_dim)
    # --------------------------------------------------------

    attention_output = attention(
        [
            decoder_seq_out,
            encoder_outputs,
        ]
    )

    # ========================================================
    # COMBINE DECODER + ATTENTION
    # ========================================================

    combined_output = Concatenate(
        axis=-1,
        name="decoder_attention_concat",
    )(
        [
            decoder_seq_out,
            attention_output,
        ]
    )

    # Combined dimension:
    #
    # decoder output = 128
    # attention output = 128
    #
    # total = 256
    #
    # The Dense layer maps this back to vocabulary size.

    decoder_dense = Dense(
        vocab_size,
        activation="softmax",
        name="decoder_dense",
    )

    decoder_outputs = decoder_dense(
        combined_output
    )

    # ========================================================
    # COMPLETE MODEL
    # ========================================================

    model = Model(
        [
            encoder_inputs,
            decoder_inputs,
        ],
        decoder_outputs,
        name="seq2seq_babi_attention",
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # ========================================================
    # SHARED LAYERS
    # ========================================================

    layers = {
        "encoder_inputs": encoder_inputs,

        "encoder_embedding": encoder_embedding,
        "encoder_lstm": encoder_lstm,

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
    latent_dim=128,
):
    """
    Builds the encoder and decoder models used during
    inference / prediction.

    Encoder:

        encoder_input
            |
            v
        encoder outputs
        + state_h
        + state_c

    Decoder:

        decoder token
        + previous state_h
        + previous state_c
        + encoder outputs
            |
            v
        decoder LSTM
            |
            v
        attention
            |
            v
        Dense
            |
            v
        next token probabilities
    """

    # ========================================================
    # GET SHARED LAYERS
    # ========================================================

    encoder_inputs = layers[
        "encoder_inputs"
    ]

    encoder_embedding = layers[
        "encoder_embedding"
    ]

    encoder_lstm = layers[
        "encoder_lstm"
    ]

    decoder_embedding = layers[
        "decoder_embedding"
    ]

    decoder_lstm = layers[
        "decoder_lstm"
    ]

    attention = layers[
        "attention"
    ]

    decoder_dense = layers[
        "decoder_dense"
    ]

    # ========================================================
    # ENCODER INFERENCE MODEL
    # ========================================================

    encoder_emb_out = encoder_embedding(
        encoder_inputs
    )

    encoder_outputs, state_h, state_c = encoder_lstm(
        encoder_emb_out
    )

    encoder_model = Model(
        encoder_inputs,
        [
            encoder_outputs,
            state_h,
            state_c,
        ],
        name="encoder_inference",
    )

    # ========================================================
    # DECODER INFERENCE INPUTS
    # ========================================================

    # One token at a time
    decoder_single_input = Input(
        shape=(1,),
        name="decoder_single_token_input",
    )

    # Previous decoder hidden state
    decoder_state_input_h = Input(
        shape=(latent_dim,),
        name="decoder_state_h_input",
    )

    # Previous decoder cell state
    decoder_state_input_c = Input(
        shape=(latent_dim,),
        name="decoder_state_c_input",
    )

    # --------------------------------------------------------
    # Encoder outputs are required by attention.
    #
    # During inference, the encoder produces:
    #
    # (batch, encoder_length, latent_dim)
    # --------------------------------------------------------

    encoder_outputs_input = Input(
        shape=(None, latent_dim),
        name="encoder_outputs_input",
    )

    # ========================================================
    # DECODER EMBEDDING
    # ========================================================

    decoder_emb = decoder_embedding(
        decoder_single_input
    )

    # ========================================================
    # ONE DECODER STEP
    # ========================================================

    decoder_seq_out, decoder_state_h, decoder_state_c = decoder_lstm(
        decoder_emb,
        initial_state=[
            decoder_state_input_h,
            decoder_state_input_c,
        ],
    )

    # ========================================================
    # ATTENTION
    # ========================================================

    attention_output = attention(
        [
            decoder_seq_out,
            encoder_outputs_input,
        ]
    )

    # ========================================================
    # COMBINE DECODER + ATTENTION
    # ========================================================

    combined_output = Concatenate(
        axis=-1,
        name="inference_attention_concat",
    )(
        [
            decoder_seq_out,
            attention_output,
        ]
    )

    # ========================================================
    # PREDICT NEXT TOKEN
    # ========================================================

    decoder_token_probs = decoder_dense(
        combined_output
    )

    # ========================================================
    # COMPLETE DECODER MODEL
    # ========================================================

    decoder_model = Model(
        [
            decoder_single_input,
            decoder_state_input_h,
            decoder_state_input_c,
            encoder_outputs_input,
        ],
        [
            decoder_token_probs,
            decoder_state_h,
            decoder_state_c,
        ],
        name="decoder_inference",
    )

    return encoder_model, decoder_model