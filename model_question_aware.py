import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense
)


# ============================================================
# QUESTION-AWARE SEQ2SEQ LSTM
# ============================================================

def build_training_model(
    vocab_size,
    encoder_len,
    decoder_len,
    embedding_dim=64,
    latent_dim=128
):
    """
    Training model.

    Encoder:
        Context + Question -> Embedding -> LSTM

    Decoder:
        Decoder input -> Embedding -> LSTM -> Softmax
    """

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    encoder_inputs = Input(
        shape=(encoder_len,),
        name="encoder_input"
    )

    encoder_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
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
        input_dim=vocab_size,
        output_dim=embedding_dim,
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

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_layer = Dense(
        vocab_size,
        activation="softmax",
        name="output"
    )

    outputs = output_layer(decoder_outputs)

    # --------------------------------------------------------
    # Complete model
    # --------------------------------------------------------

    model = Model(
        [encoder_inputs, decoder_inputs],
        outputs,
        name="question_aware_seq2seq"
    )

    return model


# ============================================================
# INFERENCE MODELS
# ============================================================

def build_inference_models(
    vocab_size,
    encoder_len,
    embedding_dim=64,
    latent_dim=128
):
    """
    Builds separate encoder and decoder models
    for greedy inference.
    """

    # ========================================================
    # ENCODER
    # ========================================================

    encoder_inputs = Input(
        shape=(encoder_len,),
        name="encoder_input"
    )

    encoder_embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="encoder_embedding"
    )

    encoder_embedding = encoder_embedding_layer(
        encoder_inputs
    )

    encoder_lstm = LSTM(
        latent_dim,
        return_state=True,
        name="encoder_lstm"
    )

    _, state_h, state_c = encoder_lstm(
        encoder_embedding
    )

    encoder_model = Model(
        encoder_inputs,
        [state_h, state_c],
        name="question_aware_encoder_inference"
    )

    # ========================================================
    # DECODER
    # ========================================================

    decoder_inputs = Input(
        shape=(1,),
        name="decoder_token"
    )

    decoder_state_h = Input(
        shape=(latent_dim,),
        name="decoder_state_h"
    )

    decoder_state_c = Input(
        shape=(latent_dim,),
        name="decoder_state_c"
    )

    decoder_embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="decoder_embedding"
    )

    decoder_embedding = decoder_embedding_layer(
        decoder_inputs
    )

    decoder_lstm = LSTM(
        latent_dim,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )

    decoder_outputs, new_h, new_c = decoder_lstm(
        decoder_embedding,
        initial_state=[
            decoder_state_h,
            decoder_state_c
        ]
    )

    output_layer = Dense(
        vocab_size,
        activation="softmax",
        name="output"
    )

    decoder_outputs = output_layer(
        decoder_outputs
    )

    decoder_model = Model(
        [
            decoder_inputs,
            decoder_state_h,
            decoder_state_c
        ],
        [
            decoder_outputs,
            new_h,
            new_c
        ],
        name="question_aware_decoder_inference"
    )

    return encoder_model, decoder_model