"""
model.py
Seq2Seq LSTM (Encoder-Decoder) for bAbI QA Task 1.

Architecture:
    Encoder: Embedding -> LSTM (returns final hidden+cell state only)
    Decoder: Embedding -> LSTM (initialized with encoder states,
             returns full sequence) -> Dense(vocab_size, softmax)

Two builders are provided:
    - build_training_model(...)  : the model you train with teacher forcing
    - build_inference_models(...): encoder_model + decoder_model used to
                                    generate answers token-by-token at
                                    prediction time (can't use teacher
                                    forcing when you don't know the answer)

Both builders share the SAME layer objects, so weights trained via
build_training_model are automatically used by build_inference_models
(as long as you build inference models AFTER loading/training the
training model's weights, from the same layer instances).
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense


def build_training_model(vocab_size, max_encoder_len, max_decoder_len,
                          embedding_dim=64, latent_dim=128):
    """
    Builds the full trainable Seq2Seq model.

    Returns:
        model: Keras Model, inputs=[encoder_input, decoder_input],
               output=decoder_dense softmax predictions
        layers: dict of the shared layer objects (encoder_embedding,
                encoder_lstm, decoder_embedding, decoder_lstm,
                decoder_dense) so build_inference_models can reuse them.
    """

    # ---------------- Encoder ----------------
    encoder_inputs = Input(shape=(max_encoder_len,), name="encoder_input")
    encoder_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="encoder_embedding",
    )
    encoder_emb_out = encoder_embedding(encoder_inputs)

    encoder_lstm = LSTM(latent_dim, return_state=True, name="encoder_lstm")
    # We only need the final states; discard the per-timestep outputs.
    _, state_h, state_c = encoder_lstm(encoder_emb_out)
    encoder_states = [state_h, state_c]

    # ---------------- Decoder ----------------
    decoder_inputs = Input(shape=(max_decoder_len,), name="decoder_input")
    decoder_embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="decoder_embedding",
    )
    decoder_emb_out = decoder_embedding(decoder_inputs)

    decoder_lstm = LSTM(
        latent_dim, return_sequences=True, return_state=True, name="decoder_lstm"
    )
    decoder_seq_out, _, _ = decoder_lstm(decoder_emb_out, initial_state=encoder_states)

    decoder_dense = Dense(vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_seq_out)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name="seq2seq_babi")

    layers = {
        "encoder_inputs": encoder_inputs,
        "encoder_embedding": encoder_embedding,
        "encoder_lstm": encoder_lstm,
        "decoder_embedding": decoder_embedding,
        "decoder_lstm": decoder_lstm,
        "decoder_dense": decoder_dense,
    }
    return model, layers


def build_inference_models(layers, latent_dim=128):
    """
    Builds the two models used at prediction time:
        encoder_model: encoder_input -> [state_h, state_c]
        decoder_model: [decoder_single_token_input, state_h_in, state_c_in]
                        -> [token_probs, state_h_out, state_c_out]

    Must be called with the SAME `layers` dict returned by
    build_training_model, after that model's weights have been
    trained or loaded, so the inference graphs reuse trained weights.
    """
    encoder_inputs = layers["encoder_inputs"]
    encoder_embedding = layers["encoder_embedding"]
    encoder_lstm = layers["encoder_lstm"]
    decoder_embedding = layers["decoder_embedding"]
    decoder_lstm = layers["decoder_lstm"]
    decoder_dense = layers["decoder_dense"]

    # ---- Encoder inference model ----
    _, state_h, state_c = encoder_lstm(encoder_embedding(encoder_inputs))
    encoder_model = Model(encoder_inputs, [state_h, state_c], name="encoder_inference")

    # ---- Decoder inference model (single timestep, fed back in a loop) ----
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_state_h_input")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_state_c_input")
    decoder_single_input = Input(shape=(1,), name="decoder_single_token_input")

    dec_emb = decoder_embedding(decoder_single_input)
    dec_seq_out, dec_state_h, dec_state_c = decoder_lstm(
        dec_emb, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )
    dec_token_probs = decoder_dense(dec_seq_out)

    decoder_model = Model(
        [decoder_single_input, decoder_state_input_h, decoder_state_input_c],
        [dec_token_probs, dec_state_h, dec_state_c],
        name="decoder_inference",
    )

    return encoder_model, decoder_model