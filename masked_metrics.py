"""
masked_metrics.py

A custom accuracy metric that ignores padding (token id 0) when
scoring predictions - so training/validation accuracy reflects
real answer-word accuracy instead of being inflated by trivially
correct padding predictions.

Usage in model.py's model.compile(...) or train.py:

    from masked_metrics import masked_accuracy

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", masked_accuracy],   # keep both to compare
    )

The gap between "accuracy" and "masked_accuracy" during training
will show you exactly how much the default metric was inflated
by padding.
"""

import tensorflow as tf


def masked_accuracy(y_true, y_pred):
    """
    y_true: (batch, T)         integer token ids, 0 = padding
    y_pred: (batch, T, vocab)  softmax probabilities
    """
    y_true = tf.cast(y_true, tf.int64)
    pred_ids = tf.argmax(y_pred, axis=-1)

    matches = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

    # Avoid divide-by-zero if a batch is somehow all padding.
    return tf.reduce_sum(matches * mask) / (tf.reduce_sum(mask) + 1e-8)