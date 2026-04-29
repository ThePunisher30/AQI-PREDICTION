"""LSTM model definition — temporal-only baseline, no spatial modeling."""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm(input_shape, output_size, h1=64, h2=32, h3=None, dropout=0.3, lr=0.001):
    """Build and compile a Keras LSTM model.

    Args:
        input_shape: (timesteps, features) e.g. (24, 522)
        output_size: total output units (horizons * cities * targets)
        h1: first LSTM hidden size
        h2: second LSTM hidden size (None for single-layer)
        h3: third LSTM hidden size (None to skip)
        dropout: dropout rate
        lr: learning rate
    Returns:
        Compiled Keras model
    """
    layers = []
    if h3:
        layers += [LSTM(h1, return_sequences=True, input_shape=input_shape),
                   Dropout(dropout), LSTM(h2, return_sequences=True),
                   Dropout(dropout), LSTM(h3), Dropout(dropout)]
    elif h2:
        layers += [LSTM(h1, return_sequences=True, input_shape=input_shape),
                   Dropout(dropout), LSTM(h2), Dropout(dropout)]
    else:
        layers += [LSTM(h1, input_shape=input_shape), Dropout(dropout)]
    layers.append(Dense(output_size))

    model = Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
