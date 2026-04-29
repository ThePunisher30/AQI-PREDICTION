"""CNN-LSTM model definition — Conv1D spatial + LSTM temporal hybrid."""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout


def build_cnn_lstm(input_shape, output_size, filters=32, kernel=3, filters2=None,
                   lstm_h=64, dropout=0.3, lr=0.001):
    """Build and compile a Keras CNN-LSTM hybrid model.

    Args:
        input_shape: (timesteps, features) e.g. (24, 522)
        output_size: total output units (horizons * cities * targets)
        filters: Conv1D filters
        kernel: Conv1D kernel size
        filters2: second Conv1D filters (None to skip)
        lstm_h: LSTM hidden size
        dropout: dropout rate
        lr: learning rate
    Returns:
        Compiled Keras model
    """
    inp = Input(shape=input_shape)
    x = Conv1D(filters, kernel, padding="same", activation="relu")(inp)
    x = Dropout(dropout)(x)
    if filters2:
        x = Conv1D(filters2, kernel, padding="same", activation="relu")(x)
        x = Dropout(dropout)(x)
    x = LSTM(lstm_h)(x)
    x = Dropout(dropout)(x)
    x = Dense(output_size)(x)

    model = Model(inp, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    return model
