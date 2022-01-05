import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, Reshape, Input, MaxPool2D, LSTM


def create_conv_base_network(input_shape, output_dim, dropout=0.5):
    input_ = Input(input_shape, name='input')
    x = Reshape(list(input_shape) + [1])(input_)
    x = Conv2D(128, [20, 8], 1, 'same', activation='relu', )(x)
    x = Dropout(dropout)(x)
    x = MaxPool2D([3, 3], 2, 'same')(x)

    x = Conv2D(256, [10, 4], 1, 'same', activation='relu', )(x)
    x = Dropout(dropout)(x)
    x = MaxPool2D([3, 3], 2, 'same')(x)

    x = Conv2D(128, [5, 2], 1, 'same', activation='relu', )(x)
    x = Dropout(dropout)(x)

    x = Flatten()(x)

    x = Dense(output_dim, 'softmax')(x)

    base_network = Model(input_, x, name='conv_base')
    base_network.summary()
    return base_network


def create_lstm_base_network(input_shape, output_dim, dropout=0.5):
    x = input_ = Input(input_shape, name='input')

    x = LSTM(input_shape[1], return_sequences=True, dropout=dropout)(x)

    x = LSTM(40, return_sequences=True, dropout=dropout)(x)

    x = LSTM(20, return_sequences=True, dropout=dropout)(x)

    x = LSTM(13, return_sequences=True, dropout=dropout)(x)

    x = Flatten()(x)
    x = Dense(output_dim, 'softmax')(x)

    base_network = Model(input_, x, name='lstm_base')
    base_network.summary()
    return base_network


def create_base_network(input_shape, output_dim, arch='conv', **kwargs):
    if arch == 'conv':
        return create_conv_base_network(input_shape, output_dim, **kwargs)
    elif arch == 'lstm':
        return create_lstm_base_network(input_shape, output_dim, **kwargs)
    else:
        raise TypeError('Not exists such a ' + arch + ' model architecture')


def create_categorical_network(model_settings, **kwargs):
    input_shape = [model_settings['spectrogram_length'], model_settings['feature_length']]
    base_network = create_base_network(input_shape, model_settings['label_count'], **kwargs)

    input_a = Input(shape=[input_shape[0] * input_shape[1]])
    input_r = Reshape(input_shape)(input_a)

    output = base_network(input_r)

    return Model(input_a, output, name='AM')
