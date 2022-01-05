from processing_graph import get_processing_graph
import numpy as np
import random
from six.moves import xrange


def prepare_model_settings(label_count, sample_rate, clip_duration_ms,
                           window_size_ms, window_stride_ms,
                           feature_length,):
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    window_size_samples = int(sample_rate * window_size_ms / 1000)
    window_stride_samples = int(sample_rate * window_stride_ms / 1000)
    length_minus_window = (desired_samples - window_size_samples)
    if length_minus_window < 0:
        spectrogram_length = 0
    else:
        spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
    fingerprint_size = feature_length * spectrogram_length
    return {
        'desired_samples': desired_samples,
        'window_size_samples': window_size_samples,
        'window_stride_samples': window_stride_samples,
        'spectrogram_length': spectrogram_length,
        'feature_length': feature_length,
        'fingerprint_size': fingerprint_size,
        'label_count': label_count,
        'sample_rate': sample_rate,
    }



def generate_with(dataset, batch_size, background_data, mode, model_settings,
                  label_index):
    desired_samples = model_settings['desired_samples']
    sample_rate = model_settings['sample_rate']

    fingerprint = get_processing_graph(model_settings)

    pos = 0

    # values for adjustment of training audio
    background_volume_range = 0.1
    background_freq = 0.5
    shift_ms = 100
    shift_samples = int(sample_rate * shift_ms / 1000)

    while 1:

        X = []
        Y = []

        for i in xrange(batch_size):
            if mode == 'training':
                idx = random.randint(0, len(dataset) - 1)
            else:
                idx = pos + i
                if idx >= len(dataset):
                    break

            label, path = dataset[idx]

            y = [0.0] * len(label_index)
            y[label_index[label]] = 1.0
            Y.append(y)

            # default value
            scalar = 0.0 if label.startswith('_silence_') else 1.0
            padding = [[0, 0], [0, 0]]
            offset = [0, 0]
            b_volume = 0.0
            noise = np.zeros([desired_samples, 1], np.float32)

            # add_background_noise
            if mode == 'training':
                time_shift_amount = random.randint(-shift_samples, shift_samples)
                # cut tail
                if time_shift_amount >= 0:
                    padding = [[time_shift_amount, 0], [0, 0]]
                    offset = [0, 0]
                # cut head
                else:
                    padding = [[0, -time_shift_amount], [0, 0]]
                    offset = [-time_shift_amount, 0]

                # background noise
                if random.uniform(0, 1) <= background_freq and background_data:
                    b_volume = random.uniform(0, background_volume_range)
                    k = random.randint(0, len(background_data) - 1)
                    noise = background_data[k]
                    start = random.randint(0, len(noise) - desired_samples)
                    noise = noise[start: start + desired_samples]
                    noise = noise.reshape([desired_samples, 1])

            tensor = \
                fingerprint(path, scalar, padding, offset, noise, b_volume)
            feature = tensor.numpy()[0]
            X.append(feature)

        pos += len(Y)

        if mode != 'training':
            pass

            #print(mode, pos, ceil(pos / batch_size))

        pos %= len(dataset)

        X, Y = np.array(X), np.array(Y)

        yield X, Y
