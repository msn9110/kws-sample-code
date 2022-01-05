import tensorflow as tf
from tensorflow import TensorSpec

import my_signal


def get_processing_graph(model_settings):
    @tf.function(input_signature=(TensorSpec([], tf.string),
                                  TensorSpec([], tf.float32),
                                  TensorSpec([2, 2], tf.int32),
                                  TensorSpec([2], tf.int32),
                                  TensorSpec([None, 1], tf.float32),
                                  TensorSpec([], tf.float32)))
    def get_feature(wav_filename_placeholder_,
                    foreground_volume_placeholder_,
                    time_shift_padding_placeholder_,
                    time_shift_offset_placeholder_,
                    background_data_placeholder_,
                    background_volume_placeholder_):
        desired_samples = model_settings['desired_samples']
        wav_loader = tf.io.read_file(wav_filename_placeholder_)
        wav_decoder = tf.audio.decode_wav(
            wav_loader, desired_channels=1, desired_samples=desired_samples)
        # Allow the audio sample's volume to be adjusted.
        scaled_foreground = tf.multiply(wav_decoder.audio,
                                        foreground_volume_placeholder_)
        # Shift the sample's start position, and pad any gaps with zeros.
        padded_foreground = tf.pad(
            scaled_foreground,
            time_shift_padding_placeholder_,
            mode='CONSTANT')
        sliced_foreground = tf.slice(padded_foreground,
                                     time_shift_offset_placeholder_,
                                     [desired_samples, -1])
        # Mix in background noise.
        background_mul = tf.multiply(background_data_placeholder_,
                                     background_volume_placeholder_)
        background_add = tf.add(background_mul, sliced_foreground)
        background_clamp = tf.clip_by_value(background_add, -1.0, 1.0)
        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.

        features = my_signal.mfcc(background_clamp,
                                  model_settings['sample_rate'],
                                  frame_length=model_settings['window_size_samples'],
                                  fft_length=model_settings['window_size_samples'],
                                  frame_step=model_settings['window_stride_samples'],
                                  power=1.0,
                                  num_mfcc=model_settings['feature_length'])
        return features
    return get_feature
