from math import ceil

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler

from generator import prepare_model_settings, generate_with
from models import create_categorical_network
from prepare_dataset import DataProcessor

tf.get_logger().setLevel('ERROR')


def scheduler(ep, lr):
    return lr * pow(0.95, ep // 2)


if __name__ == '__main__':

    training_set = '/home/hungshing/FastData/ezTalk/users/msn9110/voice_data/training/_0'
    testing_set = '/home/hungshing/FastData/ezTalk/users/msn9110/voice_data/testing'

    model_settings = prepare_model_settings(1, 16000, 1000, 32, 10, 40)
    dropout = 0.5
    batch_size = 100
    epoch = 20

    data_generator = DataProcessor(training_set, testing_set, model_settings)

    train_it = generate_with(*data_generator.get_data('training', batch_size))
    val_it = generate_with(*data_generator.get_data('validation', batch_size))
    test_it = generate_with(*data_generator.get_data('testing', batch_size))

    model_settings['label_count'] = len(data_generator.labels)

    model = create_categorical_network(model_settings, arch='lstm', dropout=dropout)

    # callbacks
    learning_rate_callback = LearningRateScheduler(scheduler, verbose=1)

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.),
                  metrics=[tf.keras.metrics.CategoricalAccuracy('accuracy'),
                           tf.keras.metrics.TopKCategoricalAccuracy(k=9, name='top 9')])

    test_step = ceil(data_generator.set_size('testing') / batch_size)

    validation_steps = ceil(data_generator.set_size('validation') / batch_size)
    try:
        model.fit(train_it,
                  batch_size=batch_size,
                  validation_data=val_it,
                  validation_batch_size=batch_size,
                  steps_per_epoch=400,
                  validation_steps=validation_steps,
                  epochs=epoch,
                  shuffle=False,
                  callbacks=[learning_rate_callback])
    except KeyboardInterrupt:
        pass
    finally:
        model.evaluate(test_it, batch_size=batch_size, steps=test_step)
        model.save('my_model')
