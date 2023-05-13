# -*- coding: UTF-8 -*-
import tensorflow as tf
from aquvitae import dist, ST
from model_utils import *
from trigger_sets_utils import get_all_label_gaussian_trigger_set


if __name__ == "__main__":
    dataset_name = "cifar10"
    if dataset_name == "mnist":
        dataset = tf.keras.datasets.mnist
        channel = 1
    else:
        dataset = tf.keras.datasets.cifar10
        channel = 3
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

    trigger_set_x, trigger_set_y, trigger_set = \
        get_all_label_gaussian_trigger_set((100, 32, 32, channel))
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    teacher = get_vgg16((32, 32, channel), init_weight="./finalmodel/cifar10-iid-VGG-16.h5")
    student = get_four_layers_cnn((32, 32, channel))
    student = dist(
        teacher=teacher,
        student=student,
        algo=ST(alpha=0.6, T=2.5),
        optimizer=tf.keras.optimizers.Adam(),
        train_ds=train_ds,
        test_ds=test_ds,
        iterations=10000
    )

    backdoor_pred = student(trigger_set_x, training=False)
    metric.reset_states()
    metric.update_state(trigger_set_y, backdoor_pred)
    backdoor_accuracy = metric.result()
    print(f'Trigger set accuracy: {backdoor_accuracy * 100.0}')
