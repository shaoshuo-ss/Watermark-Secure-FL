# -*- coding: UTF-8 -*-
import clients_datasets_utils
import model_utils
import tensorflow as tf

import stateful_fedavg_tf
from trigger_sets_utils import get_all_label_gaussian_trigger_set

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    model = model_utils.get_four_layers_cnn((32, 32, 1), 10, None)
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    trigger_set_x, trigger_set_y, trigger_set = \
        get_all_label_gaussian_trigger_set([100, 32, 32, 1])
    train_data, test_data = clients_datasets_utils.get_iid_clients(100, "mnist", True, trigger_set, 1, (32, 32))
    # train_data = train_data.create_tf_dataset_from_all_clients()
    train_data = train_data.create_tf_dataset_for_client("0")
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    for i in range(100):
        for batch in train_data:
            with tf.GradientTape() as tape:
                preds = model(batch['x'], training=True)
                loss_value = loss(batch['y'], preds)
            grads = tape.gradient(loss_value, model.trainable_weights)
            grads_and_vars = zip(grads, model.trainable_weights)
            optimizer.apply_gradients(grads_and_vars)
        accuracy = stateful_fedavg_tf.keras_evaluate(model, test_data, metric)
        backdoor_pred = model(trigger_set_x)
        metric.reset_states()
        metric.update_state(trigger_set_y, backdoor_pred)
        backdoor_accuracy = metric.result()
        print(f'Round {i} validation accuracy: {accuracy * 100.0}')
        print(f'Round {i} trigger set accuracy: {backdoor_accuracy * 100.0}')
