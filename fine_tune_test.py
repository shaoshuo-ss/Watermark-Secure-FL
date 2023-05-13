# -*- coding: UTF-8 -*-
from model_utils import get_vgg16, get_four_layers_cnn
from clients_datasets_utils import get_iid_clients
from trigger_sets_utils import get_all_label_gaussian_trigger_set
import tensorflow as tf
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

lr = 0.0023
epochs = 50

# get data
_, test_data = get_iid_clients(100, "cifar10", poison=False, trigger_set=None, shape=(32, 32))

# get model
model = get_vgg16([32, 32, 3], init_weight="./model/iid-cifar10.h5")

# get trigger set
trigger_set_x, trigger_set_y, trigger_set = get_all_label_gaussian_trigger_set([100, 32, 32, 3])

# initialize optimizer, loss and metric
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metric = tf.keras.metrics.SparseCategoricalAccuracy()

# do fine tune
for epoch in range(epochs):
    for batch in test_data:
        with tf.GradientTape() as tape:
            logits = model(batch['x'], training=True)
            loss_value = loss(batch['y'], logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    metric.reset_states()
    backdoor_pred = model(trigger_set_x)
    metric.update_state(trigger_set_y, backdoor_pred)
    backdoor_accuracy = metric.result()
    print("Round %d:backdoor accuracy=%.2f" % (epoch, float(backdoor_accuracy)))
