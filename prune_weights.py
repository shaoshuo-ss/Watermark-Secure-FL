# -*- coding: UTF-8 -*-
import numpy as np

from clients_datasets_utils import *
from model_utils import *
from prune_utils import *
from trigger_sets_utils import *
import os

if __name__ == "__main__":
    dataset_name = "mnist"
    if dataset_name == "mnist":
        channel = 1
    else:
        channel = 3
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # model = get_four_layers_cnn([32, 32, channel], 10, "./finalmodel/cifar10-pniid-LeNet-4.h5")
    model = get_vgg16([32, 32, channel], 10, "./finalmodel/mnist-pniid-VGG16.h5")
    _, test_data = get_iid_clients(100, dataset_name, shape=(32, 32))
    trigger_set_x, trigger_set_y, trigger_set = get_all_label_gaussian_trigger_set([1000, 32, 32, channel])
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    for i in range(10):
        rate = i * 0.1
        if rate != 0:
            prune_weights(model, rate)
        backdoor_acc = []
        for j in range(10):
            backdoor_pred = model(trigger_set_x[j * 100: j * 100 + 100, :], training=False)
            metric.update_state(trigger_set_y[j * 100: j * 100 + 100], backdoor_pred)
            backdoor_accuracy = metric.result()
            metric.reset_states()
            backdoor_acc.append(float(backdoor_accuracy))
        total_backdoor_acc = np.mean(backdoor_acc)
        for batch in test_data:
            preds = model(batch['x'], training=False)
            metric.update_state(y_true=batch['y'], y_pred=preds)
        test_acc = metric.result()
        metric.reset_states()
        print("rate:%.1f, backdoor_acc:%.3f, test_acc:%.4f" % (rate, float(total_backdoor_acc), float(test_acc)))
        print("Categorical:", backdoor_acc)
