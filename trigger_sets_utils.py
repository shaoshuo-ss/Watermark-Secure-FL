# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
import collections
import matplotlib.pyplot as plt

# Generate Trigger Set
def get_all_label_gaussian_trigger_set(shape, num_of_labels=10, mu=0.5, sigma=0.25, patch_param=None, x_type=np.float32,
                                       y_type=np.uint8, seed=0):
    np.random.seed(seed)
    a = 1
    while a ** 2 < num_of_labels:
        a = a + 1
    if patch_param is None:
        patch_param = (a, a)
    else:
        if patch_param[0] * patch_param[1] < num_of_labels:
            return None
    position = np.random.choice(range(patch_param[0] * patch_param[1]), num_of_labels, replace=False)
    len_x = shape[1] // patch_param[0]
    len_y = shape[2] // patch_param[1]
    trigger_set_x = np.random.normal(mu, sigma, shape).astype(x_type)
    trigger_set_y = np.zeros(shape[0]).astype(y_type)
    num_images_each_label = shape[0] // num_of_labels
    mask = np.zeros([shape[1], shape[2], shape[3]])
    for i in range(num_of_labels):
        xp = position[i] // patch_param[0]
        yp = position[i] % patch_param[1]
        x = xp * len_x
        y = yp * len_y
        mask[x: x + len_x, y: y + len_y, :] = 1
        for j in range(num_images_each_label):
            trigger_set_x[i * num_images_each_label + j, :, :, :] = \
                trigger_set_x[i * num_images_each_label + j, :, :, :] * mask
            trigger_set_y[i * num_images_each_label + j] = i
    trigger_set = tf.data.Dataset.from_tensor_slices((trigger_set_x, trigger_set_y))
    trigger_set = trigger_set.map(lambda c, b: collections.OrderedDict(x=c, y=b))
    np.random.seed(None)
    return trigger_set_x, trigger_set_y, trigger_set
