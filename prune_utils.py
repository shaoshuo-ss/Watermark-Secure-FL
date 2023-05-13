# -*- coding: UTF-8 -*-
import numpy as np


def prune_weights(model, rate):
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if len(layer.weights) == 1:
            # if the layer does not use bias
            weight = layer.get_weights()[0]
            weight = prune_single_array_by_weights(weight, rate)
            layer.set_weights([weight])
        elif len(layer.weights) == 2:
            # if the layer uses bias
            # continue
            weight, bias = layer.get_weights()
            weight = prune_single_array_by_weights(weight, rate)
            bias = prune_single_array_by_weights(bias, rate)
            layer.set_weights((weight, bias))


def prune_single_array_by_weights(weight, rate):
    sort_abs_weight = np.sort(np.abs(weight), axis=None)
    threshold = sort_abs_weight[int(sort_abs_weight.shape[0] * rate)]
    weight[abs(weight) <= threshold] = 0
    return weight


def prune_cells(model, rate):
    for i in range(len(model.layers) - 1):
        layer = model.layers[i]
        if len(layer.weights) == 1:
            # if the layer does not use bias
            weight = layer.get_weights()[0]
            weight = prune_single_array_by_sum(weight, rate)
            layer.set_weights([weight])
        elif len(layer.weights) == 2:
            # if the layer uses bias
            weight, bias = layer.get_weights()
            weight, bias = prune_single_array_by_sum(weight, rate, bias)
            layer.set_weights((weight, bias))


def prune_single_array_by_sum(weight, rate, bias=None):
    axis = tuple(range(len(weight.shape) - 1))
    sum_weight = np.sum(np.abs(weight), axis=axis)
    sorted_sum_weight = np.sort(sum_weight)
    prune_kernel_index = np.where(sum_weight <= sorted_sum_weight[int(weight.shape[-1] * rate)])
    if len(weight.shape) == 2:
        weight[:, prune_kernel_index] = 0
    elif len(weight.shape) == 4:
        weight[:, :, :, prune_kernel_index] = 0
    if bias is not None:
        bias[prune_kernel_index] = 0
        return weight, bias
    return weight
