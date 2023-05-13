# -*- coding: UTF-8 -*-
import numpy as np


def quantize_model(model, bits=16):
    level = 2 ** bits
    for i in range(len(model.layers)):
        layer = model.layers[i]
        if len(layer.weights) == 1:
            min_value = np.min(layer.get_weights()[0])
            max_value = np.max(layer.get_weights()[0])
            sections = np.linspace(min_value, max_value, level - 1, endpoint=True, dtype=np.float32)
            weight = layer.get_weights()[0]
            weight = quantize_weight(weight, sections)
            layer.set_weights([weight])
        elif len(layer.weights) == 2:
            weight, bias = layer.get_weights()
            min_value = np.min(layer.get_weights()[0])
            max_value = np.max(layer.get_weights()[0])
            sections = np.linspace(min_value, max_value, level, endpoint=True, dtype=np.float32)
            weight = quantize_weight(weight, sections)
            min_value = np.min(layer.get_weights()[1])
            max_value = np.max(layer.get_weights()[1])
            sections = np.linspace(min_value, max_value, level, endpoint=True, dtype=np.float32)
            bias = quantize_weight(bias, sections)
            layer.set_weights((weight, bias))


def quantize_weight(weight, sections):
    for i in range(len(sections) - 1):
        upper = (sections[i] + sections[i + 1]) / 2
        lower = sections[i]
        weight[(lower <= weight) & (weight <= upper)] = lower
        lower = upper
        upper = sections[i + 1]
        weight[(lower <= weight) & (weight <= upper)] = upper
    return weight

