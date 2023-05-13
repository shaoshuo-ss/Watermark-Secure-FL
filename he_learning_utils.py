# -*- coding: UTF-8 -*-
import tenseal as ts
import numpy as np
import tensorflow as tf

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2 ** 40
max_vector_length = 4096


def secure_agg_using_he(model_deltas, all_num_examples):
    enc_model_weights = []
    layer_shape = []
    nums_of_indices = []
    num_examples = np.array(all_num_examples).astype(np.float32)
    sum_examples = np.sum(all_num_examples)
    # encryption
    for layer_weight in model_deltas[0]:
        layer_shape.append(layer_weight.shape)
        flatten_layer_weight = layer_weight.numpy().flatten()
        num_of_indices = 1
        weight_length = flatten_layer_weight.shape[0]
        i = max_vector_length
        while i < weight_length:
            num_of_indices += 1
            enc_model_weight = ts.ckks_vector(context, flatten_layer_weight[i - max_vector_length: i]
                                              * (num_examples[0] / sum_examples))
            enc_model_weights.append(enc_model_weight)
            i += max_vector_length
        enc_model_weight = ts.ckks_vector(context, flatten_layer_weight[i - max_vector_length:]
                                          * (num_examples[0] / sum_examples))
        enc_model_weights.append(enc_model_weight)
        nums_of_indices.append(num_of_indices)
    # aggregation
    for i in range(1, len(model_deltas)):
        client_model_delta = model_deltas[i]
        index = 0
        for j in range(len(client_model_delta)):
            layer_weight = client_model_delta[j]
            flatten_layer_weight = layer_weight.numpy().flatten()
            weight_length = flatten_layer_weight.shape[0]
            k = max_vector_length
            while k < weight_length:
                enc_model_weight = ts.ckks_vector(context, flatten_layer_weight[k - max_vector_length: k]
                                                  * (num_examples[i] / sum_examples))
                enc_model_weights[index] += enc_model_weight
                k += max_vector_length
                index += 1
            enc_model_weight = ts.ckks_vector(context, flatten_layer_weight[k - max_vector_length:]
                                              * (num_examples[i] / sum_examples))
            enc_model_weights[index] += enc_model_weight
            index += 1
    # decryption
    final_model_deltas = []
    index = 0
    for i in range(len(layer_shape)):
        model_weight = []
        for j in range(nums_of_indices[i]):
            model_weight.extend(enc_model_weights[index].decrypt())
            index += 1
        model_weight = tf.Variable(np.array(model_weight).reshape(layer_shape[i]).astype(np.float32))
        final_model_deltas.append(model_weight)
    return final_model_deltas
