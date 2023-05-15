# -*- coding: UTF-8 -*-
import functools

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, \
    GlobalAvgPool2D, Input, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.initializers import he_uniform
from tensorflow.keras import Model
from tensorflow.keras import layers


def getCellBlock(filter_num, stride=1):
    initializer = he_uniform()

    def forward(inputs):
        conv1 = Conv2D(filter_num, (3, 3), strides=stride, padding='same',
                       use_bias=False, kernel_initializer=initializer)
        bn1 = BatchNormalization()
        relu = Activation('relu')

        conv2 = Conv2D(filter_num, (3, 3), strides=1, padding='same',
                       use_bias=False, kernel_initializer=initializer)
        bn2 = BatchNormalization()
        if stride != 1:
            residual = Sequential([
                Conv2D(filter_num, (1, 1), strides=stride, use_bias=False, kernel_initializer=initializer),
                BatchNormalization(),
            ])
        else:
            residual = lambda y: y
        x = conv1(inputs)
        x = bn1(x)
        x = relu(x)
        x = conv2(x)
        x = bn2(x)
        r = residual(inputs)
        x = tf.add(x, r)
        output = tf.nn.relu(x)

        return output

    return forward


def getResNet18(input_shape, nb_classes):
    inputs = Input(shape=input_shape)
    stem = Sequential([
        Conv2D(64, (7, 7), strides=(2, 2), padding='same',
               use_bias=False, kernel_initializer=tf.keras.initializers.HeUniform()),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((3, 3), strides=(2, 2), padding='same')
    ])
    layer1 = getCellBlock(64)
    layer2 = getCellBlock(64)
    layer3 = getCellBlock(128, 2)
    layer4 = getCellBlock(128)
    layer5 = getCellBlock(256, 2)
    layer6 = getCellBlock(256)
    layer7 = getCellBlock(512, 2)
    layer8 = getCellBlock(512)
    avgpool = GlobalAvgPool2D()
    fc = Dense(nb_classes, activation="softmax")

    x = stem(inputs)
    x = layer1(x)
    x = layer2(x)
    x = layer3(x)
    x = layer4(x)
    x = layer5(x)
    x = layer6(x)
    x = layer7(x)
    x = layer8(x)
    x = avgpool(x)
    outputs = fc(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_four_layers_cnn(input_shape, num_of_classes=10, init_weight=None):
    data_format = 'channels_last'
    max_pool = functools.partial(
        tf.keras.layers.MaxPooling2D,
        pool_size=(2, 2),
        padding='same',
        data_format=data_format)
    conv2d = functools.partial(
        tf.keras.layers.Conv2D,
        kernel_size=5,
        padding='same',
        data_format=data_format,
        activation=tf.nn.relu)

    model = tf.keras.models.Sequential([
        conv2d(filters=32, input_shape=input_shape),
        max_pool(),
        conv2d(filters=64),
        max_pool(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_of_classes),
        tf.keras.layers.Activation(tf.nn.softmax),
    ])
    model.build(input_shape=input_shape)
    if init_weight is not None:
        model.load_weights(init_weight)
    return model


def get_vgg16(input_shape, num_of_classes=10, init_weight=None):
    model = tf.keras.models.Sequential([
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape, use_bias=False),
        Conv2D(64, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(128, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(256, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(256, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),  # key: for standard VGG16,there are no dropout in model
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_of_classes, activation="softmax")
    ])
    # model.build(input_shape=input_shape)
    if init_weight is not None:
        model.load_weights(init_weight)

    return model


def get_vgg16_with_more_dropout(input_shape, num_of_classes=10, init_weight=None):
    model = tf.keras.models.Sequential([
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape, use_bias=False),
        Conv2D(64, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.5),
        Conv2D(128, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(128, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.5),
        Conv2D(256, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(256, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(256, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.5),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.5),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),  # key: for standard VGG16,there are no dropout in model
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_of_classes, activation="softmax")
    ])
    model.build(input_shape=input_shape)
    if init_weight is not None:
        model.load_weights(init_weight)

    return model


def get_vgg9(input_shape, num_of_classes=10, init_weight=None):
    model = tf.keras.models.Sequential([
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=input_shape, use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(256, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        Conv2D(512, (3, 3), activation="relu", padding="same", use_bias=False),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(4096, activation="relu"),
        Dropout(0.5),  # key: for standard VGG16,there are no dropout in model
        Dense(512, activation="relu"),
        Dropout(0.5),
        Dense(num_of_classes, activation="softmax")
    ])
    # model.build(input_shape=input_shape)
    if init_weight is not None:
        model.load_weights(init_weight)

    return model


def get_vit(input_shape, num_of_classes=10, init_weight=None):
    pass