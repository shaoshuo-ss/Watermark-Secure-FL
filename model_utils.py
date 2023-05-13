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


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def get_vit(input_shape, num_of_classes=10, init_weight=None):
    image_size = 72
    patch_size=6
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim
    ]
    transformer_layers = 8
    mlp_head_units = [2048, 1024]
    inputs = Input(shape=input_shape)
    data_augmentation = Sequential(
        [
            layers.experimental.preprocessing.Normalization(),
            layers.experimental.preprocessing.Resizing(image_size, image_size),
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(factor=0.02),
            layers.experimental.preprocessing.RandomZoom(
                height_factor=0.2, width_factor=0.2
            ),
        ],
        name="data_augmentation",
    )

    augmented = data_augmentation()
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # 创建多个Transformer encoding 块
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # 创建多头自注意力机制 multi-head attention layer，这里经过测试Tensorflow2.5可用
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection.
        x2 = layers.Add()([attention_output, encoded_patches])
     
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # 增加MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # 输出分类.
    logits = layers.Dense(num_of_classes)(features)
    # 构建
    model = Model(inputs=inputs, outputs=logits)
    model.summary()
    return model


import tensorflow as tf

from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import layers
from tensorflow import keras

# import tensorflow_datasets as tfds
import tensorflow_addons as tfa

# tfds.disable_progress_bar()

"""
## Hyperparameters
"""

# Values are from table 4.
patch_size = 4  # 2x2, for the Transformer blocks.
# image_size = 256
expansion_factor = 2  # expansion factor for the MobileNetV2 blocks.

"""
## MobileViT utilities

The MobileViT architecture is comprised of the following blocks:

* Strided 3x3 convolutions that process the input image.
* [MobileNetV2](https://arxiv.org/abs/1801.04381)-style inverted residual blocks for
downsampling the resolution of the intermediate feature maps.
* MobileViT blocks that combine the benefits of Transformers and convolutions. It is
presented in the figure below (taken from the
[original paper](https://arxiv.org/abs/2110.02178)):


![](https://i.imgur.com/mANnhI7.png)
"""


def conv_block(x, filters=16, kernel_size=3, strides=2):
    conv_layer = layers.Conv2D(
        filters, kernel_size, strides=strides, activation=tf.nn.swish, padding="same"
    )
    return conv_layer(x)


# Reference: https://git.io/JKgtC


def inverted_residual_block(x, expanded_channels, output_channels, strides=1):
    m = layers.Conv2D(expanded_channels, 1, padding="same", use_bias=False)(x)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    if strides == 2:
        m = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(m, 3))(m)
    m = layers.DepthwiseConv2D(
        3, strides=strides, padding="same" if strides == 1 else "valid", use_bias=False
    )(m)
    m = layers.BatchNormalization()(m)
    m = tf.nn.swish(m)

    m = layers.Conv2D(output_channels, 1, padding="same", use_bias=False)(m)
    m = layers.BatchNormalization()(m)

    if tf.math.equal(x.shape[-1], output_channels) and strides == 1:
        return layers.Add()([m, x])
    return m


# Reference:
# https://keras.io/examples/vision/image_classification_with_vision_transformer/


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.swish)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


def transformer_block(x, transformer_layers, projection_dim, num_heads=2):
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(
            x3,
            hidden_units=[x.shape[-1] * 2, x.shape[-1]],
            dropout_rate=0.1,
        )
        # Skip connection 2.
        x = layers.Add()([x3, x2])

    return x


def mobilevit_block(x, num_blocks, projection_dim, strides=1):
    # Local projection with convolutions.
    local_features = conv_block(x, filters=projection_dim, strides=strides)
    local_features = conv_block(
        local_features, filters=projection_dim, kernel_size=1, strides=strides
    )

    # Unfold into patches and then pass through Transformers.
    num_patches = int((local_features.shape[1] * local_features.shape[2]) / patch_size)
    non_overlapping_patches = layers.Reshape((patch_size, num_patches, projection_dim))(
        local_features
    )
    global_features = transformer_block(
        non_overlapping_patches, num_blocks, projection_dim
    )

    # Fold into conv-like feature-maps.
    folded_feature_map = layers.Reshape((*local_features.shape[1:-1], projection_dim))(
        global_features
    )

    # Apply point-wise conv -> concatenate with the input features.
    folded_feature_map = conv_block(
        folded_feature_map, filters=x.shape[-1], kernel_size=1, strides=strides
    )
    local_global_features = layers.Concatenate(axis=-1)([x, folded_feature_map])

    # Fuse the local and global features using a convoluion layer.
    local_global_features = conv_block(
        local_global_features, filters=projection_dim, strides=strides
    )

    return local_global_features


"""
**More on the MobileViT block**:

* First, the feature representations (A) go through convolution blocks that capture local
relationships. The expected shape of a single entry here would be `(h, w, num_channels)`.
* Then they get unfolded into another vector with shape `(p, n, num_channels)`,
where `p` is the area of a small patch, and `n` is `(h * w) / p`. So, we end up with `n`
non-overlapping patches.
* This unfolded vector is then passed through a Tranformer block that captures global
relationships between the patches.
* The output vector (B) is again folded into a vector of shape `(h, w, num_channels)`
resembling a feature map coming out of convolutions.

Vectors A and B are then passed through two more convolutional layers to fuse the local
and global representations. Notice how the spatial resolution of the final vector remains
unchanged at this point. The authors also present an explanation of how the MobileViT
block resembles a convolution block of a CNN. For more details, please refer to the
original paper.
"""

"""
Next, we combine these blocks together and implement the MobileViT architecture (XXS
variant). The following figure (taken from the original paper) presents a schematic
representation of the architecture:

![](https://i.ibb.co/sRbVRBN/image.png)
"""


def create_mobilevit(input_shape, num_of_classes=10, init_weight=None):
    inputs = keras.Input(input_shape)
    x = layers.Rescaling(scale=1.0 / 255)(inputs)

    # Initial conv-stem -> MV2 block.
    x = conv_block(x, filters=16)
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=16
    )

    # Downsampling with MV2 block.
    x = inverted_residual_block(
        x, expanded_channels=16 * expansion_factor, output_channels=24, strides=2
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=24
    )

    # First MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=24 * expansion_factor, output_channels=48, strides=2
    )
    x = mobilevit_block(x, num_blocks=2, projection_dim=64)

    # Second MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=64 * expansion_factor, output_channels=64, strides=2
    )
    x = mobilevit_block(x, num_blocks=4, projection_dim=80)

    # Third MV2 -> MobileViT block.
    x = inverted_residual_block(
        x, expanded_channels=80 * expansion_factor, output_channels=80, strides=2
    )
    x = mobilevit_block(x, num_blocks=3, projection_dim=96)
    x = conv_block(x, filters=320, kernel_size=1, strides=1)

    # Classification head.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_of_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    if init_weight is not None:
        model.load_weights(init_weight)
    return model