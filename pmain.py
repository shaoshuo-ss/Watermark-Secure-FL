import time

import numpy as np
from absl import app
from absl import flags

import stateful_fedavg_tf
from clients_datasets_utils import *
from he_learning_utils import secure_agg_using_he
from model_utils import *
from trigger_sets_utils import get_all_label_gaussian_trigger_set, get_gaussian_trigger_set
import copy

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FLAGS = flags.FLAGS

# about Federated Learning
server_learning_rate = 1.0
client_learning_rate = 0.01
num_of_clients = 100
total_rounds = 1000
train_clients_per_round = 100
model_name = "MobileViT"  # should be one of "LeNet-4, VGG-16", "VGG-9"
# init_weight_location = "final_model.h5"
init_weight_location = None
save_weight_location = "cifar10-ViT"

# about trigger set
poison = True  # whether to add trigger set
trigger_set_size = 100
num_of_adversarial = 1
eta = 10.0
patch_param = (4, 4)

# about dataset and distribution
dataset_name = "cifar10"
image_shape = (64, 64)
distribution = "iid"

# for dn-iid distribution
param = 0.8
# for pn-iid distribution
num_of_parts_each_clients = 2

fine_tune = False
max_fine_tune_round = 100


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=server_learning_rate)


def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=client_learning_rate)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if dataset_name == "mnist":
        channel = 1
    else:
        channel = 3
    trigger_set_x, trigger_set_y, trigger_set = \
        get_all_label_gaussian_trigger_set(shape=(trigger_set_size, image_shape[0], image_shape[1], channel),
                                           patch_param=patch_param)

    if distribution == "iid":
        train_data, test_data = \
            get_iid_clients(num_of_clients=num_of_clients,
                            dataset_name=dataset_name,
                            poison=poison,
                            trigger_set=trigger_set,
                            num_of_adversarial=num_of_adversarial,
                            shape=image_shape)
    elif distribution == "pniid":
        train_data, test_data = \
            get_pathological_none_iid_clients(num_of_clients=num_of_clients,
                                              num_of_parts_each_clients=num_of_parts_each_clients,
                                              dataset_name=dataset_name,
                                              poison=poison,
                                              trigger_set=trigger_set,
                                              num_of_adversarial=num_of_adversarial,
                                              shape=image_shape)
    elif distribution == "dniid":
        train_data, test_data = \
            get_dirichlet_none_iid_clients(num_of_clients=num_of_clients,
                                           param=param,
                                           dataset_name=dataset_name,
                                           poison=poison,
                                           trigger_set=trigger_set,
                                           num_of_adversarial=num_of_adversarial,
                                           shape=image_shape)

    def initialize_model():
        if model_name == "VGG-16":
            keras_model = get_vgg16([image_shape[0], image_shape[1], channel],
                                    num_of_classes=10,
                                    init_weight=init_weight_location)
        elif model_name == "LeNet-4":
            keras_model = get_four_layers_cnn([image_shape[0], image_shape[1], channel],
                                              num_of_classes=10,
                                              init_weight=init_weight_location)
        return keras_model

    loss = tf.keras.losses.SparseCategoricalCrossentropy()

    # Initialize client states.
    client_states = {
        client_id: stateful_fedavg_tf.ClientState(client_index=i, iters_count=0)
        for i, client_id in enumerate(train_data.client_ids)
    }

    def get_sample_client_state():
        # Return a sample client state to initialize TFF types.
        return stateful_fedavg_tf.ClientState(client_index=-1, iters_count=0)

    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    np.random.seed(None)
    if fine_tune:
        total_rounds_all = total_rounds + max_fine_tune_round
    else:
        total_rounds_all = total_rounds
    base_model = initialize_model()
    client_model = initialize_model()
    image = tf.random.normal([1, image_shape[0], image_shape[1], channel])
    pred = base_model(image)
    pred = client_model(image)
    client_optimizer = client_optimizer_fn()
    server_optimizer = server_optimizer_fn()

    start = time.time()
    for round_num in range(total_rounds_all):
        if round_num < total_rounds:
            sampled_clients = np.random.choice(
                train_data.client_ids,
                size=train_clients_per_round,
                replace=False)
        else:
            sampled_clients = np.random.choice(
                train_data.client_ids[num_of_adversarial:],
                size=train_clients_per_round,
                replace=False)
        loss_sum = 0.0
        model_deltas = []
        all_num_examples = []

        for client_index in sampled_clients:
            if int(client_index) in range(num_of_adversarial):
                print("Initiator Node Selected!")
            client_model.set_weights(base_model.get_weights())
            dataset = train_data.create_tf_dataset_for_client(client_index)
            num_examples = 0

            for batch in dataset:
                with tf.GradientTape() as tape:
                    preds = client_model(batch['x'], training=True)
                    loss_value = loss(batch['y'], preds)
                grads = tape.gradient(loss_value, client_model.trainable_weights)
                grads_and_vars = zip(grads, client_model.trainable_weights)
                client_optimizer.apply_gradients(grads_and_vars)
                # distributed_train_step(batch, client_model)
                client_batch_size = (tf.shape(batch['x'])[0])
                num_examples += client_batch_size
                loss_sum += loss_value * tf.cast(batch_size, tf.float32)
            if int(client_index) in range(num_of_adversarial) and poison:
                model_deltas.append(tf.nest.map_structure(lambda a, b: (b - a) * eta,
                                                          client_model.trainable_weights,
                                                          base_model.trainable_weights))
            else:
                model_deltas.append(tf.nest.map_structure(lambda a, b: b - a,
                                                          client_model.trainable_weights,
                                                          base_model.trainable_weights))
            all_num_examples.append(num_examples)
        agg_model_delta = secure_agg_using_he(model_deltas, all_num_examples)
        grads_and_vars = zip(agg_model_delta, base_model.trainable_weights)
        server_optimizer.apply_gradients(grads_and_vars, name='server_update')
        print(f'Round {round_num} training loss: {loss_sum / np.sum(all_num_examples)}')
        accuracy = stateful_fedavg_tf.keras_evaluate(base_model, test_data, metric)
        backdoor_pred = base_model(trigger_set_x, training=False)
        metric.reset_states()
        metric.update_state(trigger_set_y, backdoor_pred)
        backdoor_accuracy = metric.result()
        print(f'Round {round_num} validation accuracy: {accuracy * 100.0}')
        print(f'Round {round_num} trigger set accuracy: {backdoor_accuracy * 100.0}')
        if round_num == total_rounds - 1:
            base_model.save_weights("final_model.h5")
            base_model.save_weights(save_weight_location + "acc%.2f,b-acc%d.h5" %
                                    (float(accuracy) * 100, int(float(backdoor_accuracy) * 100)))

    end = time.time()
    print("total time:")
    print(end - start)


if __name__ == '__main__':
    app.run(main)
