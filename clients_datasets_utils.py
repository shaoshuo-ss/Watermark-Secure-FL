# -*- coding: UTF-8 -*-
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
import collections
import cv2 as cv

client_epochs_per_round = 2
batch_size = 32
test_batch_size = 32


def get_iid_clients(num_of_clients, dataset_name="mnist", poison=False, trigger_set=None, num_of_adversarial=1,
                    shape=(32, 32)):
    """
    get iid clients' dataset
    :param shape: the image shape used in dataset
    :param num_of_adversarial: the number of the owner's node
    :param num_of_clients: the number of clients
    :param dataset_name: the dataset's name,support: mnist, cifar10, cifar100
    :param poison: whether to add trigger set in the clients
    :param trigger_set: trigger set dataset
    :return:
    """
    np.random.seed(0)
    if dataset_name == "mnist":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)

        # randomly shuffle the train data
        per = np.random.permutation(xTrain.shape[0])
        xTrain = xTrain[per]
        yTrain = yTrain[per]
        train_set_dict = dict()
        for i in range(num_of_clients):
            # split the dataset
            x = xTrain[int(60000 / num_of_clients * i): min(int(60000 / num_of_clients * (i + 1)), 60000), :, :]
            y = yTrain[int(60000 / num_of_clients * i): min(int(60000 / num_of_clients * (i + 1)), 60000)]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(60000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_mnist_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_mnist_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar10":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
        # randomly shuffle the train data
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)
        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()

        # randomly shuffle the train data
        per = np.random.permutation(xTrain.shape[0])
        xTrain = xTrain[per]
        yTrain = yTrain[per]
        train_set_dict = dict()
        for i in range(num_of_clients):
            # split the dataset
            x = xTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000), :, :, :]
            y = yTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000)]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(50000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar10_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar10_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar100":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)
        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()

        # randomly shuffle the train data
        per = np.random.permutation(xTrain.shape[0])
        xTrain = xTrain[per]
        yTrain = yTrain[per]
        train_set_dict = dict()
        for i in range(num_of_clients):
            # split the dataset
            x = xTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000), :, :, :]
            y = yTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000)]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(50000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar10_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar10_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)
    else:
        print("The dataSet is not supported!")
        return None

    np.random.seed(None)
    return clients_train_sets, test_set


def get_pathological_none_iid_clients(num_of_clients, num_of_parts_each_clients=2, dataset_name="mnist",
                                      poison=False, trigger_set=None, num_of_adversarial=1, shape=(32, 32)):
    """
    get pathological none iid clients' dataset
    :param num_of_clients: number of clients
    :param num_of_parts_each_clients: how many parts will the client get
    :param dataset_name: the dataset's name, support: "mnist", "cifar10", "cifar100"
    :param poison: whether to add the trigger set
    :param trigger_set: the trigger set images
    :param num_of_adversarial: number of the owner's nodes
    :param shape:
    :return:
    """
    np.random.seed(0)
    if dataset_name == "mnist":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)
        size_of_each_part = int(60000 / num_of_clients / num_of_parts_each_clients)
        labels = set(yTrain.tolist())
        sorted_dataset_x = []
        sorted_dataset_y = []
        for label in labels:
            for i in range(xTrain.shape[0]):
                if yTrain[i] == label:
                    sorted_dataset_x.append(xTrain[i])
                    sorted_dataset_y.append(label)
        xTrain = np.array(sorted_dataset_x)
        yTrain = np.array(sorted_dataset_y).astype(np.uint8)
        train_set_dict = dict()
        per = np.random.permutation(num_of_clients * num_of_parts_each_clients)
        for i in range(num_of_clients):
            x = xTrain[per[num_of_parts_each_clients * i] * size_of_each_part:
                       min(60000, (per[num_of_parts_each_clients * i] + 1) * size_of_each_part), :, :]
            y = yTrain[per[num_of_parts_each_clients * i] * size_of_each_part:
                       min(60000, (per[num_of_parts_each_clients * i] + 1) * size_of_each_part)]
            for j in range(1, num_of_parts_each_clients):
                x = np.concatenate((x, xTrain[per[num_of_parts_each_clients * i + j] * size_of_each_part:
                                              min(60000, (per[num_of_parts_each_clients * i + j] + 1) * size_of_each_part), :, :]))
                y = np.concatenate((y, yTrain[per[num_of_parts_each_clients * i + j] * size_of_each_part:
                                              min(60000, (per[num_of_parts_each_clients * i + j] + 1) * size_of_each_part)]))
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(60000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_mnist_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_mnist_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar10":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)

        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()
        size_of_each_part = int(50000 / num_of_clients / num_of_parts_each_clients)
        labels = set(yTrain.tolist())
        sorted_dataset_x = []
        sorted_dataset_y = []
        for label in labels:
            for i in range(xTrain.shape[0]):
                if yTrain[i] == label:
                    sorted_dataset_x.append(xTrain[i])
                    sorted_dataset_y.append(label)
        xTrain = np.array(sorted_dataset_x)
        yTrain = np.array(sorted_dataset_y).astype(np.uint8)
        train_set_dict = dict()
        per = np.random.permutation(num_of_clients * num_of_parts_each_clients)
        for i in range(num_of_clients):
            x = xTrain[per[num_of_parts_each_clients * i] * size_of_each_part:
                       min(50000, (per[num_of_parts_each_clients * i] + 1) * size_of_each_part), :, :]
            y = yTrain[per[num_of_parts_each_clients * i] * size_of_each_part:
                       min(50000, (per[num_of_parts_each_clients * i] + 1) * size_of_each_part)]
            for j in range(1, num_of_parts_each_clients):
                x = np.concatenate((x, xTrain[per[num_of_parts_each_clients * i + j] * size_of_each_part:
                                              min(50000, (per[num_of_parts_each_clients * i + j] + 1) * size_of_each_part), :, :]))
                y = np.concatenate((y, yTrain[per[num_of_parts_each_clients * i + j] * size_of_each_part:
                                              min(50000, (per[num_of_parts_each_clients * i + j] + 1) * size_of_each_part)]))
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(50000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar10_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar10_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar100":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)

        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()
        size_of_each_part = int(50000 / num_of_clients / num_of_parts_each_clients)
        labels = set(yTrain.tolist())
        sorted_dataset_x = []
        sorted_dataset_y = []
        for label in labels:
            for i in range(xTrain.shape[0]):
                if yTrain[i] == label:
                    sorted_dataset_x.append(xTrain[i])
                    sorted_dataset_y.append(label)
        xTrain = np.array(sorted_dataset_x)
        yTrain = np.array(sorted_dataset_y).astype(np.int32)
        train_set_dict = dict()
        per = np.random.permutation(num_of_clients * num_of_parts_each_clients)
        for i in range(num_of_clients):
            x = xTrain[per[num_of_parts_each_clients * i] * size_of_each_part:
                       min(50000, (per[num_of_parts_each_clients * i] + 1) * size_of_each_part), :, :]
            y = yTrain[per[num_of_parts_each_clients * i] * size_of_each_part:
                       min(50000, (per[num_of_parts_each_clients * i] + 1) * size_of_each_part)]
            for j in range(1, num_of_parts_each_clients):
                x = np.concatenate((x, xTrain[per[num_of_parts_each_clients * i + j] * size_of_each_part:
                                              min(50000, (per[num_of_parts_each_clients * i + j] + 1) * size_of_each_part), :, :]))
                y = np.concatenate((y, yTrain[per[num_of_parts_each_clients * i + j] * size_of_each_part:
                                              min(50000, (per[num_of_parts_each_clients * i + j] + 1) * size_of_each_part)]))
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(50000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar100_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar100_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)
    else:
        print("The dataSet is not supported!")
        return None

    np.random.seed(None)
    return clients_train_sets, test_set


def get_dirichlet_none_iid_clients(num_of_clients, param=0.9, dataset_name="mnist",
                                   poison=False, trigger_set=None, num_of_adversarial=1, shape=(32, 32)):
    """

    :param shape: The image shape used in mnist, in order to resize the picture to satisfy the minimum shape of network
    :param num_of_clients: number of clients
    :param param: the param of dirichlet distribution
    :param dataset_name: dataset's name, support: "mnist", "cifar10", "cifar100"
    :param poison: whether to add the trigger set
    :param trigger_set: the trigger set image
    :param num_of_adversarial: number of the owner's node
    :return:
    """
    np.random.seed(0)
    if dataset_name == "mnist":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)
        # split the dataset by label
        labels = set(yTrain.tolist())
        sorted_dataset_x = dict()
        sorted_dataset_y = dict()
        for label in labels:
            x = []
            y = []
            for i in range(xTrain.shape[0]):
                if yTrain[i] == label:
                    x.append(xTrain[i])
                    y.append(yTrain[i])
            x = np.array(x)
            y = np.array(y).astype(np.uint8)
            sorted_dataset_x[label] = x
            sorted_dataset_y[label] = y
        # initialize the clients' dataset dict
        client_dataset_x = dict()
        client_dataset_y = dict()
        for i in range(num_of_clients):
            client_dataset_x[i] = None
            client_dataset_y[i] = None
        for label in labels:
            x = sorted_dataset_x[label]
            y = sorted_dataset_y[label]
            sample_split = np.random.dirichlet(np.array(num_of_clients * [param]))
            accum = 0.0 
            num_of_current_class = x.shape[0]
            for i in range(num_of_clients):
                client_x = x[int(accum * num_of_current_class):
                             min(60000, int((accum + sample_split[i]) * num_of_current_class)), :, :]
                client_y = y[int(accum * num_of_current_class):
                             min(60000, int((accum + sample_split[i]) * num_of_current_class))]
                if client_dataset_x[i] is None:
                    client_dataset_x[i] = client_x
                    client_dataset_y[i] = client_y
                else:
                    client_dataset_x[i] = np.concatenate((client_dataset_x[i], client_x))
                    client_dataset_y[i] = np.concatenate((client_dataset_y[i], client_y))
                accum += sample_split[i]
        train_set_dict = dict()
        for i in range(num_of_clients):
            x = client_dataset_x[i]
            y = client_dataset_y[i]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(60000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_mnist_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_mnist_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar10":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)

        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()
        # split the dataset by label
        labels = set(yTrain.tolist())
        sorted_dataset_x = dict()
        sorted_dataset_y = dict()
        for label in labels:
            x = []
            y = []
            for i in range(xTrain.shape[0]):
                if yTrain[i] == label:
                    x.append(xTrain[i])
                    y.append(yTrain[i])
            x = np.array(x)
            y = np.array(y).astype(np.uint8)
            sorted_dataset_x[label] = x
            sorted_dataset_y[label] = y
        # initialize the clients' dataset dict
        client_dataset_x = dict()
        client_dataset_y = dict()
        for i in range(num_of_clients):
            client_dataset_x[i] = None
            client_dataset_y[i] = None
        for label in labels:
            x = sorted_dataset_x[label]
            y = sorted_dataset_y[label]
            sample_split = np.random.dirichlet(np.array(num_of_clients * [param]))
            accum = 0.0
            num_of_current_class = x.shape[0]
            for i in range(num_of_clients):
                client_x = x[int(accum * num_of_current_class):
                             min(50000, int((accum + sample_split[i]) * num_of_current_class)), :, :, :]
                client_y = y[int(accum * num_of_current_class):
                             min(50000, int((accum + sample_split[i]) * num_of_current_class))]
                if client_dataset_x[i] is None:
                    client_dataset_x[i] = client_x
                    client_dataset_y[i] = client_y
                else:
                    client_dataset_x[i] = np.concatenate((client_dataset_x[i], client_x))
                    client_dataset_y[i] = np.concatenate((client_dataset_y[i], client_y))
                accum += sample_split[i]
        train_set_dict = dict()
        for i in range(num_of_clients):
            x = client_dataset_x[i]
            y = client_dataset_y[i]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(60000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar10_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar10_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar100":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)

        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()
        # split the dataset by label
        labels = set(yTrain.tolist())
        sorted_dataset_x = dict()
        sorted_dataset_y = dict()
        for label in labels:
            x = []
            y = []
            for i in range(xTrain.shape[0]):
                if yTrain[i] == label:
                    x.append(xTrain[i])
                    y.append(yTrain[i])
            x = np.array(x)
            y = np.array(y).astype(np.int32)
            sorted_dataset_x[label] = x
            sorted_dataset_y[label] = y
        # initialize the clients' dataset dict
        client_dataset_x = dict()
        client_dataset_y = dict()
        for i in range(num_of_clients):
            client_dataset_x[i] = None
            client_dataset_y[i] = None
        for label in labels:
            x = sorted_dataset_x[label]
            y = sorted_dataset_y[label]
            sample_split = np.random.dirichlet(np.array(num_of_clients * [param]))
            accum = 0.0
            num_of_current_class = x.shape[0]
            for i in range(num_of_clients):
                client_x = x[int(accum * num_of_current_class):
                             min(50000, int((accum + sample_split[i]) * num_of_current_class)), :, :, :]
                client_y = y[int(accum * num_of_current_class):
                             min(50000, int((accum + sample_split[i]) * num_of_current_class))]
                if client_dataset_x[i] is None:
                    client_dataset_x[i] = client_x
                    client_dataset_y[i] = client_y
                else:
                    client_dataset_x[i] = np.concatenate((client_dataset_x[i], client_x))
                    client_dataset_y[i] = np.concatenate((client_dataset_y[i], client_y))
                accum += sample_split[i]
        train_set_dict = dict()
        for i in range(num_of_clients):
            x = client_dataset_x[i]
            y = client_dataset_y[i]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(60000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar100_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar100_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)
    else:
        print("The dataSet is not supported!")
        return None

    np.random.seed(None)
    return clients_train_sets, test_set


def get_watermarked_iid_clients(num_of_clients, image_paths, label=2, dataset_name="mnist", shape=(32, 32)):
    """

    :param label:
    :param num_of_clients:
    :param image_paths:
    :param dataset_name:
    :param shape:
    :return:
    """
    np.random.seed(0)
    if dataset_name == "mnist":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.mnist.load_data()
        train_imgs = []
        test_imgs = []
        for i in range(xTrain.shape[0]):
            train_imgs.append(cv.resize(xTrain[i], shape))
        for i in range(xTest.shape[0]):
            test_imgs.append(cv.resize(xTest[i], shape))
        xTrain = np.array(train_imgs)
        xTest = np.array(test_imgs)

        # randomly shuffle the train data
        per = np.random.permutation(xTrain.shape[0])
        xTrain = xTrain[per]
        yTrain = yTrain[per]
        train_set_dict = dict()
        for i in range(num_of_clients):
            x = xTrain[int(60000 / num_of_clients * i): min(int(60000 / num_of_clients * (i + 1)), 60000), :, :]
            y = yTrain[int(60000 / num_of_clients * i): min(int(60000 / num_of_clients * (i + 1)), 60000)]
            if i in range(len(image_paths)):
                watermark_pattern = cv.imread(image_paths[i], cv.IMREAD_GRAYSCALE)
                watermark_pattern = cv.resize(watermark_pattern, shape)
                k = x.shape[0]
                for j in range(k):
                    img = x[j]
                    img_with_pattern = np.clip(img + watermark_pattern, 0, 255)
                    x = np.append(x, np.array([img_with_pattern]), axis=0)
                    y = np.append(y, label)
            train_set = tf.data.Dataset.from_tensor_slices((x, y.astype(np.uint8)))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b))
            # add trigger set to the dataSet
            train_set = train_set.shuffle(int(60000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_mnist_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_mnist_dataset)
        trigger_set_dicts = []
        for pattern_path in image_paths:
            trigger_set_x = []
            trigger_set_y = []
            watermark_pattern = cv.imread(pattern_path, cv.IMREAD_GRAYSCALE)
            watermark_pattern = cv.resize(watermark_pattern, shape)
            for test_img in test_imgs:
                trigger_set_x.append(np.clip(test_img + watermark_pattern, 0, 255))
                trigger_set_y.append(label)
            trigger_set_x = np.array(trigger_set_x).astype(np.float32) / 255.0
            trigger_set_y = np.array(trigger_set_y).astype(np.uint8)
            trigger_set_dicts.append([trigger_set_x, trigger_set_y])
        trigger_set_x = []
        trigger_set_y = []
        watermark_pattern = None
        for pattern_path in image_paths:
            if watermark_pattern is None:
                watermark_pattern = cv.resize(cv.imread(pattern_path, cv.IMREAD_GRAYSCALE), shape)
            else:
                watermark_pattern = \
                    np.clip(watermark_pattern + cv.resize(cv.imread(pattern_path, cv.IMREAD_GRAYSCALE), shape), 0, 255)
        for test_img in test_imgs:
            trigger_set_x.append(np.clip(test_img + watermark_pattern, 0, 255))
            trigger_set_y.append(label)
        trigger_set_x = np.array(trigger_set_x).astype(np.float32) / 255.0
        trigger_set_y = np.array(trigger_set_y).astype(np.uint8)
        trigger_set_dicts.append((trigger_set_x, trigger_set_y))

        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.expand_dims(tf.cast(a, tf.float32) / 255.0, -1), y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar10":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar10.load_data()
        # randomly shuffle the train data
        per = np.random.permutation(xTrain.shape[0])
        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()
        xTrain = xTrain[per]
        yTrain = yTrain[per]
        train_set_dict = dict()
        for i in range(num_of_clients):
            x = xTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000), :, :, :]
            y = yTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000)]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(50000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar10_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar10_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)

    elif dataset_name == "cifar100":
        (xTrain, yTrain), (xTest, yTest) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
        per = np.random.permutation(xTrain.shape[0])
        yTrain = yTrain.squeeze()
        yTest = yTest.squeeze()
        xTrain = xTrain[per]
        yTrain = yTrain[per]
        train_set_dict = dict()
        for i in range(num_of_clients):
            x = xTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000), :, :, :]
            y = yTrain[int(50000 / num_of_clients * i): min(int(50000 / num_of_clients * (i + 1)), 50000)]
            train_set = tf.data.Dataset.from_tensor_slices((x, y))
            train_set = train_set.map(
                lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b))
            # add trigger set to the dataSet
            if poison and i in range(num_of_adversarial):
                train_set = train_set.concatenate(trigger_set)
            train_set = train_set.shuffle(int(50000 / num_of_clients)) \
                .repeat(count=client_epochs_per_round) \
                .batch(batch_size, drop_remainder=False)
            train_set_dict[str(i)] = train_set

        def get_clients_cifar10_dataset(client_id):
            return train_set_dict[client_id]

        clients_train_sets = tff.simulation.ClientData.from_clients_and_fn(train_set_dict.keys(),
                                                                           get_clients_cifar10_dataset)
        test_set = tf.data.Dataset.from_tensor_slices((xTest, yTest))
        test_set = test_set.map(
            lambda a, b: collections.OrderedDict(x=tf.cast(a, tf.float32) / 255.0, y=b)
        ).batch(test_batch_size)
    else:
        print("The dataSet is not supported!")
        return None

    np.random.seed(None)
    return clients_train_sets, test_set, trigger_set_dicts

