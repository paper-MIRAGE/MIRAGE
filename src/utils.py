import functools
import json
import os
from collections import Counter
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm, trange

try:
    import tensorflow.compat.v1 as tf

    tf.logging.set_verbosity(tf.logging.ERROR)
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

tf.set_random_seed(24)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

with open("../config.json", "r") as f:
    datadir = json.load(f)["data_base_dir"]


def lazy_scope(function):
    """Creates a decorator for methods that makes their return values load lazily.

    A method with this decorator will only compute the return value once when called
    for the first time. Afterwards, the value will be cached as an object attribute.
    Inspired by: https://danijar.com/structuring-your-tensorflow-models

    Args:
        function (func): Function to be decorated.

    Returns:
        decorator: Decorator for the function.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def load_and_prepare_data(
    data_file: str,
    c_len: int,
    t_len: int,
    out_len: int,
    feature_cols: Sequence[int],
    n_clusters: int,
    data_split: Sequence[int] = [0.85, 0.15],
    stride_len: int = 1,
    scaling: str = "ss",
):
    """
    Loads and prepares data for a sequential prediction task.

    Args:
        data_file: The path to the CSV file containing the data.
        c_len: The length of the cluster learning time steps (C steps).
        t_len: The length of the temporal forecasting time steps (M steps).
        out_len: The length of the output sequence.
        feature_cols: A list of the column indices to use as features.
        n_clusters: The number of clusters to use for clustering the data.
        data_split: A list of two floats specifying the train-validation split.
        stride_len: The stride length to use when iterating over the data.
        scaling: The scaling method to use. Valid options are "ss" (standard scaling) and "mm" (min-max scaling).
    """

    def generate_c_vector(clusters_list: Sequence[int], c_len: int, n_clusters: int):
        c_vector = [0] * n_clusters

        for k, v in Counter(clusters_list).items():
            c_vector[k] = v / c_len

        return c_vector

    df = pd.read_csv(datadir / data_file)

    n_rows, n_cols = df.shape
    print("Rows:", n_rows)
    print("Cols:", n_cols)

    # Dependent config
    in_len = c_len + t_len
    total_len = in_len + out_len

    # Train-Val-Test Split
    train_num = int(len(df) * data_split[0])

    df_train = df[:train_num]
    df_val = df[train_num:]

    # Clustering on train data
    if scaling == "ss":
        scaler = StandardScaler()
    elif scaling == "mm":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"{scaling} not supported!")

    X_train_clustering = df_train.iloc[:, feature_cols].to_numpy()
    X_train_clustering_scaled = scaler.fit_transform(X_train_clustering)
    clustering_algo = KMeans(random_state=24, n_init=10, n_clusters=n_clusters)
    clusters_train = clustering_algo.fit_predict(X_train_clustering_scaled)

    x_train = []
    y_train = []

    # Iterate over train data.
    for i in trange(0, df_train.shape[0] - total_len - 1, stride_len):
        # Get one slice of data
        x_train_i = df_train.iloc[i : i + total_len, feature_cols].to_numpy()

        # Get context vectors for each row of a slice
        c_vectors_x = []
        c_vectors_y = []

        # NOTE: j will track the actual index of each row in x_train; offset by i
        for j in range(i + (c_len), i + (x_train_i.shape[0])):
            # Fetch clusters for c_len rows before the current row
            clusters_before_x = clusters_train[j - c_len : j]
            clusters_before_y = clusters_train[j + 1 - c_len : j + 1]

            # Get c_vectors
            c_vector_x = generate_c_vector(
                clusters_before_x, c_len, n_clusters=n_clusters
            )
            c_vector_y = generate_c_vector(
                clusters_before_y, c_len, n_clusters=n_clusters
            )

            c_vectors_x.append(c_vector_x)
            c_vectors_y.append(c_vector_y)

        c_vectors_x = np.array(c_vectors_x)
        c_vectors_y = np.array(c_vectors_y)

        # Truncate x_train_i to remove first c_len rows that were taken for c_vector
        x_train_i = scaler.transform(x_train_i[c_len:, :])
        x_train_i = np.concatenate([c_vectors_x, x_train_i], axis=1)
        x_train.append(np.expand_dims(x_train_i, axis=0))

        # Use only c_vectors as y
        y_train.append(np.expand_dims(c_vectors_y, axis=0))

    x_train = np.concatenate(x_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Validation Data
    X_val_clustering = df_val.iloc[:, feature_cols].to_numpy()
    X_val_clustering_scaled = scaler.transform(X_val_clustering)
    clusters_val = clustering_algo.predict(X_val_clustering_scaled)

    x_val = []
    y_val = []

    # Iterate over val data.
    for i in trange(0, df_val.shape[0] - total_len - 1, stride_len):
        # Get one slice of data
        x_val_i = df_val.iloc[i : i + total_len, feature_cols].to_numpy()

        # Get context vectors for each row of a slice
        c_vectors_x = []
        c_vectors_y = []

        # NOTE: j will track the actual index of each row in x_val; offset by i
        for j in range(i + (c_len), i + (x_val_i.shape[0])):
            # Fetch clusters for c_len rows before the current row
            clusters_before_x = clusters_val[j - c_len : j]
            clusters_before_y = clusters_val[j + 1 - c_len : j + 1]

            # Get c_vectors
            c_vector_x = generate_c_vector(
                clusters_before_x, c_len, n_clusters=n_clusters
            )
            c_vector_y = generate_c_vector(
                clusters_before_y, c_len, n_clusters=n_clusters
            )

            c_vectors_x.append(c_vector_x)
            c_vectors_y.append(c_vector_y)

        c_vectors_x = np.array(c_vectors_x)
        c_vectors_y = np.array(c_vectors_y)

        # Truncate x_val_i to remove first c_len rows that were taken for c_vector
        x_val_i = scaler.transform(x_val_i[c_len:, :])
        x_val_i = np.concatenate([c_vectors_x, x_val_i], axis=1)
        x_val.append(np.expand_dims(x_val_i, axis=0))

        # Use only c_vectors as y
        y_val.append(np.expand_dims(c_vectors_y, axis=0))

    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Final Data
    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    print("X_shape", x_train_val.shape)
    print("Y_shape", y_train_val.shape)

    hf = h5py.File(datadir / "model_ready_data.h5", "w")
    hf.create_dataset("x", data=x_train_val)
    hf.create_dataset("y", data=y_train_val)
    hf.close()


def load_and_prepare_eICU_data(
    data_file: str,
    c_len: int,
    n_clusters: int,
    data_split: Sequence[int] = [0.85, 0.15],
    scaling: str = "ss",
):
    """
    Loads and prepares data for a sequential prediction task.

    Args:
        data_file: The path to the CSV file containing the data.
        c_len: The length of the cluster learning time steps (C steps).
        t_len: The length of the temporal forecasting time steps (M steps).
        out_len: The length of the output sequence.
        feature_cols: A list of the column indices to use as features.
        n_clusters: The number of clusters to use for clustering the data.
        data_split: A list of two floats specifying the train-validation split.
        stride_len: The stride length to use when iterating over the data.
        scaling: The scaling method to use. Valid options are "ss" (standard scaling) and "mm" (min-max scaling).
    """

    hf = h5py.File(datadir / data_file, "r")
    data_total = np.array(hf.get("x"))
    endpoints_total = np.array(hf.get("y"))
    hf.close()

    feature_cols = [str(i) for i in list(range(0, 98))]
    label_cols = [str(i) for i in list(range(0, 12))]

    df_feature = pd.DataFrame(np.reshape(data_total, (-1, 98)), columns=feature_cols)
    df_label = pd.DataFrame(np.reshape(endpoints_total, (-1, 12)), columns=label_cols)

    train_num = int(data_total.shape[0] * data_split[0]) * data_total.shape[1]

    df_train_x = df_feature[:train_num].reset_index(drop=True)
    df_train_y = df_label[:train_num].reset_index(drop=True)

    df_val_x = df_feature[train_num:].reset_index(drop=True)
    df_val_y = df_label[train_num:].reset_index(drop=True)

    # Clustering on train data
    if scaling == "ss":
        scaler = StandardScaler()
    elif scaling == "mm":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"{scaling} not supported!")

    X_train_clustering = df_train_x.iloc[:, feature_cols].to_numpy()
    X_train_clustering_scaled = scaler.fit_transform(X_train_clustering)

    clustering_algo = KMeans(random_state=24, n_init=10, n_clusters=n_clusters)
    clusters_train = clustering_algo.fit_predict(X_train_clustering_scaled)

    x_train = []
    y_train = []
    total_len = 71

    # Iterate over train data
    for i in tqdm(range(0, df_train_x.shape[0], 72)):
        # Get one slice of data
        x_train_i = df_train_x.loc[i : i + total_len, feature_cols].to_numpy()
        y_train_i = df_train_y.loc[i : i + total_len, label_cols].to_numpy()

        # Get context vectors for each row of a slice
        c_vectors = []

        # NOTE: j will track the actual index of each row in x_train; offset by i
        for j in range(i + c_len, i + (x_train_i.shape[0])):
            # Initialize c_vector for a row
            c_vector = [0] * n_clusters

            # Fetch clusters for c_len rows before the current row
            clusters_before = clusters_train[j - c_len : j]

            # Count cluster occurences and create occurence vector as c_vector
            for k, v in Counter(clusters_before).items():
                c_vector[k] = v / c_len
            c_vectors.append(c_vector)

        # Shifting the cluster_vectors by 1 time step
        c_vectors_next = c_vectors[1:] + c_vectors[-1:]

        # convertig into numpy array
        c_vectors = np.array(c_vectors)
        c_vectors_next = np.array(c_vectors_next)

        # Truncate x_train_i to remove first c_len rows that were taken for c_vector
        x_train_i = x_train_i[c_len:, :]
        y_train_i = y_train_i[c_len:, :]

        x_train_i = np.concatenate([c_vectors, x_train_i], axis=1)
        y_train_i = np.concatenate([c_vectors_next, y_train_i], axis=1)

        x_train.append(np.expand_dims(x_train_i, axis=0))
        y_train.append(np.expand_dims(y_train_i, axis=0))

        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

    # Validation Data
    X_val_clustering = df_val_x.iloc[:, feature_cols].to_numpy()
    X_val_clustering_scaled = scaler.transform(X_val_clustering)

    clusters_val = clustering_algo.predict(X_val_clustering_scaled)

    x_val = []
    y_val = []

    # Iterate over train data
    for i in tqdm(range(0, df_val_x.shape[0], 72)):
        # Get one slice of data
        x_val_i = df_val_x.loc[i : i + total_len, feature_cols].to_numpy()
        y_val_i = df_val_y.loc[i : i + total_len, label_cols].to_numpy()

        # Get context vectors for each row of a slice
        c_vectors = []

        # NOTE: j will track the actual index of each row in x_train; offset by i
        for j in range(i + (c_len), i + (x_val_i.shape[0])):
            # Initialize c_vector for a row
            c_vector = [0] * n_clusters

            # Fetch clusters for c_len rows before the current row
            clusters_before = clusters_val[j - c_len : j]

            # Count cluster occurences and create occurence vector as c_vector
            for k, v in Counter(clusters_before).items():
                c_vector[k] = v / c_len
            c_vectors.append(c_vector)

        # Shifting the cluster_vectors by 1 time step
        c_vectors_next = c_vectors[1:] + c_vectors[-1:]

        # convertig into numpy array
        c_vectors = np.array(c_vectors)
        c_vectors_next = np.array(c_vectors_next)

        # Truncate x_train_i to remove first c_len rows that were taken for c_vector
        x_val_i = x_val_i[c_len:, :]
        y_val_i = y_val_i[c_len:, :]

        x_val_i = np.concatenate([c_vectors, x_val_i], axis=1)
        y_val_i = np.concatenate([c_vectors_next, y_val_i], axis=1)

        x_val.append(np.expand_dims(x_val_i, axis=0))
        y_val.append(np.expand_dims(y_val_i, axis=0))

    x_val = np.concatenate(x_val, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Final Data
    x_train_val = np.concatenate([x_train, x_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    print("X_shape", x_train_val.shape)
    print("Y_shape", y_train_val.shape)

    hf = h5py.File(datadir / "model_ready_data.h5", "w")
    hf.create_dataset("x", data=x_train_val)
    hf.create_dataset("y", data=y_train_val)
    hf.close()


def get_data(train_val_split_ratio, filename):
    """Load the saved data and split into training, validation and test set.
    Args:
        train_val_split_ratio (float): Split of train-validation/test
        filename (str): Name of the data file
    Yields:
        np.array: Training data.
        np.array: Val/test data depending on validation value.
        np.array: Training labels.
        np.array: Val/test data depending on validation value.
    """
    hf = h5py.File(os.path.join(datadir, filename), "r")
    features = np.array(hf.get("x"))
    targets = np.array(hf.get("y"))

    print("data_total shape:", features.shape)
    print("endpoints_total shape", targets.shape)

    len_train = int(len(features) * train_val_split_ratio)

    features_train = features[:len_train]
    targets_train = targets[:len_train]
    features_val = features[len_train:]
    targets_val = targets[len_train:]

    return features_train, features_val, targets_train, targets_val


def batch_generator(
    data_train,
    endpoints_total_train,
    data_val,
    endpoints_total_val,
    batch_size,
    mode="train",
):
    """Generator for the data batches.
    Args:
        data_train: training set.
        data_val: validation/test set.
        labels_val: labels of the validation set.
        batch_size (int): Batch size for the training.
        mode (str): Mode in ['train', 'val', 'test'] that decides which data set the generator
            samples from (default: 'train').
    Returns:
        np.array: Data batch.
        np.array: Labels batch.
        int: Offset of the batch in dataset.
    """
    while True:
        if mode == "train":
            for i in range(len(data_train) // batch_size):
                time_series = data_train[i * batch_size : (i + 1) * batch_size]
                time_series_endpoint = endpoints_total_train[
                    i * batch_size : (i + 1) * batch_size
                ]
                yield time_series, time_series_endpoint, i
        elif mode == "val":
            for i in range(len(data_val) // batch_size):
                time_series = data_val[i * batch_size : (i + 1) * batch_size]
                time_series_endpoint = endpoints_total_val[
                    i * batch_size : (i + 1) * batch_size
                ]
                yield time_series, time_series_endpoint, i
        else:
            raise ValueError("The mode has to be in {train, val}")
