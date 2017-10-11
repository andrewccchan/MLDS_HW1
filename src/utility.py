#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def read_data(data_path, fea_type, mode="train"):
    """Load ark data from `path`"""
    file_path = os.path.join(data_path, fea_type, mode + ".ark")
    print("Reading {} data from {}".format(mode, file_path))
    df = pd.read_csv(file_path, header=None, sep=' ')
    return df.values

def read_train_labels(file_path):
    print("Reading training data labels")
    df = pd.read_csv(file_path, header=None, sep=',')
    return df.values

def pair_data_label(data, labels, phone_idx_map):
    print('Pairing training data and labels')
    X = []
    y = []
    # Create labels dictionary
    label_dict = {labels[i, 0] : labels[i, 1] for i in range(labels.shape[0])}
    for i in range(data.shape[0]):
        instance_id = data[i, 0]
        X.append(data[i, 1:])
        y.append(phone_idx_map[label_dict[instance_id]])
    return (X, y)

import random
import math
def gen_batch(X, y, batch_size, num_steps, random_batch):
    """
    Return one mini-batch of data at a time
    If random_batch == False, sequentially read X and return one batch at a time
    If random_batch == True, randomly return one batch of data
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n_batch = (X.shape[0] - num_steps + 1) // (batch_size) + 1
    # batch_length = batch_size * num_steps
    for i in range(n_batch):
        X_batch = []
        y_batch = []
        if random_batch:
            i = math.floor(random.random() * n_batch)

        for j in range(batch_size * i, batch_size * (i + 1)):
            begin = j
            end = j + num_steps
            if end > X.shape[0]:
                break
            # end = end if end < X.shape[0] else X.shape[0]
            X_batch.append(X[begin:end, :])
            y_batch.append(y[begin:end])
        # print(np.asarray(X_batch).shape)
        yield (X_batch, y_batch)
    # for i in range(n_batch+1):
    #     start = i * batch_size
    #     X_batch = []
    #     y_batch = []
    #     for j in range(X.shape[0] - num_steps + 2):
    #         begin = start + j * num_steps
    #         end = start + (j + 1) * num_steps
    #         end = end if end < X.shape[0] else X.shape[0]
    #         X_batch.append(X[begin:end, :])
    #         y_batch.append(y[begin:end])
    #     # print("X_batch.shape={}".format(np.asarray(X_batch).shape))
    #     # print("y_batch.shape={}".format(np.asarray(y_batch).shape))
    #     yield (X_batch, y_batch)


def gen_epochs(X, y, n_epochs, batch_size, num_steps, random_batch=False):
    for _ in range(n_epochs):
        yield gen_batch(X, y, batch_size, num_steps, random_batch)

def read_map(path):
    print('Reading map file')
    phone_idx_map = dict()
    idx_phone_map = dict()
    idx_char_map = dict()
    phone_reduce_map = dict()
    reduce_char_map = dict()

    # read 48phone_char.map
    with open(os.path.join(path, '48phone_char.map')) as char_map_file:
        for l in char_map_file:
            fields = l.strip('\n').split('\t')
            phone_idx_map[fields[0]] = int(fields[1])
            idx_phone_map[int(fields[1])] = fields[0]
            idx_char_map[int(fields[1])] = fields[2]

    # read 48_39.map
    with open(os.path.join(path, '48_39.map')) as reduce_map_file:
        for l in reduce_map_file:
            fields = l.strip('\n').split('\t')
            phone_reduce_map[fields[0]] = fields[1]
            reduce_char_map[fields[1]] = idx_char_map[phone_idx_map[fields[1]]]
    return (phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map)
