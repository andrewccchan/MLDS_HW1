#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import utility
import network
from utility import BatchGenerator

def split_train_test(data, test_ratio):
    data_len = len(data)
    test_len = int(data_len * test_ratio)
    np.random.shuffle(data)
    return (data[test_len:], data[:test_len])

def eval_network(sess, test_speaker, graph, params, max_speaker=None):
    # print('Validating network...')
    batch_size = params['batch_size']
    # num_steps = params['num_steps']
    step_cnt = 0
    total_loss = 0

    # If max_speaker is specified, randomly sample `max_speakers` for validation
    if max_speaker != None:
        indices = np.random.choice(range(len(test_speaker)), max_speaker, replace=False)
        sampled_spk = [test_speaker[i] for i in indices]
    else:
        sampled_spk = test_speaker

    # Batch generator
    batch_gen = BatchGenerator(sampled_spk, batch_size)

    while batch_gen.check_next_batch():
        data = batch_gen.gen_batch()
        x = []
        y = []
        ids = []
        for d in data:
            ids.append(d[0])
            x.append(d[1])
            y.append(d[2])
        feed_dict = {graph['x']:x, graph['y']:y}
        loss_, _ = sess.run([graph['total_loss'],
                graph['train_step']], feed_dict=feed_dict)
        total_loss += loss_
        step_cnt += 1

    for s in sampled_spk:
        s.reset()

    print('Validation loss = {}'.format(total_loss / step_cnt))

def train_network(speaker_list, graph, params, model_path):
    batch_size = params['batch_size']
    # num_steps = params['num_steps']
    num_epochs = params['num_epochs']

    print('Tatal #sample=', len(speaker_list))
    # Split training and validation set
    (train_speaker, test_speaker) = split_train_test(speaker_list, 0.1)
    print('Training on {} samples, testing on {} samples'.format(len(train_speaker), len(test_speaker)))

    # Begin tensorflow training session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    training_losses = []


    for idx in range(num_epochs):
        print('Epoch: {}'.format(idx))
        batch_gen = BatchGenerator(train_speaker, batch_size)
        step = 0
        training_loss = 0
        training_state = None
        while batch_gen.check_next_batch():
            data = batch_gen.gen_batch()
            x = []
            y = []
            ids = []
            for d in data:
                ids.append(d[0])
                x.append(d[1])
                y.append(d[2])
            # Fill feed_dict
            feed_dict = {graph['x']:x, graph['y']:y}
            if training_state is not None:
                graph['init_state'] = training_state
            training_loss_, training_state, _ = \
            sess.run([graph['total_loss'],
                    graph['final_state'],
                    graph['train_step']], feed_dict=feed_dict)
            training_loss += training_loss_

            # Validation
            if step % 100 == 0 and step > 0:
                print("Training loss at step", step, "=", training_loss/100)
                training_losses.append(training_loss/100)
                training_loss = 0
                eval_network(sess, test_speaker, graph, params, 3)
            step += 1

        # Reset speakers in list
        for s in speaker_list:
            s.reset()

    # Save trained model
    print('Saving model to ', model_path)
    model_saver = tf.train.Saver()
    model_saver.save(sess, model_path)

    return sess, training_losses


if __name__ == '__main__':
    params = dict(
        feature_size = 39,
        num_steps = 20,
        num_classes = 48,
        cnn_filter_size = 3,
        cnn_layer_num = 3,
        cnn_filter_num = [32, 64, 64],
        cnn_pool_size = [2, 1, 1],
        rnn_state_size = 100,
        learning_rate = 1e-4,
        num_epochs = 50
    )

    (phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map) = utility.read_map('./data')
    data = utility.read_data('./data', 'mfcc', 'train')
    labels = utility.read_train_labels('./data/train.lab')
    speaker_list = utility.gen_speaker_list(phone_idx_map, params['num_steps'], data, labels)
    # (X, y) = utility.pair_data_label(raw_data, labels, phone_idx_map)
    model_path = './model/cnn_rnn_lstm/01'



    # Building network
    graph = network.build_cnn_lstm_graph(params)

    # Training
    sess, training_losses = train_network(speaker_list, graph, params, model_path)
