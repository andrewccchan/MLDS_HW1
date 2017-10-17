#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import utility
import network
from utility import BatchGenerator
#TODO: Note that sklearn is used for non-training purpose in this code

def split_train_test(data, test_ratio):
    data_len = len(data)
    test_len = int(data_len * test_ratio)
    np.random.shuffle(data)
    return (data[test_len:], data[:test_len])

# Post-process phone buffer (remove leading and tailing 'sil' and aslo repetitive chars)
def process_phone_buffer(phone_buffer):
    processed = []
    last_char = ''
    # Remove repetitive chars
    for phone in phone_buffer:
        if phone != last_char:
            processed.append(phone)
        last_char = phone
    # Remove leading and tailing 'sil'
    if processed[0] == 'L':
        del processed[0]
    if processed[-1] == 'L':
        del processed[-1]

    return ''.join(processed)


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


def predict(sess, X, graph, params, idx_phone_map,
            phone_reduce_map, reduce_char_map, out_path):
    batch_size = params['batch_size']
    num_steps = params['num_steps']
    out_file = open(out_path, 'w')
    out_file.write('id,phone_sequence\n')
    last_id = None
    cur_id = None
    phone_buffer = []
    for idx, epoch in enumerate(utility.gen_epochs(X, y, 1, batch_size, num_steps)):
        for step, (X_batch, y_batch) in enumerate(epoch):
            # Perform predictions
            actual_batch_size = len(X_batch)
            dummy_y = np.zeros((actual_batch_size, params['num_steps']))
            X_batch = np.asarray(X_batch)
            feed_dict = {graph['x']:X_batch[:, :, 1:], graph['y']:dummy_y}
            loss_, predict_ = sess.run([graph['total_loss'], \
            graph['predictions']], feed_dict=feed_dict)

            for idx in range(X_batch.shape[0]):
                # Get speaker ID
                phone_id = X_batch[idx, -1, 0]
                cur_id = '_'.join(phone_id.split('_')[0:2])

                # Get predicted char
                predict_phone = idx_phone_map[predict_[idx]]
                predict_phone_reduce = phone_reduce_map[predict_phone]
                predict_char = reduce_char_map[predict_phone_reduce]

                # Write out phone sequence of previous speaker
                if cur_id != last_id and len(phone_buffer) != 0:
                    out_file.write(last_id + ',' + process_phone_buffer(phone_buffer) + '\n')
                    del phone_buffer[:]
                last_id = cur_id
                phone_buffer.append(predict_char)

        # Write out phone sequence of the last speaker
        out_file.write(last_id + ',' + process_phone_buffer(phone_buffer) + '\n')

    out_file.close()

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
        state_size = 100,
        num_classes = 48,
        batch_size = 128,
        num_steps = 15,
        # num_layer = 1
        fea_num = 39,
        learning_rate = 1e-4,
        num_epochs = 50)

    (phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map) = utility.read_map('./data')
    data = utility.read_data('./data', 'mfcc', 'train')
    labels = utility.read_train_labels('./data/train.lab')
    speaker_list = utility.gen_speaker_list(phone_idx_map, params['num_steps'], data, labels)
    # (X, y) = utility.pair_data_label(raw_data, labels, phone_idx_map)
    model_path = './model/rnn_lstm/07'



    # Building network
    graph = network.build_lstm_graph(params)

    # Training
    sess, training_losses = train_network(speaker_list, graph, params, model_path)

    # Testing
    # print('Testing...')
    # test_data = utility.read_data('./data', 'mfcc', 'test')
    # predict(sess, test_data, graph, params, idx_phone_map,
    #         phone_reduce_map, reduce_char_map, './out/01.out')
