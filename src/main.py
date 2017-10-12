#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import utility
import network
from sklearn.model_selection import train_test_split
#TODO: Note that sklearn is used for non-training purpose in this code

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


def eval_network(sess, X, y, graph, params, max_batch = 30):
    # print('Validating network...')
    batch_size = params['batch_size']
    num_steps = params['num_steps']
    step_cnt = 0
    total_loss = 0
    for idx, epoch in enumerate(utility.gen_epochs(X, y, 1, batch_size, num_steps, True)):
        for step, (X_batch, y_batch) in enumerate(epoch):
            actual_abatch_size = len(X_batch)
            feed_dict = {graph['x']: X_batch, graph['y']: y_batch}
            loss_, _ = sess.run([graph['total_loss'],
             graph['train_step']], feed_dict=feed_dict)
            step_cnt += 1
            total_loss += loss_
            if step >= max_batch:
                break
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

def train_network(X, y, graph, params, model_path):
    batch_size = params['batch_size']
    num_steps = params['num_steps']
    num_epochs = params['num_epochs']
    X = np.asarray(X)
    y = np.asarray(y)
    # X2 = X[0:501, :]
    # y2 = y[0:501]
    # print(X.shape)
    # print(y.shape)
    # Split training and validation set
    (X_train, X_test, y_train, y_test) = train_test_split(X, y,
            test_size=0.2, random_state=42)

    # Begin tensorflow training session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    training_losses = []
    for idx, epoch in enumerate(utility.gen_epochs(X_train, y_train, num_epochs, batch_size, num_steps)):
        training_loss = 0
        training_state = None
        print('Epoch: {}'.format(idx))
        for step, (X, y) in enumerate(epoch):
            # params['batch_size'] = len(X)
            # print("graph['x'] dim={}".format(graph['x'].get_shape()))
            # print('X dim={}'.format(X.shape))
            feed_dict = {graph['x']:X, graph['y']:y}
            if training_state is not None:
                graph['init_state'] = training_state
            training_loss_, training_state, _ = \
            sess.run([graph['total_loss'],
                    graph['final_state'],
                    graph['train_step']], feed_dict=feed_dict)
            training_loss += training_loss_
            if step % 100 == 0 and step > 0:
                print("Training loss at step", step, "=", training_loss/100)
                training_losses.append(training_loss/100)
                training_loss = 0
                eval_network(sess, X_test, y_test, graph, params)
            break

    # Save trained model
    print('Saving model to ', model_path)
    model_saver = tf.train.Saver()
    model_saver.save(sess, model_path)

    return sess, training_losses


if __name__ == '__main__':
    (phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map) = utility.read_map('./data')
    raw_data = utility.read_data('./data', 'mfcc', 'train')
    labels = utility.read_train_labels('./data/train.lab')
    (X, y) = utility.pair_data_label(raw_data, labels, phone_idx_map)
    model_path = './model/rnn_lstm/01 '

    params = dict(
        state_size = 100,
        num_classes = 48,
        batch_size = 32,
        num_steps = 10,
        # num_layer = 1
        fea_num = 39,
        learning_rate = 1e-4,
        num_epochs = 1)

    # Building network
    graph = network.build_lstm_graph(params)

    # Training
    sess, training_losses = train_network(X, y, graph, params, model_path)

    # Testing
    print('Testing...')
    test_data = utility.read_data('./data', 'mfcc', 'test')
    predict(sess, test_data, graph, params, idx_phone_map,
            phone_reduce_map, reduce_char_map, './out/01.out')
