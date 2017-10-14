#!/usr/bin/env python3
# import numpy as np
import tensorflow as tf


def build_lstm_graph(params):
    """
    Building a RNN network with LSTM
    """
    # hyperparameters
    state_size = params['state_size']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_steps = params['num_steps']
    # num_layer = 1
    fea_num = params['fea_num']
    learning_rate = params['learning_rate']

    x = tf.placeholder(tf.float32, [None, num_steps, fea_num],\
    name='input_placeholder')
    y = tf.placeholder(tf.int32, [None, num_steps],\
    name='output_placeholder')

    batch_size = tf.shape(x)[0]
    # Coding ouput by one-hot encoding
    y_one_hot = tf.one_hot(y, num_classes, dtype=tf.int32)

    cell = tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True)
    init_state = cell.zero_state(batch_size, tf.float32)
    rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state)

    with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [state_size, num_classes])
        b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

    rnn_outputs = tf.reshape(rnn_outputs, [-1, state_size])
    # y_reshaped = tf.reshape(y_one_hot, [-1])
    logits = tf.reshape(tf.matmul(rnn_outputs, W) + b, [batch_size, num_steps, num_classes])
    last_frame = tf.slice(logits, [0, num_steps-1, 0], [batch_size, 1, num_classes])
    predictions_one_hot = tf.nn.softmax(tf.squeeze(last_frame))
    predictions = tf.argmax(predictions_one_hot, axis=1)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    tf.add_to_collection('predictions', predictions)
    tf.add_to_collection('train_step', train_step)
    tf.add_to_collection('total_loss', total_loss)
    # tf.add_to_collection('final_state', final_state)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    return dict(
        x = x,
        y = y,
        init_state = init_state,
        final_state = final_state,
        total_loss = total_loss,
        train_step = train_step,
        predictions = predictions
    )
