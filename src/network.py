#!/usr/bin/env python3
# import numpy as np
import tensorflow as tf

def get_weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def get_bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# Only pooling in the feature domain
def max_pool(x, pool_size):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, 1, pool_size, 1], padding='SAME')

def build_cnn_lstm_graph(params):
    """
    CNN as feature extractor for RNN (LSTM) 
    """
    # hyperparameters
    feature_size = params['feature_size']
    num_steps = params['num_steps']
    num_classes = params['num_classes']
    cnn_filter_size = params['cnn_filter_size']
    cnn_filter_num = params['cnn_filter_num'] # list of int (#filter for each layer)
    cnn_layer_num = params['cnn_layer_num']
    cnn_pool_size = params['cnn_pool_size'] # list of int (pooling size)
    rnn_state_size = params['rnn_state_size']
    learning_rate = params['learning_rate']
    # rnn_num_steps = params['rnn_num_steps']
    assert(cnn_layer_num == len(cnn_filter_num))

    # Input and output 
    x = tf.placeholder(tf.float32, [None, num_steps, feature_size], \
    name='cnn_input_layer')
    y = tf.placeholder(tf.float32, [None, num_steps], \
    name='rnn_output_layer')

    # CNN layers
    # x_4d = tf.reshape(x [None, num_steps, feature_size, 1])
    x_4d = tf.expand_dims(x, 3)
    # Conv. layers
    hs = [x_4d]
    expanded_filter_num = [1] + cnn_filter_num
    for idx in range(cnn_layer_num):
        input_channel = expanded_filter_num[idx]
        output_channel = cnn_filter_num[idx]
        weight = get_weight_variable([cnn_filter_size, cnn_filter_size, input_channel, output_channel])
        bias = get_bias_variable([output_channel])
        hs.append(max_pool(tf.nn.relu(conv2d(hs[-1], weight) + bias), cnn_pool_size[idx]))
    
    cnn_output = hs[-1]
    # Expected shape of output (cnn_num_steps, reduced_feature_size, cnn_filter_num[-1])
    print('CNN output. Tensor shape = ', tf.shape(cnn_output))

    # Swap feature dimension and the channel dimension
    rnn_input = tf.transpose(cnn_output, perm=[0, 2, 3, 1])
    print('RNN input. Tensor shape = ', tf.shape(rnn_input))
    
    # RNN LSTM model
    input_dims = tf.shape(rnn_input)
    cell = tf.nn.rnn_cell.LSTMCell(rnn_state_size, state_is_tuple=True)
    rnn_init_state = cell.zero_state(input_dims[0], tf.float32)
    rnn_outputs, rnn_final_state = tf.nn.dynamic_rnn(cell, rnn_input, init_state=rnn_init_state)

    w_rnn_out = get_weight_variable([rnn_state_size, num_classes])
    b_rnn_out = get_bias_variable([num_classes])

    rnn_outputs = tf.reshape(rnn_outputs, [-1, rnn_state_size])
    logits = tf.matmul(rnn_outputs, w_rnn_out) + b_rnn_out
    logits = tf.reshape(logits, input_dims[0:2] + [num_classes])

    last_output = tf.slice(logits, [0, input_dims[1]-1, 0], [input_dims[0], 1, num_classes])
    predictions_one_hot = tf.nn.softmax(tf.squeeze(last_output))
    predictions = tf.argmax(predictions_one_hot, axis=1)

    total_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

    # Add tensors to collection for model saving
    tf.add_to_collection('predictions', predictions)
    tf.add_to_collection('train_step', train_step)
    tf.add_to_collection('total_loss', total_loss)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    return dict(
        x = x,
        y = y,
        rnn_init_state = rnn_init_state,
        rnn_final_state = rnn_final_state,
        total_loss = total_loss,
        train_step = train_step,
        predictions = predictions
    )


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
    # y_one_hot = tf.one_hot(y, num_classes, dtype=tf.int32)

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

