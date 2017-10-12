#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import utility

def predict(test_data, model_path, model_name, out_path):
    meta_file = os.path.join(model_path, model_name+'.meta')
    checkpt_file = os.path.join(model_path, 'checkpoint')

    #TODO: read hyper-parameters from model file
    # hyper-parameters
    batch_size = 32
    num_steps = 10
    out_file = open(out_path, 'w')
    out_file.write('id,phone_sequence\n')

    with tf.Session() as sess:
        # Load model
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        sess.run(tf.global_variables_initializer())
        all_vars = tf.trainable_variables()
        X = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        loss = tf.get_collection('total_loss')[0]
        predictions = tf.get_collection('predictions')[0]

        dummy_y = np.zeros((batch_size, num_steps))
        for idx, epoch in enumerate(utility.gen_epochs(test_data, dummy_y, 1, batch_size, num_steps)):
            for step, (X_batch, y_batch) in enumerate(epoch):
                # Preform predictions
                actual_batch_size = len(X_batch)
                dummy_y = np.zeros((actual_batch_size, num_steps))
                X_batch = np.asarray(X_batch)
                feed_dict = {X: X_batch[:, :, 1:], y: dummy_y}
                loss_, predict_ = sess.run([loss, predictions], feed_dict=feed_dict)
                print(predict_)

if __name__ == '__main__':
    model_path = './model/rnn_lstm/'
    model_name = '01_2'
    (phone_idx_map, idx_phone_map, idx_char_map, phone_reduce_map, reduce_char_map) = utility.read_map('./data')
    test_data = utility.read_data('./data', 'mfcc', 'test')
    predict(test_data, model_path, model_name, './out/test.out')
