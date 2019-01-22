import numpy as np
import tensorflow as tf
from tqdm import tqdm
import argparse
import sys
sys.path.append('../utils')
from vgg19 import VGG19
import load
import augment

learning_rate = 1e-3
batch_size = 128

def train():
    x = tf.placeholder(tf.float32, [None, 96, 96, 3])
    t = tf.placeholder(tf.int32, [None])
    is_training = tf.placeholder(tf.bool, [])

    model = VGG19(x, t, is_training)
    sess = tf.Session()
    with tf.variable_scope('vgg19'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = opt.minimize(model.loss, global_step=global_step)
    init = tf.global_variables_initializer()
    sess.run(init)

    # Restore the latest model
    if tf.train.get_checkpoint_state('backup/'):
        saver = tf.train.Saver()
        saver.restore(sess, 'backup/latest')

    # Load the dataset
    x_train, t_train, x_test, t_test = load.load()

    # Train
    while True:
        epoch = int(sess.run(global_step) / np.ceil(len(x_train)/batch_size)) + 1
        print('epoch:', epoch)
        perm = np.random.permutation(len(x_train))
        x_train = x_train[perm]
        t_train = t_train[perm]
        sum_loss_value = 0
        for i in tqdm(range(0, len(x_train), batch_size)):
            x_batch = augment.augment(x_train[i:i+batch_size])
            t_batch = t_train[i:i+batch_size]
            _, loss_value = sess.run(
                [train_op, model.loss],
                feed_dict={x: x_batch, t: t_batch, is_training: True})
            sum_loss_value += loss_value
        print('loss:', sum_loss_value)

        saver = tf.train.Saver()
        saver.save(sess, 'backup/latest', write_meta_graph=False)

        prediction = np.array([])
        answer = np.array([])
        for i in range(0, len(x_test), batch_size):
            x_batch = augment.augment(x_test[i:i+batch_size])
            t_batch = t_test[i:i+batch_size]
            output = model.out.eval(
                feed_dict={x: x_batch, is_training: False}, session=sess)
            prediction = np.concatenate([prediction, np.argmax(output, 1)])
            answer = np.concatenate([answer, t_batch])
            correct_prediction = np.equal(prediction, answer)
        accuracy = np.mean(correct_prediction)
        print('accuracy:', accuracy)


if __name__ == '__main__':
    train()

