import tensorflow as tf
import os
from mnist_cnn import mnist_inference
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "G:\移动\桌面\TF\mnist_cnn\model"
MODEL_NAME = "model.ckpt"

def train():
    mnist = input_data.read_data_sets("G:\移动\桌面\TF\mnist_cnn\MNIST_data", one_hot=True)
    x = tf.placeholder(
        tf.float32, [BATCH_SIZE, mnist_inference.IMAGE_SIZE, mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS], name='input-x'
    )
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODE], name='input-y'
    )
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables()
    )
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)#设置最高显存占用
    config = tf.ConfigProto(gpu_options=gpu_options)#设置最高显存占用
    with tf.Session(config=config) as sess: #设置最高显存占用
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(xs, (BATCH_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.IMAGE_SIZE,mnist_inference.NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshape_xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training steps(s),loss on training ""batch is %g."%(step, loss_value))
                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step
                )
