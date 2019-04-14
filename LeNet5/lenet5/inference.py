
import numpy as np
import tensorflow as tf
import tt
from hyperprameter import *


input_node=FLAGS.input_node
output_node=FLAGS.output_node



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    return weights


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def inference(inputs):
    W_conv1 = weight_variable([5, 5, 1, 6])
    b_conv1 = bias_variable([6])

    x_image = tf.reshape(inputs, [-1, 28, 28, 1])
    # 28x28x1
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    # 28x28x6
    h_pool1 = max_pool_2x2(h_conv1)
    # 14x14x6
    W_conv2 = weight_variable([5, 5, 6, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    # 10x10x16
    h_pool2 = max_pool_2x2(h_conv2)
    # 5x5x16
    W_conv3 = weight_variable([5, 5, 16, 120])
    b_conv3 = bias_variable([120])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # 1x1x120
    inputs = tf.reshape(h_conv3, [-1, 120])
    W_fc1 = weight_variable([120, 84])
    h_fc1 = tf.nn.relu(tf.matmul(inputs, W_fc1))
    W_fc2 = weight_variable([84, 10])
    output = tf.matmul(h_fc1, W_fc2)

  
    return output
