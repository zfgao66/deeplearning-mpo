
import numpy as np
import tensorflow as tf
import tt
from hyperprameter import *

r_1 = FLAGS.tt_ranks_1
r_2 = FLAGS.tt_ranks_2
r_3 = FLAGS.tt_ranks_3

input_node=FLAGS.input_node
output_node=FLAGS.output_node

# TTO_layer2
inp_modes2 = [2, 10, 10, 2]
out_modes2 = [2, 5, 6, 2]  
mat_rank2 = [1, r_1, r_1, r_1, 1]

# TTO_layer3
inp_modes3 = [2, 5, 6, 2]
out_modes3 = [2, 3, 7, 2]  
mat_rank3 = [1, r_2, r_2, r_2, 1]

# TTO_layer4
inp_modes4 = [2, 3, 7, 2]  
out_modes4 = [1, 5, 2, 1]
mat_rank4 = [1, r_3, r_3, r_3, 1]


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
    inputs = tf.reshape(h_pool2,[-1,400])
    inputs = tf.nn.relu(tt.tto(inputs,
                               np.array(inp_modes1, dtype=np.int32),
                               np.array(out_modes1, dtype=np.int32),
                               np.array(mat_rank1, dtype=np.int32),
                               scope='tt_scope_1'))
    inputs = tf.nn.relu(tt.tto(inputs,
                               np.array(inp_modes2, dtype=np.int32),
                               np.array(out_modes2, dtype=np.int32),
                               np.array(mat_rank2, dtype=np.int32),
                               scope='tt_scope_2'))
    
    output = tt.tto(inputs,
                    np.array(inp_modes3, dtype=np.int32),
                    np.array(out_modes3, dtype=np.int32),
                    np.array(mat_rank3, dtype=np.int32),
                    scope='tt_scope_3') 


    return output
