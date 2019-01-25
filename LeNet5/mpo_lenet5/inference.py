"""
This network we change the lenet5's full-connected layer with mpo structure and 
set the mpo structure as 2-5-6-2, 2-3-7-2,1-5-2-1 and rank=4.
compress ratio is 10.9
"""
import numpy as np
import tensorflow as tf
import tt
from hyper_parameters import *

input_node  = FLAGS.input_node
output_node = FLAGS.output_node


r_1 = FLAGS.tt_ranks_1
r_2 = FLAGS.tt_ranks_2

#TTO_layer1
inp_modes1 =  [2,5,6,2]          
out_modes1 =  [2,3,7,2]        
mat_rank1  =  [1,r_1,r_1,r_1,1]

#TTO_layer2
inp_modes2 = [2,3,7,2]    
out_modes2 = [1,5,2,1]        
mat_rank2 =  [1,r_2,r_2,r_2,1]



def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

def inference(inputs):
	W_conv1 = weight_variable([5, 5, 1, 6])
	b_conv1 = bias_variable([6])
	
	x_image = tf.reshape(inputs, [-1,28,28,1])
	#28x28x1
	h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1,1,1,1], padding='SAME') + b_conv1)
	# 28x28x6
	h_pool1 = max_pool_2x2(h_conv1)
	#14x14x6
	W_conv2 = weight_variable([5, 5, 6, 16])
	b_conv2 = bias_variable([16])
	
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	#10x10x16
	h_pool2 = max_pool_2x2(h_conv2)
	#5x5x16
	W_conv3 = weight_variable([5, 5, 16, 120])
	b_conv3 = bias_variable([120])
	h_conv3 = tf.nn.relu(conv2d(h_pool2,W_conv3) + b_conv3)
	#1x1x120
	inputs = tf.reshape(h_conv3,[-1, 120])
	inputs = tf.nn.relu(tt.tto(inputs,
							   np.array(inp_modes1,dtype=np.int32),
							   np.array(out_modes1,dtype=np.int32),
							   np.array(mat_rank1,dtype=np.int32),
							   biases_initializer=None,
							   scope='tt_scope_1'))

	output = tt.tto(inputs,
							   np.array(inp_modes2,dtype=np.int32),
							   np.array(out_modes2,dtype=np.int32),
							   np.array(mat_rank2,dtype=np.int32),
							   biases_initializer=None,
							   scope='tt_scope_2')
	return output

	

        
        
                        
    
