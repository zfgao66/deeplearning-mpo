# -*- coding: utf-8 -*-
"""
@author: zfgao
"""

# FC2 network without biases
import tensorflow as tf
from hyper_parameters import *
#define the parameters 
input_node=FLAGS.input_node
output_node=FLAGS.output_node
hidden1_node=FLAGS.hidden_node  
def get_weight_variable(shape):
    weights=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    tf.summary.histogram('weights', weights)
    return weights

def get_biases_variable(shape):
    biases=tf.Variable(tf.zeros(shape))
    tf.summary.histogram('biases', biases)
    return biases
# we set this full-connected layers without biases.
def inference(input_tensor):
    w1=get_weight_variable([input_node,hidden1_node])
    y1=tf.matmul(input_tensor,w1)
    y1=tf.nn.relu(y1)
    w2=get_weight_variable([hidden1_node,output_node])
    y=tf.matmul(y1,w2)
    return y

     
        
