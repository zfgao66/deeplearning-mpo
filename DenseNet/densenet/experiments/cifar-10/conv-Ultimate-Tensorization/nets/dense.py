import tensorflow as tf
import math
import numpy as np
import sys

sys.path.append('../../../')
import tensornet

sys.path.append('../')
from hyper_parameters import *

NUM_CLASSES = 10

opts = {}
opts['use_dropout'] = True
opts['keep_prob'] = 1.0
opts['ema_decay'] = 0.99
opts['batch_norm_epsilon'] = 1e-3

dense_conv_drop_prob = FLAGS.dense_conv_drop_prob
layers_per_block = FLAGS.layers_per_block
in_feature = FLAGS.in_feature
growth = FLAGS.growth_rate


def batch_norm_relu(inputs, train_phase,cpu_variables=False,scope=None):
  """Performs a batch normalization followed by a ReLU."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  # inputs = tf.layers.batch_normalization(
  #     inputs=inputs, axis=3,
  #     momentum=opts['ema_decay'], epsilon=_BATCH_NORM_EPSILON, center=True,
  #     scale=True, training=train_phase, fused=True)
  inputs = tensornet.layers.batch_normalization(inputs, train_phase,
                                                cpu_variables=cpu_variables,
                                                ema_decay=opts['ema_decay'],
                                                eps=opts['batch_norm_epsilon'],
                                                scope=scope)
  inputs = tf.nn.relu(inputs)
  return inputs
  
def batch_activ_conv(inputs,in_features,out_features,kernel_size,train_phase, strides,
					  cpu_variables=False, prefix=None):
  """Standard building block for densely connected networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    out_features: The number of filters for the convolutions.
    train_phase: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts (typically
      a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block.
  """

  bn_scope = prefix + '_bn0'
  inputs = batch_norm_relu(inputs, train_phase, cpu_variables, bn_scope)
  conv_scope = prefix + '_conv0'
  inputs = tensornet.layers.conv(inputs, in_features, out_features, [kernel_size, kernel_size], strides,
								   cpu_variables=cpu_variables,
								   biases_initializer=None,
								   scope=conv_scope)
  if(dense_conv_drop_prob > 0.0):
    do_scope = prefix + '_do'
    inputs = tf.nn.dropout(inputs, keep_prob=1-dense_conv_drop_prob, name=do_scope)
  return inputs 
def dense_conv_block(inputs,layers,in_features,growth,train_phase,
                     cpu_variables=False, prefix=None):
  """Creates one layer of blocks for the densely connected model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    train_phase: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """
  current = inputs
  features = in_features
  for i in range(layers):
    blocks_prefix = prefix + '_%d' % (i)
    tmp = batch_activ_conv(inputs=current,in_features=features,kernel_size=3,
							   out_features=growth, train_phase=train_phase, 
							   strides=[1,1],cpu_variables=cpu_variables, prefix=blocks_prefix)
    current = tf.concat((current,tmp), axis=3)
    features += growth
  return current, features
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)
def avg_pool(input, s):
  return tf.nn.avg_pool(input, [ 1, s, s, 1 ], [1, s, s, 1 ], 'VALID')

def inference(inputs, train_phase, cpu_variables=False):
  layers = layers_per_block
  current = tensornet.layers.conv(inputs, in_ch=3, out_ch=in_feature, window_size=[3, 3],cpu_variables=cpu_variables,
								   biases_initializer=None,scope='initial_conv')
	######################################################################################################
  current, features = dense_conv_block(current, layers, in_features=in_feature, growth=growth,train_phase=train_phase,
										 cpu_variables=cpu_variables,prefix='dense_conv_block1')
  current = batch_activ_conv(current, features, features, 1, train_phase,strides=[1,1],prefix='batch_activ_conv1')
  current = avg_pool(current, 2)
	#######################################################################################################
  current, features = dense_conv_block(current, layers, in_features=features, growth=growth, train_phase=train_phase,
										 cpu_variables=cpu_variables, prefix='dense_conv_block2')
  current = batch_activ_conv(current, features, features, 1, train_phase,strides=[1,1], prefix='batch_activ_conv2')
  current = avg_pool(current, 2)
	#######################################################################################################
  current, features = dense_conv_block(current, layers, in_features= features, growth=growth, train_phase=train_phase,
										 cpu_variables=cpu_variables, prefix='dense_conv_block3')
  current = batch_norm_relu(current, train_phase=train_phase, cpu_variables=cpu_variables,scope='batch_norm_relu1')
  current = avg_pool(current, 8)
  final_dim = features
  current = tf.reshape(current, [-1, final_dim])
  output = tensornet.layers.linear(current, NUM_CLASSES, cpu_variables=cpu_variables,scope='final_linear')
  return output
def losses(logits, labels):
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
  loss = tf.reduce_mean(xentropy, name='loss')
  return [loss]	
def evaluation(logits, labels):
  correct_flags = tf.nn.in_top_k(logits, labels, 1)
  return tf.cast(correct_flags, tf.int32)





	



