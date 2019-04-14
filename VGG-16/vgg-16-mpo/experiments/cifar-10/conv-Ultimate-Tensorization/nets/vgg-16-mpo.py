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
vgg_conv_drop_prob = 0
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
  if(vgg_conv_drop_prob > 0.0):
    do_scope = prefix + '_do'
    inputs = tf.nn.dropout(inputs, keep_prob=1-vgg_conv_drop_prob, name=do_scope)
  return inputs



def maxpooling(x, scope):
  return tf.nn.max_pool(x, ksize=[1,2,2,1],strides=[1,2, 2,1],padding="SAME",name=scope)
def inference(inputs, train_phase, cpu_variables=False):
  inputs = batch_activ_conv(inputs,3,64, 3, strides=[1,1],cpu_variables=cpu_variables,train_phase=train_phase, prefix='conv1_1')
  inputs = batch_activ_conv(inputs,64,64, 3, strides=[1,1],cpu_variables=cpu_variables,train_phase=train_phase, prefix='conv1_2')
  inputs = maxpooling(inputs,scope='max_pool1')
# 16x16
  inputs = batch_activ_conv(inputs,64,128, 3, strides=[1,1],cpu_variables=cpu_variables,train_phase=train_phase, prefix='conv2_1')
  inputs = batch_activ_conv(inputs,128,128, 3, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv2_2')
  inputs = maxpooling(inputs,scope='max_pool2')
#8x8
  inputs = batch_activ_conv(inputs,128,256, 3, strides=[1,1],cpu_variables=cpu_variables,train_phase=train_phase, prefix='conv3_1')
  inputs = batch_activ_conv(inputs,256,256, 3, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv3_2')
  inputs = batch_activ_conv(inputs,256,256, 1, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv3_3')
  inputs = maxpooling(inputs,scope='max_pool3')
#4x4
  inputs = batch_activ_conv(inputs,256,512, 3, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv4_1')
  inputs = batch_activ_conv(inputs,512,512, 3, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv4_2')
  inputs = batch_activ_conv(inputs,512,512, 1, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv4_3')
  inputs = maxpooling(inputs,scope='max_pool4')   
#2x2
  # inputs = batch_activ_conv(inputs,512,512, 3, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv5_1')
  # inputs = batch_activ_conv(inputs,512,512, 3, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv5_2')
  inputs = batch_norm_relu(inputs, train_phase, cpu_variables, scope='tt_00_bn')
  inputs = tensornet.layers.tt(inputs,
                               np.array([2,8,8,8,2],dtype=np.int32),
                               np.array([2,8,8,8,2],dtype=np.int32),
                               np.array([1,4,4,4,4,1],dtype=np.int32),
                               cpu_variables=cpu_variables,
                               scope='tt_00')
  inputs = batch_norm_relu(inputs, train_phase, cpu_variables, scope='tt_0_bn')
  inputs = tensornet.layers.tt(inputs,
                               np.array([2,8,8,8,2],dtype=np.int32),
                               np.array([2,8,8,8,2],dtype=np.int32),
                               np.array([1,4,4,4,4,1],dtype=np.int32),
                               cpu_variables=cpu_variables,
                               scope='tt_0')
  inputs = tf.reshape(inputs, [-1,2,2,512])
  inputs = batch_activ_conv(inputs,512,512, 1, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv5_3')
#  inputs = batch_activ_conv(inputs,512,512, 3, strides=[1,1],cpu_variables=cpu_variables, train_phase=train_phase,prefix='conv5_4')
  inputs = maxpooling(inputs,scope='max_pool4') 
#1x1
  inputs = tf.reshape(inputs, [-1, 1*1*512])
  inputs = tensornet.layers.tt(inputs,
                               np.array([4,4,4,4,2],dtype=np.int32),
                               np.array([4,4,8,8,4],dtype=np.int32),
                               np.array([1,4,4,4,4,1],dtype=np.int32),
                               cpu_variables=cpu_variables,
                               scope='tt_1')
  inputs = tf.nn.relu(inputs)
  inputs = tf.nn.dropout(inputs, keep_prob=0.8)
  inputs = tensornet.layers.tt(inputs,
                               np.array([4,4,8,8,4],dtype=np.int32),
                               np.array([4,4,8,8,4],dtype=np.int32),
                               np.array([1,4,4,4,4,1],dtype=np.int32),
                               cpu_variables=cpu_variables,
                               scope='tt_2')
  inputs = tf.nn.relu(inputs)
  inputs = tf.nn.dropout(inputs, keep_prob=0.8)
  inputs = tensornet.layers.tt(inputs,
                               np.array([4,4,8,8,4],dtype=np.int32),
                               np.array([1,10,1,1,1],dtype=np.int32),
                               np.array([1,4,4,4,4,1],dtype=np.int32),
                               biases_initializer=None,
                               cpu_variables=cpu_variables,
                               scope='tt_4')  
  return inputs

def losses(logits, labels):
  xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
  loss = tf.reduce_mean(xentropy, name='loss')
  return [loss]	
def evaluation(logits, labels):
  correct_flags = tf.nn.in_top_k(logits, labels, 1)
  return tf.cast(correct_flags, tf.int32)





	



