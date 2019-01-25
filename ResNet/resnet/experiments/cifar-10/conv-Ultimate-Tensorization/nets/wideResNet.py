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

res_conv_drop_prob = FLAGS.res_conv_drop_prob
res_TT_drop_prob   = FLAGS.res_TT_drop_prob
num_blocks = FLAGS.num_blocks
k = FLAGS.widening_factor

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

def res_conv_block(inputs, filters, train_phase, projection_shortcut, strides,
                   cpu_variables=False, prefix=None):
  """Standard building block for residual networks with BN before convolutions.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
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
  shortcut = inputs
  bn_scope = prefix + '_bn0'
  inputs = batch_norm_relu(inputs,train_phase,cpu_variables,bn_scope)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    projection_scope = prefix + '_projection'
    shortcut = projection_shortcut(inputs,projection_scope)

  conv_scope= prefix + '_conv0'
  inputs = tensornet.layers.conv(inputs, filters, [3, 3],strides,
                                 cpu_variables=cpu_variables,
                                 biases_initializer=None,
                                 scope=conv_scope)

  bn_scope = prefix + '_bn1'
  inputs = batch_norm_relu(inputs, train_phase,cpu_variables,bn_scope)

  if res_conv_drop_prob > 0.0:
      do_scope = prefix + '_do'
      inputs = tf.nn.dropout(inputs, keep_prob=1-res_conv_drop_prob, name=do_scope)

  conv_scope = prefix + '_conv1'
  inputs = tensornet.layers.conv(inputs, filters, [3, 3],[1,1],
                                 cpu_variables=cpu_variables,
                                 biases_initializer=None,
                                 scope=conv_scope)
  return inputs + shortcut

def res_conv_layer(inputs, filters, blocks, strides, train_phase,
                   cpu_variables=False, prefix=None):
  """Creates one layer of blocks for the ResNet model.

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

  def projection_shortcut(inputs,scope):
    return tensornet.layers.conv(inputs, filters, [1, 1],strides,
                                 cpu_variables=cpu_variables,
                                 biases_initializer=None,
                                 scope=scope)

  blocks_prefix = prefix + '_0'
  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = res_conv_block(inputs, filters, train_phase, projection_shortcut, strides,
                          cpu_variables=cpu_variables, prefix=blocks_prefix)

  for i in range(1, blocks):
    blocks_prefix = prefix + '_%d' % (i)
    inputs = res_conv_block(inputs, filters, train_phase, None, [1,1],
                            cpu_variables=cpu_variables, prefix=blocks_prefix)
  return inputs

def res_tt_layer(inputs,train_phase, inp_modes,out_modes,mat_ranks,
                 biases_initializer,cpu_variables,prefix):
    shortcut = inputs
    bn_scope = prefix + '_bn'
    inputs = batch_norm_relu(inputs, train_phase, cpu_variables, bn_scope)

    if res_TT_drop_prob > 0.0:
        do_scope = prefix + '_do'
        inputs = tf.nn.dropout(inputs, keep_prob=1 - res_TT_drop_prob, name=do_scope)

    tt_scope = prefix + '_tt'
    inputs = tensornet.layers.tt(inputs,
                                 np.array(inp_modes, dtype=np.int32),
                                 np.array(out_modes, dtype=np.int32),
                                 np.array(mat_ranks, dtype=np.int32),
                                 biases_initializer=None,
                                 cpu_variables=cpu_variables,
                                 scope=tt_scope)
    shortcut = tf.reshape(shortcut, inputs.get_shape().as_list())
    return inputs + shortcut

def inference(inputs, train_phase, cpu_variables=False):
    """Build the model up to where it may be used for inference.
    Args:
        images: Images placeholder.
        train_phase: Train phase placeholder
    Returns:
        logits: Output tensor with the computed logits.
    """
    tn_init = lambda dev: lambda shape: tf.truncated_normal(shape, stddev=dev)
    tu_init = lambda bound: lambda shape: tf.random_uniform(shape, minval = -bound, maxval = bound)
    batch_size = inputs.get_shape().as_list()[0]

    inputs = tensornet.layers.conv(inputs, 16, [3, 3],
                                   cpu_variables=cpu_variables,
                                   biases_initializer=None,
                                   scope='initial_conv')
    ##################################################################################
    inputs = res_conv_layer(inputs=inputs, filters=16*k, blocks=num_blocks,
                            strides=[1,1], train_phase=train_phase,
                            cpu_variables=cpu_variables,
                            prefix='res_conv_layer1')
    ##################################################################################
    inputs = res_conv_layer(inputs=inputs, filters=32*k, blocks=num_blocks,
                            strides=[2,2], train_phase=train_phase,
                            cpu_variables=cpu_variables,
                            prefix='res_conv_layer2')
    ##################################################################################
    inputs = res_conv_layer(inputs=inputs, filters=64*k, blocks=num_blocks,
                            strides=[2,2], train_phase=train_phase,
                            cpu_variables=cpu_variables,
                            prefix='res_conv_layer3')
    ##################################################################################
    inputs = batch_norm_relu(inputs, train_phase, cpu_variables, scope='final_bn')
    inputs = tf.layers.average_pooling2d(inputs=inputs, pool_size=8, strides=1,
                                         padding='VALID',name='final_avg_pool')
    inputs = tf.reshape(inputs, [batch_size, -1])
    ##################################################################################
    inputs = tensornet.layers.linear(inputs, NUM_CLASSES,cpu_variables=cpu_variables,
                                    scope='final_dense')
    ##################################################################################
    return inputs

def losses(logits, labels):
    """Calculates losses from the logits and the labels.
    Args:
        logits: input tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
    Returns:
        losses: list of loss tensors of type float.
    """
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='xentropy')
    loss = tf.reduce_mean(xentropy, name='loss')
    return [loss]

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).
    Returns:
        A scalar int32 tensor with the number of examples (out of batch_size)
        that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    correct_flags = tf.nn.in_top_k(logits, labels, 1)
    # Return the number of true entries.
    return tf.cast(correct_flags, tf.int32)
