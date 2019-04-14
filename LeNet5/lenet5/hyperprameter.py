# ==============================================================================
import tensorflow as tf
import sys

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('input_node', 784, 'picture size you want to input to the network')
flags.DEFINE_integer('hidden_node', 256, 'number of hidden units')
flags.DEFINE_integer('output_node', 10, 'labels number you want to input to the network')
flags.DEFINE_integer('global_step', 300000, 'total step the network to train')

flags.DEFINE_integer('tt_ranks_1',  26, 'ranks of the first tensor train')
flags.DEFINE_integer('tt_ranks_2',  2, 'ranks of the second tensor train')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  ')
flags.DEFINE_float('REGULARIZER_RATE',  0.0001, 'L2 regularizer_rate')
flags.DEFINE_float('LEARNING_RATE_BASE',  0.01, 'L2 learning_rate_base')
flags.DEFINE_float('LEARNING_RATE_DECAY',  0.99, 'L2 learning_rate_decay')

##===========================================================================================================

def print_hyper_parameters():

    sys.stdout.write('batch_size      %d\n' % FLAGS.batch_size*FLAGS.num_gpus)

    sys.stdout.write('tt_ranks_1        %d\n' % FLAGS.tt_ranks_1)

    sys.stdout.write('tt_ranks_2        %d\n' % FLAGS.tt_ranks_2)