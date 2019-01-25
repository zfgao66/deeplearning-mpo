
# ==============================================================================
import tensorflow as tf
import sys

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('input_node', 784, 'picture size you want to input to the network')
flags.DEFINE_integer('hidden_node', 256, 'number of hidden units')
flags.DEFINE_integer('output_node', 10, 'labels number you want to input to the network')
flags.DEFINE_integer('train_steps', 30000, 'total step the network to train')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate of the network')
flags.DEFINE_float('moving_decay', 0.99, '.')

flags.DEFINE_integer('tt_ranks_1',  4, 'ranks of the first tensor train')
flags.DEFINE_integer('tt_ranks_2',  4, 'ranks of the second tensor train')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  ')


##===========================================================================================================

def print_hyper_parameters():

    sys.stdout.write('batch_size      %d\n' % FLAGS.batch_size*FLAGS.num_gpus)

    sys.stdout.write('tt_ranks_1        %d\n' % FLAGS.tt_ranks_1)

    sys.stdout.write('tt_ranks_2        %d\n' % FLAGS.tt_ranks_2)