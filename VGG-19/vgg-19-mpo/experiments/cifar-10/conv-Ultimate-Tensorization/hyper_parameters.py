
# ==============================================================================
import tensorflow as tf
import sys
import math
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

net_type = 'VGG-19' 


flags.DEFINE_boolean('norm_after_aug', True, 'whether normalization after augmentation or before')
# by_sample:  subtract mean of all images and divide each entry by its standard deviation
# by_channel: subtract mean of every channel and divide each channel data by its standard deviation
flags.DEFINE_string('normalization','by_sample','image normalization method')

flags.DEFINE_integer('random_seed',12345, 'random seed.')

flags.DEFINE_float('MOMENTUM',     0.9, 'The Nesterov momentum of Momentum optimizer')
flags.DEFINE_float('WEIGHT_DECAY', 2e-4, 'weight decay for L2 regularization')
flags.DEFINE_float('res_conv_drop_prob', 0.0,  'residual convolution block dropout probability. 0 for no dropout.')
flags.DEFINE_float('res_TT_drop_prob',   0.0,  'residual TT block dropout probability. 0 for no dropout.')
flags.DEFINE_float('vgg_conv_drop_prob', 0.0,  'vgg convolution block dropout probability. 0 for no dropout.')
flags.DEFINE_float('initial_learning_rate', 0.1, 'initial learning rate')
flags.DEFINE_integer('num_epochs_per_decay', 50, 'NEPD')
flags.DEFINE_integer('learning_rate_decay_steps', 6, '.')
# flags.DEFINE_integer('learning_rate_decay_factor', 0.5, '.')
flags.DEFINE_float('learning_rate_decay_factor', math.pow(0.1, 0.5), '.')


flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('log_steps', 100, 'Summary log steps')
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus for training')

if net_type == 'VGG-19':
    flags.DEFINE_string('net_module', './nets/vgg-19-mpo.py', 'Module with architecture description.')
    flags.DEFINE_integer('tt_rank1', 4, 'mpo rank of 1st full-connected layer')
    flags.DEFINE_integer('tt_rank2', 4, 'mpo rank of 2nd full-connected layer')
    flags.DEFINE_integer('tt_rank3', 4, 'mpo rank of 3rd full-connected layer')

    log_dir = './log/VGG-19_wd%.1e_NEPD%d_LRDS%d' %\
              (FLAGS.WEIGHT_DECAY, FLAGS.num_epochs_per_decay,
               FLAGS.learning_rate_decay_steps)

max_epochs = FLAGS.num_epochs_per_decay * FLAGS.learning_rate_decay_steps + FLAGS.num_epochs_per_decay/2

learning_rate_decay_boundary = [100, 200, 250]
learning_rate_decay_value = [1, 0.1, 0.01,0.001]
flags.DEFINE_integer("max_epochs", int(max_epochs), 'Number of epochs to run trainer.')


flags.DEFINE_string('log_dir', log_dir, 'Directory to put log files.')
flags.DEFINE_string('data_dir', '../data/', 'Directory to put the training data.')

flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

##===========================================================================================================

def print_hyper_parameters():
    sys.stdout.write('random_seed     %d\n' % FLAGS.random_seed)

    sys.stdout.write('batch_size      %d\n' % FLAGS.batch_size*FLAGS.num_gpus)
    if net_type == 'VGG-19':
      sys.stdout.write('r1            %d\n' % FLAGS.tt_rank1)
      sys.stdout.write('r2            %d\n' % FLAGS.tt_rank2)
      sys.stdout.write('r3            %d\n' % FLAGS.tt_rank3)

    sys.stdout.write('MOMENTUM     %.1f\n' % FLAGS.MOMENTUM)
    sys.stdout.write('WEIGHT_DECAY %.1e\n' % FLAGS.WEIGHT_DECAY)

    sys.stdout.write('net_module   %s\n'   % FLAGS.net_module)
    sys.stdout.write('log_dir      %s\n'   % FLAGS.log_dir)