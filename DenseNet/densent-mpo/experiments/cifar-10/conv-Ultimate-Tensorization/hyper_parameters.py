
# ==============================================================================
import tensorflow as tf
import sys
import math
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

net_type = 'DenseNet'


flags.DEFINE_boolean('norm_after_aug', True, 'whether normalization after augmentation or before')
# by_sample:  subtract mean of all images and divide each entry by its standard deviation
# by_channel: subtract mean of every channel and divide each channel data by its standard deviation
flags.DEFINE_string('normalization','by_sample','image normalization method')

flags.DEFINE_integer('random_seed',12345, 'random seed.')

flags.DEFINE_float('MOMENTUM',     0.9, 'The Nesterov momentum of Momentum optimizer')
flags.DEFINE_float('WEIGHT_DECAY', 2e-4, 'weight decay for L2 regularization')
flags.DEFINE_float('dense_conv_drop_prob',   0.0,  'densenet mpo layer dropout probability. 0 for no dropout.')

flags.DEFINE_float('initial_learning_rate', 0.1, 'initial learning rate')
flags.DEFINE_integer('num_epochs_per_decay', 50, 'NEPD')
flags.DEFINE_integer('learning_rate_decay_steps', 6, '.')
# flags.DEFINE_integer('learning_rate_decay_factor', 0.5, '.')
flags.DEFINE_float('learning_rate_decay_factor', math.pow(0.1, 0.5), '.')

flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('log_steps', 100, 'Summary log steps')
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus for training')

if net_type == 'DenseNet':
    flags.DEFINE_integer('growth_rate', 12, 'Growth rate for every layer, choices in paper: [12,24,40]')
    flags.DEFINE_integer('in_feature', 16, 'input filters to DenseNet first layers')
    flags.DEFINE_integer('total_blocks', 3, 'Total blocks of layers stack')
    flags.DEFINE_integer('layers_per_block', 12, 'depth=layers_per_block*total_blocks+total_blocks+1+n_tt_after_dense')
    flags.DEFINE_float('reduction', 0.5, 'reduction Theta at transition layer for DenseNets-BC models')
    flags.DEFINE_integer('tt_rank', 4, 'MPO dimention in the last mpo layer')

    flags.DEFINE_string('net_module', './nets/dense-mpo.py', 'Module with architecture description.')
    log_dir = './log/dense_r%d_wd%.1e_NEPD%d_LRDS%d' % \
              (FLAGS.tt_rank,
               FLAGS.WEIGHT_DECAY,
               FLAGS.num_epochs_per_decay,
               FLAGS.learning_rate_decay_steps)
    flags.DEFINE_string('pretrained_ckpt', './log/dense-TT0-ap_r0_wd2.0e-04/model.ckpt-211140',
                        'Pretrained checkpoint file.')

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

    if net_type == 'DenseNet':
       sys.stdout.write('growth_rate       %d\n' % FLAGS.growth_rate)
       sys.stdout.write('total_blocks      %d\n' % FLAGS.total_blocks)
       sys.stdout.write('layers_per_block  %d\n' % FLAGS.layers_per_block)
       sys.stdout.write('reduction       %.2f\n' % FLAGS.reduction)
       sys.stdout.write('r                 %d\n' % FLAGS.tt_rank)
       depth = FLAGS.layers_per_block * FLAGS.total_blocks * FLAGS.growth_rate + 4
       sys.stdout.write('depth             %d\n' % depth)


    sys.stdout.write('MOMENTUM     %.1f\n' % FLAGS.MOMENTUM)
    sys.stdout.write('WEIGHT_DECAY %.1e\n' % FLAGS.WEIGHT_DECAY)

    sys.stdout.write('net_module   %s\n'   % FLAGS.net_module)
    sys.stdout.write('log_dir      %s\n'   % FLAGS.log_dir)