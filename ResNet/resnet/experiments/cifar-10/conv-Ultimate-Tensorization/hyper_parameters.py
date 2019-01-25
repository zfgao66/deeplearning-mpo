
# ==============================================================================
import tensorflow as tf
import sys
import math
# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS

net_type = 'ResNet' # ResNet 

flags.DEFINE_boolean('norm_after_aug', True, 'whether normalization after augmentation or before')
# by_sample:  subtract mean of all images and divide each entry by its standard deviation
# by_channel: subtract mean of every channel and divide each channel data by its standard deviation
flags.DEFINE_string('normalization','by_sample','image normalization method')

flags.DEFINE_integer('random_seed',12345, 'random seed.')

flags.DEFINE_float('MOMENTUM',     0.9, 'The Nesterov momentum of Momentum optimizer')
flags.DEFINE_float('WEIGHT_DECAY', 2e-4, 'weight decay for L2 regularization')
flags.DEFINE_float('res_conv_drop_prob', 0.0,  'residual convolution block dropout probability. 0 for no dropout.')
flags.DEFINE_float('res_TT_drop_prob',   0.0,  'residual TT block dropout probability. 0 for no dropout.')

flags.DEFINE_float('initial_learning_rate', 0.1, 'initial learning rate')
flags.DEFINE_integer('num_epochs_per_decay', 50, 'NEPD')
flags.DEFINE_integer('learning_rate_decay_steps', 6, '.')
# flags.DEFINE_integer('learning_rate_decay_factor', 0.5, '.')
flags.DEFINE_float('learning_rate_decay_factor', math.pow(0.1, 0.5), '.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_integer('log_steps', 100, 'Summary log steps')
flags.DEFINE_integer('num_gpus', 1, 'Number of gpus for training')

if net_type == 'ResNet':
    flags.DEFINE_integer('num_blocks', 3, 'number of blocks')
    flags.DEFINE_integer('widening_factor', 4, 'k')
    flags.DEFINE_string('net_module', './nets/wideResNet.py', 'Module with architecture description.')
    log_dir = './log/wideResNet%d-ap_layer%d_wd%.1e_NEPD%d_LRDS%d_batch%d_convDP%.1f' % \
              (FLAGS.widening_factor,
               FLAGS.num_blocks * 6 + 2,
               FLAGS.WEIGHT_DECAY,
               FLAGS.num_epochs_per_decay, FLAGS.learning_rate_decay_steps,
               FLAGS.batch_size*FLAGS.num_gpus,
               FLAGS.res_conv_drop_prob)
    pretrained_ckpt = './log/wideResNet%d-TT0-ap_layer%d_r0_wd%.1e_NEPD%d_LRDS%d_batch%d_convDP%.1f/model.ckpt-508300' % \
              (FLAGS.widening_factor, FLAGS.num_blocks * 6  + 2,
               FLAGS.WEIGHT_DECAY, FLAGS.num_epochs_per_decay, FLAGS.learning_rate_decay_steps,
               FLAGS.batch_size * FLAGS.num_gpus,
               FLAGS.res_conv_drop_prob)
    flags.DEFINE_string('pretrained_ckpt', pretrained_ckpt,
                        'Pretrained checkpoint file.')
max_epochs = FLAGS.num_epochs_per_decay * FLAGS.learning_rate_decay_steps + FLAGS.num_epochs_per_decay/2
flags.DEFINE_integer("max_epochs", int(max_epochs), 'Number of epochs to run trainer.')


flags.DEFINE_string('log_dir', log_dir, 'Directory to put log files.')
flags.DEFINE_string('data_dir', '../data/', 'Directory to put the training data.')

flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
def print_hyper_parameters():
    sys.stdout.write('random_seed     %d\n' % FLAGS.random_seed)

    sys.stdout.write('batch_size      %d\n' % FLAGS.batch_size*FLAGS.num_gpus)

    if net_type == 'ResNet':
        sys.stdout.write('num_blocks      %d\n' % FLAGS.num_blocks)
        sys.stdout.write('widening_factor %d\n' % FLAGS.widening_factor)
        depth = FLAGS.num_blocks * 6 + 2
        sys.stdout.write('depth           %d\n' % depth)

    sys.stdout.write('MOMENTUM     %.1f\n' % FLAGS.MOMENTUM)
    sys.stdout.write('WEIGHT_DECAY %.1e\n' % FLAGS.WEIGHT_DECAY)

    sys.stdout.write('net_module   %s\n'   % FLAGS.net_module)
    sys.stdout.write('log_dir      %s\n'   % FLAGS.log_dir)