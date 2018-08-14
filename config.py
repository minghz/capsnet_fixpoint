import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('epoch', 2, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')


############################
#   environment setting    #
############################
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')

flags.DEFINE_string('checkpoint_dir', 'checkpoint_dir', 'Dir checkpoints are saved')
flags.DEFINE_integer('save_checkpoint_steps', 10, 'save checkpoint every #(steps)')

flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_string('data_dir', './data', 'Dir for affnist data')
flags.DEFINE_string('dataset', 'mnist', 'mnist OR fashion-mnist')

flags.DEFINE_string('fix_method', 'trunc', '[trunc | nearest | stochastic]')
flags.DEFINE_integer('digit_bits', 10, 'number of digit bits for fix point')
flags.DEFINE_integer('fraction_bits', 22, 'number of fraction bits for fix point')

flags.DEFINE_string('logdir', 'logdir', 'logs directory')
flags.DEFINE_integer('train_sum_freq', 5, 'the frequency of saving train summary(step)')
flags.DEFINE_integer('val_sum_freq', 5, 'the frequency of saving valuation summary(step)')
flags.DEFINE_string('results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################
flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
