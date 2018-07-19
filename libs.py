import tensorflow as tf
from config import cfg


# Defining custom operations
rf = tf.load_op_library('./custom_ops/fix_resolution.so')
fix_resolution = rf.fix_resolution
tf.NoGradient("FixResolution")


def fix(x):
    return fix_resolution(x, cfg.digit_bits, cfg.fraction_bits)


# TODO: Not used
# def conv_layer(input, in_channels, num_outputs,
#                kernel_size, stride, padding, act=tf.nn.relu):
#     W = tf.get_variable('W',
#         initializer=tf.truncated_normal([kernel_size, kernel_size,
#             in_channels, num_outputs], stddev=0.1))
#     conv = act(tf.nn.conv2d(input, W,
#         strides=[1, stride, stride, 1], padding=padding))
# 
#     if cfg.is_fixed:
#         conv = fix_resolution(conv,
#                 cfg.fixed_fine_range_bits, cfg.fixed_fine_precision_bits)
# 
#     tf.summary.histogram('W', W)
#     tf.summary.histogram('conv', conv)
# 
#     return conv
