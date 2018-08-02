import tensorflow as tf

from tensorflow.python.framework import ops
from config import cfg

# Defining custom operations
rf = tf.load_op_library('./custom_ops/fix_resolution.so')
fix_resolution = rf.fix_resolution
# Gradient registration for out custom operation
@ops.RegisterGradient("FixResolution")
def _fix_resolution_grad(op, grad):
    return rf.fix_resolution_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])



def fix(x):
    return fix_resolution(x, cfg.digit_bits, cfg.fraction_bits)

#def conv_layer(input,
#        in_channels,
#        num_outputs,
#        kernel_size,
#        stride,
#        padding,
#        act=tf.nn.relu):
#
#    W = tf.get_variable(
#            'W',
#            initializer=tf.truncated_normal(
#                [kernel_size, kernel_size, in_channels, num_outputs],
#                stddev=0.1)
#            )
#
#    conv = act(tf.nn.conv2d(
#        input,
#        W,
#        strides=[1, stride, stride, 1], padding=padding)
#        )
#
#    conv = fix(conv)
#
#    tf.summary.histogram('W', W)
#    tf.summary.histogram('conv', conv)
#
#    return conv
