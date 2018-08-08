import tensorflow as tf

from tensorflow.python.framework import ops
from config import cfg

# Flooring resolution - custom OP
rf = tf.load_op_library('./custom_ops/floor_resolution.so')
floor_resolution = rf.floor_resolution
@ops.RegisterGradient("FloorResolution")
def _floor_resolution_grad(op, grad):
    return rf.floor_resolution_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


# Nearest resolution - custom OP
rf = tf.load_op_library('./custom_ops/nearest_resolution.so')
nearest_resolution = rf.nearest_resolution
# Gradient registration for out custom operation
@ops.RegisterGradient("NearestResolution")
def _nearest_resolution_grad(op, grad):
    return rf.nearest_resolution_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


# Stochastic resolution - custom OP
rf = tf.load_op_library('./custom_ops/stochastic_resolution.so')
stochastic_resolution = rf.stochastic_resolution
# Gradient registration for out custom operation
@ops.RegisterGradient("StochasticResolution")
def _stochastic_resolution_grad(op, grad):
    return rf.stochastic_resolution_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


def fix(x):
    if cfg.fix_method == 'floor':
        return floor_resolution(x, cfg.digit_bits, cfg.fraction_bits)
    elif cfg.fix_method == 'nearest':
        return nearest_resolution(x, cfg.digit_bits, cfg.fraction_bits)
    elif cfg.fix_method == 'stochastic':
        return stochastic_resolution(x, cfg.digit_bits, cfg.fraction_bits)
    else:
        return false
