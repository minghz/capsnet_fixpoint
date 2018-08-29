import tensorflow as tf

from tensorflow.python.framework import ops
from config import cfg

# Truncating resolution - custom OP
tr = tf.load_op_library('./custom_ops/trunc_resolution.so')
trunc_resolution = tr.trunc_resolution
@ops.RegisterGradient("TruncResolution")
def _trunc_resolution_grad(op, grad):
    return tr.trunc_resolution_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


# Nearest resolution - custom OP
nr = tf.load_op_library('./custom_ops/nearest_resolution.so')
nearest_resolution = nr.nearest_resolution
# Gradient registration for out custom operation
@ops.RegisterGradient("NearestResolution")
def _nearest_resolution_grad(op, grad):
    return nr.nearest_resolution_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


# Stochastic resolution - custom OP
sr = tf.load_op_library('./custom_ops/stochastic_resolution.so')
stochastic_resolution = sr.stochastic_resolution
# Gradient registration for out custom operation
@ops.RegisterGradient("StochasticResolution")
def _stochastic_resolution_grad(op, grad):
    return sr.stochastic_resolution_grad(grad, op.inputs[0], op.inputs[1], op.inputs[2])


def fix(x, ss=False):
    # not fixing if training
    if cfg.is_training:
        return x

    if ss:
        digit_bits = cfg.ss_digit_bits
        fraction_bits = cfg.ss_fraction_bits
    else:
        digit_bits = cfg.digit_bits
        fraction_bits = cfg.fraction_bits

    if cfg.fix_method == 'trunc':
        return trunc_resolution(x, digit_bits, fraction_bits)
    elif cfg.fix_method == 'nearest':
        return nearest_resolution(x, digit_bits, fraction_bits)
    elif cfg.fix_method == 'stochastic':
        return stochastic_resolution(x, digit_bits, fraction_bits)
    else:
        return false
