# Different loss terms that can be added on.

# Chainer[170ecf]
import chainer

import common

# Both jacobianNorm(...) and hessianNorm(...) are zero-dimensional in their return value because
# chainer.functions.mean_squared_error [cf684f] is and they may need to be
# added to what it hands back (see workbench.py). The "jacobianNorm" function refers to GMR in
# [cfb615, section 3.2.1]
####################################################################################################
#                                                                                                  #

def jacobianNorm(batch, loss):
    partial = chainer.grad((loss, ), (batch, ), enable_double_backprop=True)
    return common.n_norm(2, partial[0], None)

def hessianNorm(batch, loss):
    return common.n_norm(2,
            chainer.functions.grad( (batch.grad,), (batch,), notsurewhattoputhere ),
            None)

#                                                                                                  #
####################################################################################################

# Not sure if this is the wrong way to do this, but this implements mean squared error (if
# "aggregate" is set to True) that is mathematically the same as
# chainer.functions.mean_squared_error(...)[cf684f] except for the fact that
# the average is calculated by dividing the sum of squares just by how many prediction vectors
# there are in "guesses"
def squaredDistance(guesses, labels, aggregate=True):
    
    # Using subtraction in the numerator may have been from [7adec0, MeanSquaredError.forward_(cpu/gpu)(...) method],
    # [b177d3, body of SquaredError.forward(...)], or some other reference, but I mean I would have done it this
    # way even if I hadn't seen that reference, so this may be a needless citation (and, again, I'm
    # not even sure if I used the reference for this). They also might have done one big subtraction
    # (instead of on a per-sample basis) for performance reasons, and I'm doing the same for those
    # reasons as well, so maybe I should credit them for that motivation too if it actually did
    # influence me to do so.
    numerator = None
    denominator = None
    if aggregate:
        numerator   = chainer.functions.sum( common.n_norm(2, guesses - labels, 1) ** 2 )
        denominator = guesses.shape[0]
    else:
        numerator   = common.n_norm(2, guesses - labels, 1) ** 2
        denominator = 1


    return numerator / denominator