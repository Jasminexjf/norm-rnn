import theano
import numpy as np


def test_sgd():
    from norm_rnn import SGD
    sgd = SGD()
    p = theano.shared(np.cast[theano.config.floatX](1))
    updates = sgd([p], [p])

    f = theano.function([], p, updates=updates)

    for i in range(1, 10):
        assert sgd.iteration.get_value() == i
        f()

        print sgd.lr.get_value()

test_sgd()