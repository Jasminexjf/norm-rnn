import theano
import theano.tensor as T
import numpy as np


def test_penn_treebank():
    from norm_rnn import PennTreebank
    ptb = PennTreebank()

    x1, y1 = ptb.__iter__().next()
    x2, y2 = ptb.__iter__().next()


def test_softmax():
    from norm_rnn import Softmax

    x = T.tensor3()
    f = theano.function([x], Softmax()(x))

    x = np.ones((2, 3, 4))
    assert f(x).shape == (6, 4)


def test_crossentropy():
    from norm_rnn import CrossEntropy

    x = T.matrix()
    y = T.ivector()
    f = theano.function([x, y], CrossEntropy()(x, y))

    x = np.ones((6, 4))
    y = np.ones(6, dtype=np.int32)
    assert f(x, y).shape == ()


test_penn_treebank()