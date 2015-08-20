import theano
import theano.tensor as T
import numpy as np


batch_size = 2
time_steps = 3
input_size = 4
layer_size = 5


def test_penn_treebank():
    from norm_rnn import PennTreebank
    ptb = PennTreebank()

    x1, y1 = ptb.__iter__().next()
    x2, y2 = ptb.__iter__().next()


def test_progress_bar():
    from norm_rnn import ProgressBar

    for i in ProgressBar(range(10)):
        pass


def test_softmax():
    from layers import Softmax

    x = T.tensor3()
    f = theano.function([x], Softmax()(x))

    x = np.ones((batch_size, time_steps, input_size))
    assert f(x).shape == (batch_size * time_steps, input_size)


def test_crossentropy():
    from norm_rnn import CrossEntropy

    x = T.matrix()
    y = T.ivector()
    f = theano.function([x, y], CrossEntropy()(x, y))

    x = np.ones((batch_size * time_steps, input_size))
    y = np.ones(batch_size * time_steps, dtype=np.int32)
    assert f(x, y).shape == ()


def test_normalized_lstm():
    from layers import NormalizedLSTM

    x = T.tensor3()
    f = theano.function([x], NormalizedLSTM(input_size, layer_size)(x))

    x = np.ones((batch_size, time_steps, input_size))
    assert f(x).shape == (batch_size, time_steps, layer_size)


test_progress_bar()