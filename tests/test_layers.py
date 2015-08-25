import theano
import theano.tensor as T
import numpy as np


batch_size = 2
time_steps = 3
input_size = 4
layer_size = 5


def test_softmax():
    from layers import Softmax

    x = T.tensor2()
    f = theano.function([x], Softmax()(x))

    x = np.ones((batch_size, time_steps, input_size))
    assert f(x).shape == (batch_size * time_steps, input_size)


def test_embed():
    from layers import Embed

    x = T.imatrix()
    f = theano.function([x], Embed(input_size, layer_size)(x))

    X = np.ones((batch_size, time_steps), dtype=np.int32)
    assert f(X).shape == (batch_size, time_steps, layer_size)


def test_crossentropy():
    from norm_rnn import CrossEntropy

    x = T.matrix()
    y = T.ivector()
    f = theano.function([x, y], CrossEntropy()(x, y))

    x = np.ones((batch_size * time_steps, input_size))
    y = np.ones(batch_size * time_steps, dtype=np.int32)
    assert f(x, y).shape == ()


def test_lstm():
    from layers import LSTM
    lstm = LSTM(input_size, layer_size)

    x = T.tensor3()
    f = theano.function([x], lstm(x))

    X = np.ones((batch_size, time_steps, input_size), dtype=np.float32)
    assert f(X).shape == (batch_size, time_steps, layer_size)

    lstm.set_state(batch_size)

    f = theano.function([x], lstm(x), updates=lstm.updates)
    assert f(X).shape == (batch_size, time_steps, layer_size)


def test_batch_norm():
    from layers import BN
    bn = BN(input_size)

    x = T.tensor3()
    f = theano.function([x], bn(x))

    x = np.ones((batch_size, time_steps, input_size), dtype=np.float32)
    x[1] *= 2

    bn = (x - x.mean(0)) / (x.std(0))
    assert np.allclose(f(x), bn)


def test_batch_norm_lstm():
    pass


test_batch_norm()