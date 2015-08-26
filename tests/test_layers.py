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
    lstm.set_state(batch_size)

    x = T.tensor3()
    f = theano.function([x], lstm(x), updates=lstm.updates)

    X = np.ones((batch_size, time_steps, input_size), dtype=np.float32)
    assert f(X).shape == (batch_size, time_steps, layer_size)


def test_bn():
    from layers import BN
    bn = BN(input_size)
    bn.set_state(input_size, time_steps)

    x = T.tensor3()
    f = theano.function([x], bn(x), updates=bn.updates)

    X = np.ones((batch_size, time_steps, input_size), dtype=np.float32)
    assert f(X).shape == (batch_size, time_steps, input_size)


def test_bn_lstm():
    from layers import BNLSTM
    bn_lstm = BNLSTM(input_size, layer_size)
    bn_lstm.set_state(batch_size, time_steps)

    x = T.tensor3()
    X = np.ones((batch_size, time_steps, input_size), dtype=np.float32)

    f = theano.function([x], bn_lstm(x), updates=bn_lstm.updates)
    assert f(X).shape == (batch_size, time_steps, layer_size)


test_bn_lstm()