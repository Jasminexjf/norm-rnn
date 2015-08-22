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
    lstm.set_state(np.zeros((batch_size, layer_size), dtype=np.float32),
                   np.zeros((batch_size, layer_size), dtype=np.float32))

    x = T.tensor3()
    f = theano.function([x], lstm(x), updates=lstm.updates)

    # check shape
    X = np.ones((batch_size, time_steps, input_size))
    assert f(X).shape == (batch_size, time_steps, layer_size)

    h1, c1 = lstm.h.get_value(), lstm.c.get_value()

    f(X)

    h2, c2 = lstm.h.get_value(), lstm.c.get_value()

    assert not np.all(h1 == h2)
    assert not np.all(c1 == c2)


def test_batch_norm():
    from layers import BatchNormalization

    for axis in [0, 1, 2]:
        x = T.tensor3()
        f = theano.function([x], BatchNormalization(input_size, axis)(x))

        x = np.ones((batch_size, time_steps, input_size))
        assert f(x).shape == (batch_size, time_steps, input_size)


def test_norm_layers():
    from layers import NormalizedLSTM, HiddenNormalizedLSTM
    from layers import NonGateNormalizedLSTM, NonGateHiddenNormalizedLSTM
    from layers import PreActNormalizedLSTM, NonGatePreActNormalizedLSTM

    for layer in [NormalizedLSTM, HiddenNormalizedLSTM,
                  NonGateNormalizedLSTM, NonGateHiddenNormalizedLSTM,
                  PreActNormalizedLSTM, NonGatePreActNormalizedLSTM]:

        for axis in [0, 1]:
            x = T.tensor3()
            f = theano.function([x], layer(input_size, layer_size, norm_axis=axis)(x))

            x = np.ones((batch_size, time_steps, input_size))
            assert f(x).shape == (batch_size, time_steps, layer_size)


def test_lstm_update():
    from layers import LSTM, Linear
    lstm = LSTM(input_size, layer_size)
    linear = Linear(layer_size, input_size)

    x = T.tensor3()
    y = T.ivector()

    X = np.ones((batch_size, time_steps, input_size), dtype=np.float32)
    Y = np.ones((batch_size * time_steps), dtype=np.int32)

    from norm_rnn import CrossEntropy
    cost = CrossEntropy()(linear(lstm(x)), y)
    grads = [T.grad(cost, param) for param in lstm.params]

    #for param in lstm.params:
    #    print param

    #for grad in grads:
    #    print grad.eval({x: X, y: Y}).dtype

    from norm_rnn import SGD
    updates = SGD()(lstm.params, grads)

    f = theano.function([x, y], cost)

    print f(x, y)


test_lstm_update()