import sys
import timeit

import theano
import theano.tensor as T
import numpy as np


class PennTreebank(object):

    train_path = 'data/ptb.train.txt'
    valid_path = 'data/ptb.valid.txt'
    test_path = 'data/ptb.test.txt'

    def __init__(self, batch_size=20, time_steps=20, path=train_path,
                 tokenizer=str.split, vocab=None):
        text = open(path).read()

        if tokenizer is not None:
            # tokenizer = str.split gives words
            tokens = tokenizer(text)
        else:
            # tokenizer = None gives chars
            tokens = text

        # round off extra tokens
        extra_tokens = len(tokens) % (batch_size * time_steps)
        if extra_tokens:
            tokens = tokens[:-extra_tokens]

        # create vocab
        if vocab is None:
            vocab = sorted(set(tokens))

        self.token_to_index = dict((t, i) for i, t in enumerate(vocab))

        X = [self.token_to_index[t] for t in tokens]
        y = X[1:] + X[-1:]

        # cast to int32 for gpu compatibility
        X = np.asarray(X, dtype=np.int32)
        y = np.asarray(y, dtype=np.int32)

        # reshape so that sentences are continuous across batches
        self.X = X.reshape(-1, batch_size, time_steps)
        self.y = y.reshape(-1, batch_size, time_steps)

        self.batch_size = batch_size
        self.time_steps = time_steps

    def __iter__(self):
        for X, y in zip(self.X, self.y):
            yield X, y.reshape(-1) # move reshape to cost function

    def __len__(self):
        # number of batches
        return len(self.X)

    @property
    def vocab(self):
        return sorted(self.token_to_index.keys())

    @property
    def index_to_token(self):
        return dict((v, k) for k, v in self.token_to_index.iteritems())


class ProgressBar(object):

    # http://code.activestate.com/recipes/576986-progress-bar-for-console-programs-as-iterator/

    def __init__(self, iter, size=60):
        self.iter = iter
        self.size = size
        self.loss = 0

    def __iter__(self):
        start = timeit.default_timer()
        count = len(self.iter)

        def _show(_i):
            x = int(self.size * _i / count)
            progress = '#' * x
            remaining = '.' * (self.size - x)
            time_elapsed = timeit.default_timer() - start
            bar = '[{}{}] batch({}/{}) loss({:.2f}), time({:.2f})\r'.format(
                progress, remaining, _i, count, self.loss, time_elapsed)
            sys.stdout.write(bar)
            sys.stdout.flush()

        _show(0)

        for i, item in enumerate(self.iter, 1):
            yield item
            _show(i)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def set_loss(self, loss):
        self.loss = loss


class CrossEntropy(object):

    def __call__(self, x, y):
        return T.nnet.categorical_crossentropy(x, y).mean()


class RMSprop(object):

    def __init__(self, lr=2e-3, decay=0.95, epsilon=1e-6):
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon

    def __call__(self, params, grads):

        print 'add gradient clipping!'

        accumulators = [theano.shared(p.get_value() * 0) for p in params]
        updates = []

        for p, g, a in zip(params, grads, accumulators):
            new_a = self.decay * a + (1 - self.decay) * g ** 2
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))

        return updates


class SGD(object):

    def __init__(self, lr=1):
        self.lr = lr

    def __call__(self, params, grads):
        # norm clipping
        norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
        grads = [self.clip_norm(g, 1, norm) for g in grads]

        updates = []

        for p, g in zip(params, grads):
            new_p = p - self.lr * g
            updates.append((p, new_p))

        return updates

    def clip_norm(self, g, c, n):
        if c > 0:
            g = T.switch(T.ge(n, c), g * c / n, g)
        return g


class List(object):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    @property
    def params(self):
        params = []
        for layer in self.layers:
            params.extend(layer.params)
        return params


def compile_model(model, update=True):
    x = T.tensor3()
    y = T.ivector()

    # gradients
    cost = CrossEntropy()(model(x), y)
    grads = [T.grad(cost, param) for param in model.params]

    # param updates
    if update:
        updates = SGD()(model.params, grads)
    else:
        updates = []

    # state updates
    for layer in model.layers:
        try:
            updates.extend(layer.updates)
        except AttributeError:
            pass

    # compile
    return theano.function([x, y], cost, updates=updates)