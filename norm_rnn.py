import sys
import theano
import theano.tensor as T
import numpy as np


class PennTreebank(object):

    train_path = 'data/ptb.train.txt'
    valid_path = 'data/ptb.valid.txt'
    test_path = 'data/ptb.test.txt'

    def __init__(self, batch_size=20, time_steps=20, path=train_path, tokenizer=str.split):
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

        # vocab to index dicts
        vocab = sorted(set(tokens))
        self.token_to_index = dict((t, i) for i, t in enumerate(vocab))
        self.index_to_token = dict((i, t) for i, t in enumerate(vocab))

        X = [self.token_to_index[t] for t in tokens]
        y = X[1:] + X[-1:]

        # reshape so that sentences are continuous across batches
        self.X = np.asarray(X).reshape(-1, batch_size, time_steps)
        self.y = np.asarray(y, np.int32).reshape(-1, batch_size, time_steps)

    def __iter__(self):
        for X, y in zip(self.X, self.y):
            # one_hot.shape = (batch_size, time_steps, vocab_size)
            one_hot = np.zeros(X.shape + (len(self.index_to_token),))

            for batch, x in enumerate(X):
                for step, i in enumerate(x):
                    one_hot[batch, step, i] = 1

            # y.reshape
            yield one_hot, y.reshape(-1)

    def __len__(self):
        # number of batches
        return len(self.X)


class LSTM(object):

    # stripped down LSTM from Keras

    def __init__(self, input_size, output_size, activation, inner_activation, weight_init):
        W_shape = input_size, output_size
        U_shape = output_size, output_size
        b_shape = output_size

        self.W_i = theano.shared(weight_init(W_shape))
        self.U_i = theano.shared(weight_init(U_shape))
        self.b_i = theano.shared(weight_init(b_shape))

        self.W_f = theano.shared(weight_init(W_shape))
        self.U_f = theano.shared(weight_init(U_shape))
        self.b_f = theano.shared(weight_init(b_shape))

        self.W_c = theano.shared(weight_init(W_shape))
        self.U_c = theano.shared(weight_init(U_shape))
        self.b_c = theano.shared(weight_init(b_shape))

        self.W_o = theano.shared(weight_init(W_shape))
        self.U_o = theano.shared(weight_init(U_shape))
        self.b_o = theano.shared(weight_init(b_shape))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        self.activation = activation
        self.inner_activation = inner_activation

    def __call__(self, x):
        x = x.dimshuffle((1, 0, 2))

        xi = T.dot(x, self.W_i) + self.b_i
        xf = T.dot(x, self.W_f) + self.b_f
        xc = T.dot(x, self.W_c) + self.b_c
        xo = T.dot(x, self.W_o) + self.b_o

        def alloc_zeros_matrix(*dims):
            return T.alloc(np.cast[theano.config.floatX](0.), *dims)

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(alloc_zeros_matrix(x.shape[1], xi.shape[2]), 1),
                T.unbroadcast(alloc_zeros_matrix(x.shape[1], xi.shape[2]), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
        )

        return outputs.dimshuffle((1, 0, 2))

    def _step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):

        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t


class Linear(object):

    def __init__(self, input_size, output_size, activation, weight_init):
        self.W = theano.shared(weight_init((input_size, output_size)))
        self.b = theano.shared(weight_init(output_size))
        self.params = [self.W, self.b]

        self.activation = activation

    def __call__(self, x):
        return self.activation(T.dot(x, self.W) + self.b)


class Softmax(object):

    def __call__(self, x):
        return T.nnet.softmax(x.reshape((-1, x.shape[-1])))


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


def compile(update=True):
    weight_init = lambda shape: np.random.uniform(-0.1, 0.1, shape)

    model = List([
        LSTM(9998, 200, T.tanh, T.nnet.sigmoid, weight_init),
        LSTM(200, 200, T.tanh, T.nnet.sigmoid, weight_init),
        Linear(200, 9998, Softmax(), weight_init)
    ])

    x = T.tensor3()
    y = T.ivector()

    cost = CrossEntropy()(model(x), y)
    grads = [T.grad(cost, param) for param in model.params]

    if update:
        updates = SGD()(model.params, grads)
    else:
        updates = None

    fit = theano.function([x, y], cost, updates=updates)
    return fit


class ProgressBar(object):

    def __init__(self, iter, prefix="", size=60):
        self.iter = iter
        self.prefix = prefix
        self.size = size

    def __iter__(self):
        count = len(self.iter)

        def _show(_i):
            x = int(self.size * _i / count)
            sys.stdout.write("%s[%s%s] %i/%i\r" % (self.prefix, "#"*x, "."*(self.size-x), _i, count))
            sys.stdout.flush()

        _show(0)

        for i, item in enumerate(self.iter, 1):
            yield item
            _show(i)
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == '__main__':
    fit = compile()
    validate = compile(update=False)

    for epoch in range(10):

        # how to reuse dataset iterator?

        # progress bar decorates dataset iterator
        train_set = ProgressBar(PennTreebank())
        valid_set = ProgressBar(PennTreebank(path=PennTreebank.valid_path))

        # train
        cost_list = []
        for X, y in train_set:
            cost_list.append(fit(X, y))
            train_set.prefix = str(np.mean(cost_list))

        # validate
        cost_list = []
        for X, y in valid_set:
            cost_list.append(validate(X, y))
            valid_set.prefix = str(np.mean(cost_list))

    # test
    test_set = ProgressBar(PennTreebank(path=PennTreebank.test_path))

    # static evaluation
    cost_list = []
    for X, y in test_set:
        cost_list.append(validate(X, y))
        test_set.prefix = str(np.mean(cost_list))

    # dynamic evaluation
    cost_list = []
    for X, y in test_set:
        cost_list.append(fit(X, y))
        test_set.prefix = str(np.mean(cost_list))
