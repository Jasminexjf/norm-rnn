import theano
import theano.tensor as T
import numpy as np


class CrossEntropy(object):

    def __call__(self, x, y):
        return T.nnet.categorical_crossentropy(x, y).mean()


class Accuracy(object):

    def __call__(self, x, y):
        return T.eq(T.argmax(x, axis=-1), y).mean()


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


class DecayEvery(object):

    # every = batches per epoch * times max_epoch from lstm reference
    def __init__(self, every=2218 * 4, rate=0.5):
        self.every = every
        self.rate = np.cast[theano.config.floatX](rate)


class SGD(object):

    def __init__(self, lr=1, grad_norm=5, decay=DecayEvery()):
        self.lr = theano.shared(np.cast[theano.config.floatX](lr))
        self.iteration = theano.shared(1)
        self.grad_norm = grad_norm
        self.decay = decay

    def __call__(self, params, grads):
        norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
        grads = [self.clip_norm(g, self.grad_norm, norm) for g in grads]

        updates = [(self.iteration, self.iteration + 1),
                   (self.lr, T.switch(self.iteration % self.decay.every, self.lr, self.lr * self.decay.rate))]

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


def compile_model(model, dataset, optimizer=None):
    x = T.imatrix()
    y = T.ivector()

    X = theano.shared(dataset.X)
    Y = theano.shared(dataset.Y)

    # givens
    i = T.iscalar()
    givens = {x: X[i], y: Y[i]}

    # grads
    cost = CrossEntropy()(model(x), y)
    accuracy = Accuracy()(model(x), y)

    # updates
    if optimizer is None:
        updates = []
    else:
        grads = [T.grad(cost, param) for param in model.params]
        updates = optimizer(model.params, grads)

    for layer in model.layers:
        try:
            updates.extend(layer.updates)
        except AttributeError:
            pass

    return theano.function([i], accuracy, None, updates, givens)


