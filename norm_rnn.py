import theano
import theano.tensor as T
import numpy as np


class CrossEntropy(object):

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, x, y):
        return T.nnet.categorical_crossentropy(x + self.epsilon, y).mean()


class Accuracy(object):

    def __call__(self, x, y):
        return T.eq(T.argmax(x, axis=-1), y).mean() * 100.


class DecayEvery(object):

    # every = batches per epoch * times max_epoch from lstm reference
    def __init__(self, every=2218 * 4, rate=0.5):
        self.every = every
        self.rate = np.cast[theano.config.floatX](rate)


class GradientNorm(object):

    def __init__(self, max_norm=5):
        self.max_norm = max_norm

    def __call__(self, grads):
        norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
        return [self.clip_norm(g, self.max_norm, norm) for g in grads]

    def clip_norm(self, g, c, n):
        if c > 0:
            g = T.switch(T.ge(n, c), g * c / n, g)
        return g


class SGD(object):

    def __init__(self, lr=1, grad_norm=GradientNorm(), decay=DecayEvery()):
        self.lr = theano.shared(np.cast[theano.config.floatX](lr))
        self.iteration = theano.shared(1)
        self.grad_norm = grad_norm
        self.decay = decay

    def __call__(self, params, grads):
        grads = self.grad_norm(grads)

        updates = [(self.iteration, self.iteration + 1),
                   (self.lr, T.switch(self.iteration % self.decay.every, self.lr, self.lr * self.decay.rate))]

        for p, g in zip(params, grads):
            new_p = p - self.lr * g
            updates.append((p, new_p))

        return updates


class RMS(object):

    def __init__(self, lr=2e-3, rho=0.95, epsilon=1e-6, grad_norm=GradientNorm(), decay=DecayEvery()):
        self.lr = theano.shared(np.cast[theano.config.floatX](lr))
        self.iteration = theano.shared(1)

        self.rho = rho
        self.epsilon = epsilon
        self.grad_norm = grad_norm
        self.decay = decay

    def __call__(self, params, grads):
        grads = self.grad_norm(grads)

        updates = [(self.iteration, self.iteration + 1),
                   (self.lr, T.switch(self.iteration % self.decay.every, self.lr, self.lr * self.decay.rate))]

        accumulators = [theano.shared(p.get_value() * 0) for p in params]

        for p, g, a in zip(params, grads, accumulators):
            new_a = self.rho * a + (1 - self.rho) * g ** 2
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))

        return updates


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
            try:
	        params.extend(layer.params)
            except AttributeError:
		pass # layer has no params (i.e. dropout)
	return params


def compile_model(model, dataset, optimizer=None):
    x = T.imatrix()
    y = T.ivector()

    X = theano.shared(dataset.X)
    Y = theano.shared(dataset.Y)

    # givens
    i = T.iscalar()
    givens = {x: X[i], y: Y[i]}

    # scale cost by time_steps to account for differences with torch
    # https://github.com/skaae/nntools/blob/pentree_recurrent/examples/pentree.py
    cost = CrossEntropy()(model(x), y)    
    scaled_cost = cost * dataset.time_steps
   
    # perplexity
    perplexity = T.exp(cost)

    # percent correct
    accuracy = Accuracy()(model(x), y)

    # updates
    if optimizer is None:
        updates = []
    else:
        grads = [T.grad(scaled_cost, param) for param in model.params]
        updates = optimizer(model.params, grads)

    for layer in model.layers:
        try:
            updates.extend(layer.updates)
        except AttributeError:
            pass

    return theano.function([i], (perplexity, accuracy), None, updates, givens)


