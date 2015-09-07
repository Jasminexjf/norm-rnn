import theano
import theano.tensor as T
import numpy as np


class DecayEvery(object):

    # every = batches per epoch * times max_epoch from lstm reference
    def __init__(self, every=2218 * 4, rate=0.5):
        self.every = every
        self.rate = np.cast[theano.config.floatX](rate)


class MaxNorm(object):

    def __init__(self, max_norm=5):
        self.max_norm = max_norm

    def __call__(self, grads):
        norm = T.sqrt(sum([T.sum(g ** 2) for g in grads]))
        return [self.clip_norm(g, self.max_norm, norm) for g in grads]

    def clip_norm(self, g, c, n):
        if c > 0:
            g = T.switch(T.ge(n, c), g * c / n, g)
        return g


class Clip(object):

    def __init__(self, clip=5):
        self.clip = clip

    def __call__(self, grads):
        return [T.clip(g, -self.clip, self.clip) for g in grads]


class SGD(object):

    def __init__(self, lr=1, grad=MaxNorm(), decay=DecayEvery()):
        self.lr = theano.shared(np.cast[np.float32](lr))
        self.iteration = theano.shared(1)
        self.grad = grad
        self.decay = decay

    def __call__(self, params, grads):
        grads = self.grad(grads)

        updates = [(self.iteration, self.iteration + 1)]

        if self.decay is not None:
            updates.append((self.lr, T.switch(self.iteration % self.decay.every, self.lr, self.lr * self.decay.rate)))

        for p, g in zip(params, grads):
            new_p = p - self.lr * g
            updates.append((p, new_p))

        return updates


class RMS(object):

    def __init__(self, lr=0.001, rho=0.9, epsilon=1e-6, grad=Clip(), decay=DecayEvery()):
        self.lr = theano.shared(np.cast[theano.config.floatX](lr))
        self.iteration = theano.shared(1)

        self.rho = rho
        self.epsilon = epsilon
        self.grad = grad
        self.decay = decay

    def __call__(self, params, grads):
        grads = self.grad(grads)

        updates = [(self.iteration, self.iteration + 1),
                   (self.lr, T.switch(self.iteration % self.decay.every, self.lr, self.lr * self.decay.rate))]

        accumulators = [theano.shared(p.get_value() * 0) for p in params]

        for p, g, a in zip(params, grads, accumulators):
            new_a = self.rho * a + (1 - self.rho) * g ** 2
            updates.append((a, new_a))

            new_p = p - self.lr * g / T.sqrt(new_a + self.epsilon)
            updates.append((p, new_p))

        return updates
