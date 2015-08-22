import theano
import theano.tensor as T


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


def compile_model(model, dataset, update=True):
    x = T.imatrix()
    y = T.ivector()

    X = theano.shared(dataset.X)
    Y = theano.shared(dataset.Y)

    # givens
    i = T.iscalar()
    givens = {x: X[i], y: Y[i]}

    # grads
    cost = CrossEntropy()(model(x), y)

    # updates
    if update:
        grads = [T.grad(cost, param) for param in model.params]
        updates = SGD()(model.params, grads)
    else:
        updates = []

    return theano.function([i], cost, None, updates, givens)
