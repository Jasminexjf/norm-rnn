import theano
import theano.tensor as T
from utils import TrainProgressBar


class CrossEntropy(object):

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon

    def __call__(self, x, y):
        return T.nnet.categorical_crossentropy(x + self.epsilon, y.flatten()).mean()


class Accuracy(object):

    def __call__(self, x, y):
        return T.eq(T.argmax(x, axis=-1), y.flatten()).mean() * 100.


class List(object):

    def __init__(self, layers):
        self.layers = layers
        self.fit = None
        self.val = None
        self.fit_results = []
        self.val_results = []

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def compile(self, dataset, optimizer=None):
        # symbolic variables
        x = T.imatrix()
        y = T.imatrix()
        i = T.iscalar()

        # prediction
        p = self(x)

        # compute metrics
        cost = CrossEntropy()(p, y)
        scaled_cost = cost * dataset.time_steps
        perplexity = T.exp(cost)
        accuracy = Accuracy()(p, y)

        # get non-param updates (e.g. LSTM/BN state)
        updates = []
        from layers import LSTM
        for layer in self.layers:
            if isinstance(layer, LSTM):
                updates.extend(layer.updates)

        if optimizer is None:
            givens = {x: dataset.X_valid[i * dataset.batch_size:(i+1) * dataset.batch_size],
                      y: dataset.y_valid[i * dataset.batch_size:(i+1) * dataset.batch_size]}

            self.val = theano.function([i], (perplexity, accuracy), None, updates, givens)
            self.val_batches = len(dataset.X_valid.get_value())
        else:
            givens = {x: dataset.X_train[i * dataset.batch_size:(i+1) * dataset.batch_size],
                      y: dataset.y_train[i * dataset.batch_size:(i+1) * dataset.batch_size]}

            grads = [T.grad(scaled_cost, param) for param in self.params]
            updates = optimizer(self.params, grads)
            self.fit = theano.function([i], (perplexity, accuracy), None, updates, givens)
            self.fit_batches = len(dataset.X_train.get_value())

    def train(self, epochs):
        for epoch in range(epochs):
            progress_bar = TrainProgressBar(epoch, self.fit_batches, self.val_batches)

            # fit
            fit_results = []
            for batch in range(self.fit_batches):
                fit_results.append(self.fit(batch))
                progress_bar.fit_update(fit_results)

            # validate
            val_results = []
            for batch in range(self.val_batches):
                val_results.append(self.val(batch))
                progress_bar.val_update(val_results)

            self.fit_results.append(fit_results)
            self.val_results.append(val_results)
            print

    def mode(self, train=True):
        for layer in self.layers:
            try:
                layer.train = train
            except AttributeError:
                pass # layer logic is independent of train/valid

    def reset_state(self, batch_size, time_steps):
        for layer in self.layers:
            if isinstance(layer, LSTM):
                layer.set_state(batch_size, time_steps)

    @property
    def params(self):
        params = []
        for layer in self.layers:
            try:
                params.extend(layer.params)
            except AttributeError:
                pass # layer has no params (i.e. dropout)
        return params