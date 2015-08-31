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
        self.reset_state(dataset.batch_size, dataset.time_steps)
        self.batch_size = dataset.batch_size
        self.time_steps = dataset.time_steps

        # symbolic variables
        x = T.imatrix()
        y = T.imatrix()
        i = T.iscalar()

        # prediction
        if optimizer is None:
            self.mode(train=False)

        p = self(x)

        # compute metrics
        cost = CrossEntropy()(p, y)
        accuracy = Accuracy()(p, y)
        perplexity = T.exp(cost)

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
            self.val_batches = len(dataset.X_valid.get_value()) / dataset.batch_size
        else:
            givens = {x: dataset.X_train[i * dataset.batch_size:(i+1) * dataset.batch_size],
                      y: dataset.y_train[i * dataset.batch_size:(i+1) * dataset.batch_size]}

            # link to pentree.py
            scaled_cost = cost * dataset.time_steps
            grads = [T.grad(scaled_cost, param) for param in self.params]
            updates.extend(optimizer(self.params, grads))
            self.fit = theano.function([i], (perplexity, accuracy), None, updates, givens)
            self.fit_batches = len(dataset.X_train.get_value()) / dataset.batch_size

    def train(self, epochs):
        for epoch in range(1, epochs + 1):
            progress_bar = TrainProgressBar(epoch, self.fit_batches, self.val_batches)

            # fit
            fit_results = []
            for batch in range(self.fit_batches):
                fit_results.append(self.fit(batch))
                progress_bar.fit_update(fit_results)

            self.reset_state(self.batch_size, self.time_steps)

            # validate
            val_results = []
            for batch in range(self.val_batches):
                val_results.append(self.val(batch))
                progress_bar.val_update(val_results)

            self.reset_state(self.batch_size, self.time_steps)

            self.fit_results.extend(fit_results)
            self.val_results.extend(val_results)
            print

    def dump(self, file_name):
        import cPickle
        with open(file_name, 'w') as save_file:
            cPickle.dump((self.fit_results, self.val_results), save_file)

    def mode(self, train=True):
        for layer in self.layers:
            try:
                layer.train = train
            except AttributeError:
                pass # layer logic is independent of train/valid

    def reset_state(self, batch_size, time_steps):
        from layers import LSTM
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
