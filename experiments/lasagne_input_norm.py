import numpy as np
np.random.seed(0)

from datasets import PennTreebank
from utils import ProgressBar
from norm_rnn import *
from layers import *

# lasagne loader for debugging
from tests.test_dataset import LasagneLoader

# hyper parameters
layer_size = 200
time_steps = 20
batch_size = 20
max_norm = 5
decay_rate = 0.5
decay_epoch = 4
learning_rate = 1
epochs = 6

dataset = LasagneLoader()
X_train, y_train = dataset()
X_valid, y_valid = dataset(False)

vocab_size = len(dataset.vocab_map)
train_batches = len(X_train) / batch_size
valid_batches = len(X_valid) / batch_size

# config model
models = {
    'reference':
    List([
        Embed(vocab_size, layer_size),
        LSTM(layer_size, layer_size),
        LSTM(layer_size, layer_size),
        Linear(layer_size, vocab_size)
    ]),

    'batch-norm':
    List([
        Embed(vocab_size, layer_size),
        BN(layer_size, time_steps=time_steps),
        LSTM(layer_size, layer_size),
        BN(layer_size, time_steps=time_steps),
        LSTM(layer_size, layer_size),
        Linear(layer_size, vocab_size)
    ]),
}

for model_name, model in models.iteritems():

    print 'model({})'.format(model_name)

    # initialize shared memory for lstm states
    # (find a way to push this into the layer)
    for layer in model.layers:
        if isinstance(layer, LSTM):
            layer.set_state(batch_size)

    # initialize optimizer
    grad_norm = GradientNorm(max_norm)
    decay = DecayEvery(decay_epoch * train_batches, decay_rate)
    optimizer = SGD(learning_rate, grad_norm, decay)

    # compile theano functions
    fit = compile_model_lasagne(model, (X_train, y_train), optimizer)
    val = compile_model_lasagne(model, (X_valid, y_valid))

    # train
    for epoch in range(1, epochs + 1):
        print 'epoch({})'.format(epoch)

        # fit
        perplexity_list = []
        accuracy_list = []
        train_progress = ProgressBar(range(train_batches))
        for batch in train_progress:
            perplexity, accuracy = fit(batch)
            perplexity_list.append(perplexity)
            accuracy_list.append(accuracy)
            train_progress.perplexity = np.mean(perplexity_list)
            train_progress.accuracy = np.mean(accuracy_list)

        # validate
        perplexity_list = []
        accuracy_list = []
        valid_progress = ProgressBar(range(valid_batches))
        for batch in valid_progress:
        perplexity, accuracy = val(batch)
            perplexity_list.append(perplexity)
            accuracy_list.append(accuracy)
            valid_progress.perplexity = np.mean(perplexity_list)
            valid_progress.accuracy = np.mean(accuracy_list)

        # reset lstm layers
        for layer in model.layers:
            if isinstance(layer, LSTM):
                layer.set_state(batch_size)