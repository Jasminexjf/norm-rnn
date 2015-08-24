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
epochs = 15

# load dataset
dataset = LasagneLoader()
vocab_size = len(dataset.vocab_map)
train_batches = len(dataset.X_train) / batch_size
valid_batches = len(dataset.X_valid) / batch_size
test_batches = len(dataset.X_test) / batch_size

# config model
model = List([
    Embed(vocab_size, layer_size),
    LSTM(layer_size, layer_size),
    LSTM(layer_size, layer_size),
    Linear(layer_size, vocab_size)
])

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
fit = compile_model_lasagne(model, (dataset.X_train, dataset.y_train), optimizer)
val = compile_model_lasagne(model, (dataset.X_valid, dataset.y_valid))

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


# reset lstm layers
for layer in model.layers:
    if isinstance(layer, LSTM):
        layer.set_state(batch_size)

print 'test type(static)'
val = compile_model_lasagne(model, (dataset.X_test, dataset.y_test))

# static test validation
perplexity_list = []
accuracy_list = []
test_progress = ProgressBar(range(test_batches))
for batch in test_progress:
    perplexity, accuracy = val(batch)
    perplexity_list.append(perplexity)
    accuracy_list.append(accuracy)
    test_progress.perplexity = np.mean(perplexity_list)
    test_progress.accuracy = np.mean(accuracy_list)


# reset lstm layers
for layer in model.layers:
    if isinstance(layer, LSTM):
        layer.set_state(batch_size)


print 'test type(dynamic)'
fit = compile_model_lasagne(model, (dataset.X_test, dataset.y_test), optimizer)

# dynamic test validation
perplexity_list = []
accuracy_list = []
test_progress = ProgressBar(range(test_batches))
for batch in test_progress:
    perplexity, accuracy = val(batch)
    perplexity_list.append(perplexity)
    accuracy_list.append(accuracy)
    test_progress.perplexity = np.mean(perplexity_list)
    test_progress.accuracy = np.mean(accuracy_list)
