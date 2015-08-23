import numpy as np
np.random.seed(0)

from datasets import PennTreebank
from utils import ProgressBar
from norm_rnn import *
from layers import *

# hyper parameters
layer_size = 1500
time_steps = 35
batch_size = 20
learning_rate = 1
decay_rate = 0.5
max_norm = 10
decay_epoch = 14
epochs = 55
drop_prob = 0.65
init_range = 0.04

# load data
train_set = PennTreebank(batch_size, time_steps)
valid_set = PennTreebank(batch_size, time_steps,
                         PennTreebank.valid_path, train_set.vocab)

# weight init
weight_init = Uniform(init_range)

# config model
model = List([
    Embed(len(train_set.vocab), layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    Linear(layer_size, len(train_set.vocab), weight_init=weight_init)
])

# initialize shared memory for lstm states
# (need to find a way to push this into the layer)
for layer in model.layers:
    if isinstance(layer, LSTM):
        layer.set_state(train_set.batch_size)

# initialize optimizer
grad_norm = GradientNorm(max_norm)
decay = DecayEvery(decay_epoch * len(train_set), decay_rate)
optimizer = SGD(learning_rate, grad_norm, decay)

# compile theano functions
fit = compile_model(model, train_set, optimizer)

# strip out drop before compiling valid function (so lazy)
def not_dropout(layer): return not isinstance(layer, Dropout)
model.layers = filter(not_dropout, model.layers)

val = compile_model(model, valid_set)

# train
for epoch in range(1, epochs + 1):
    print 'epoch({})'.format(epoch)

    # fit
    perplexity_list = []
    accuracy_list = []
    train_progress = ProgressBar(range(len(train_set)))
    for batch in train_progress:
        perplexity, accuracy = fit(batch)
        perplexity_list.append(perplexity)
        accuracy_list.append(accuracy)
        train_progress.perplexity = np.mean(perplexity_list)
        train_progress.accuracy = np.mean(accuracy_list)

    # validate
    perplexity_list = []
    accuracy_list = []
    valid_progress = ProgressBar(range(len(valid_set)))
    for batch in valid_progress:
        perplexity, accuracy = val(batch)
        accuracy_list.append(accuracy)
        valid_progress.perplexity = np.mean(perplexity_list)
        valid_progress.accuracy = np.mean(accuracy_list)

    # reset lstm layers
    for layer in model.layers:
        if isinstance(layer, LSTM):
            layer.set_state(train_set.batch_size)
