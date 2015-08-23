import numpy as np
np.random.seed(0)

from datasets import PennTreebank
from norm_rnn import List
from norm_rnn import compile_model
from norm_rnn import SGD, DecayEvery
from utils import ProgressBar
from layers import *

# load data
train_set = PennTreebank(time_steps=35)
valid_set = PennTreebank(time_steps=35,
                         path=PennTreebank.valid_path,
                         vocab=train_set.vocab)

# hyper parameters
vocab_size = len(train_set.vocab)
layer_size = 1500
drop_prob = 0.65

# config model
model = List([
    Embed(vocab_size, layer_size),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size),
    Dropout(drop_prob),
    Linear(layer_size, vocab_size)
])

# initialize shared memory for lstm states
# (need to find a way to push this into the layer)
for layer in model.layers:
    if isinstance(layer, LSTM):
        layer.set_state(train_set.batch_size)

# initialize optimizer
decay = DecayEvery(14 * len(train_set), 1/1.15)
sgd = SGD(lr=1, grad_norm=10, decay=decay)

# compile theano functions
fit = compile_model(model, train_set, sgd)

def not_dropout(layer): return not isinstance(layer, Dropout)

model.layers = filter(not_dropout, model.layers)
val = compile_model(model, valid_set, None)

# train
for epoch in range(1, 15):
    print 'epoch({})'.format(epoch)

    # fit
    train_errors = []
    train_progress = ProgressBar(range(len(train_set)))
    for batch in train_progress:
        error = fit(batch)
        train_errors.append(error)
        train_progress.cost = np.mean(train_errors)

    # validate
    valid_errors = []
    valid_progress = ProgressBar(range(len(valid_set)))
    for batch in valid_progress:
        error = val(batch)
        valid_errors.append(error)
        valid_progress.cost = np.mean(valid_errors)
