import numpy as np
np.random.seed(0)

from datasets import PennTreebank
from utils import ProgressBar
from norm_rnn import *
from layers import *

# load data
train_set = PennTreebank()
valid_set = PennTreebank(path=PennTreebank.valid_path, vocab=train_set.vocab)

# hyper parameters
vocab_size = len(train_set.vocab)
layer_size = 200

# config model
model = List([
    Embed(vocab_size, layer_size),
    LSTM(layer_size, layer_size),
    LSTM(layer_size, layer_size),
    Linear(layer_size, vocab_size)
])

# initialize shared memory for lstm states
# (need to find a way to push this into the layer)
for layer in model.layers:
    if isinstance(layer, LSTM):
        layer.set_state(train_set.batch_size)

# initialize optimizer
grad_norm = GradientNorm(max_norm=5)
decay = DecayEvery(4 * len(train_set), 0.5)
sgd = SGD(lr=1, grad_norm=grad_norm, decay=decay)

# compile theano functions
fit = compile_model(model, train_set, sgd)
val = compile_model(model, valid_set)

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
