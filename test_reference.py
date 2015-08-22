import numpy as np
np.random.seed(0)

from datasets import PennTreebank
from norm_rnn import List
from norm_rnn import compile_model
from utils import ProgressBar
from layers import *

# load data
train_set = PennTreebank()
valid_set = PennTreebank(path=PennTreebank.valid_path, vocab=train_set.vocab)

# config model
model = List([
    Embed(9998, 200),
    LSTM(200, 200),
    Linear(200, 9998)
])

# compile theano functions
fit = compile_model(model, train_set, update=True)
val = compile_model(model, valid_set, update=False)

# train
for epoch in range(1, 10):
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