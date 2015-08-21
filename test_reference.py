import numpy as np
np.random.seed(0)

from norm_rnn import ProgressBar
from norm_rnn import PennTreebank
from norm_rnn import List
from norm_rnn import compile_model
from layers import *

# load data
train_set = PennTreebank()
valid_set = PennTreebank(path=PennTreebank.valid_path, vocab=train_set.vocab)
test_set = PennTreebank(path=PennTreebank.test_path, vocab=train_set.vocab)

batch_size = train_set.batch_size

# progress bars
# make progress bars a decorator in order to
# be able to access underlying function
train_set = ProgressBar(train_set)
valid_set = ProgressBar(valid_set)
test_set = ProgressBar(test_set)

# config model
model = List([
    LSTM(9998, 200),
    LSTM(200, 200),
    Linear(200, 9998)
])

# set LSTM states
state = np.zeros([batch_size, 200])
model.layers[0].set_state(state, state)
model.layers[0].set_state(state, state)


# compile theano functions
fit = compile_model(model)
validate = compile_model(model, update=False)

# train
for epoch in range(1, 11):
    print 'Epoch:', epoch

    # fit
    cost_list = []
    for X, y in train_set:
        cost_list.append(fit(X, y))
        train_set.set_loss(np.mean(cost_list))

    # validate
    cost_list = []
    for X, y in valid_set:
        cost_list.append(validate(X, y))
        valid_set.set_loss(np.mean(cost_list))

# static evaluation
cost_list = []
for X, y in test_set:
    cost_list.append(validate(X, y))
    test_set.set_loss(np.mean(cost_list))

# dynamic evaluation
cost_list = []
for X, y in test_set:
    cost_list.append(fit(X, y))
    test_set.set_loss(np.mean(cost_list))
