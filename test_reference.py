import numpy as np
np.random.seed(0)

from norm_rnn import ProgressBar
from norm_rnn import PennTreebank
from norm_rnn import List
from norm_rnn import compile_model
from layers import *

# load data
train_set = PennTreebank()

# config model
model = List([
    Embed(9998, 200),
    LSTM(200, 200),
    LSTM(200, 200),
    Linear(200, 9998)
])

# compile theano functions
fit = compile_model(model, train_set)

# train
for epoch in range(1, 11):
    print 'Epoch:', epoch

    # fix prog bar
    prog_bar = ProgressBar(range(len(train_set)))

    # fit
    cost_list = []
    for batch in prog_bar:
        cost_list.append(fit(batch))
        prog_bar.set_loss(np.mean(cost_list))
