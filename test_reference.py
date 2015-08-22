import numpy as np
np.random.seed(0)

from datasets import PennTreebank
from norm_rnn import List
from norm_rnn import compile_model
from layers import *

# load data
train_set = PennTreebank()

# config model
model = List([
    Embed(9998, 200),
    LSTM(200, 200),
    Linear(200, 9998)
])

# compile theano functions
fit = compile_model(model, train_set)

# train
for epoch in range(1, 11):
    print 'Epoch:', epoch

    # fit
    cost_list = []
    for batch in train_set:
        print fit(batch)