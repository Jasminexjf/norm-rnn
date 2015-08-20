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

# progress bars
train_set = ProgressBar(train_set)
valid_set = ProgressBar(valid_set)
test_set = ProgressBar(test_set)

# initialize models
models = {

    'Reference LSTM':
    List([
        LSTM(9998, 200),
        LSTM(200, 200),
        Linear(200, 9998)
    ]),

    'Input batch normalized LSTM':
    List([
        LSTM(9998, 200),
        BatchNormalization(200),  # normalizes over time
        LSTM(200, 200),
        BatchNormalization(200),
        Linear(200, 9998)
    ]),

    'Input-to-hidden batch normalized LSTM':
    List([
        NormalizedLSTM(9998, 200),
        NormalizedLSTM(200, 200),
        Linear(200, 9998)
    ]),

    'Hidden-to-hidden batch normalized LSTM':
    List([
        HiddenNormalizedLSTM(9998, 200),
        HiddenNormalizedLSTM(200, 200),
        Linear(200, 9998)
    ]),

    'Pre-activation batch normalized LSTM':
    List([
        PreActNormalizedLSTM(9998, 200),
        PreActNormalizedLSTM(200, 200),
        Linear(200, 9998)
    ]),

    'Input-to-hidden non-gate batch normalized LSTM':
    List([
        NonGateNormalizedLSTM(9998, 200),
        NonGateNormalizedLSTM(200, 200),
        Linear(200, 9998)
    ]),

    'Hidden-to-hidden non-gate batch normalized LSTM':
    List([
        NonGateHiddenNormalizedLSTM(9998, 200),
        NonGateHiddenNormalizedLSTM(200, 200),
        Linear(200, 9998)
    ])
}


for model_name, model in models.iteritems():

    # compile theano functions
    fit = compile_model(model)
    validate = compile_model(model, update=False)

    print 'Model: {}'.format(model_name)

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
