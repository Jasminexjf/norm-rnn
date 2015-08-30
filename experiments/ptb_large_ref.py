import numpy as np
np.random.seed(0)

from datasets import PennTreebank
from optimizers import *
from model import *
from layers import *

# models params
init_range = 0.04
layer_size = 1500
drop_prob = 0.65

# data params
time_steps = 35
batch_size = 20
epochs = 50

# optimizer params
learning_rate = 1
decay_rate = 1 / 1.15
decay_epoch = 14
max_norm = 10

# data
dataset = PennTreebank(batch_size, time_steps)
vocab_size = len(dataset.vocab_map)

# weight init
weight_init = Uniform(init_range)

# model
model = List([
    Embed(vocab_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    Linear(layer_size, vocab_size, weight_init=weight_init)])

# optimizer
grad = MaxNorm()
decay = DecayEvery(decay_epoch * len(dataset.X_train.get_value()), decay_rate)
optimizer = SGD(learning_rate, grad, decay)


if __name__ == '__main__':
    # compile
    model.compile(dataset, optimizer)
    model.compile(dataset)

    # train
    model.train(epochs)
    model.dump('ptb_large_ref_results.pkl')