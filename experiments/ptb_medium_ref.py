import numpy as np
np.random.seed(0)

from datasets import *
from optimizers import *
from model import *
from layers import *

# data params
time_steps = 35
batch_size = 32

# models params
layer_size = 650
drop_prob = 0.5
init_range = 0.05

# optimizer params
learning_rate = 1
decay_rate = 1 / 1.2
decay_epoch = 6
max_norm = 5
epochs = 50

# data
train_set = Text(ptb_train_path, batch_size, time_steps)
valid_set = Text(ptb_valid_path, batch_size, time_steps,
                 vocab_map=train_set.vocab_map, vocab_index=train_set.vocab_idx)
test_set = Text(ptb_test_path, batch_size, time_steps,
                 vocab_map=train_set.vocab_map, vocab_index=train_set.vocab_idx)
vocab_size = len(train_set.vocab_map)

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
    Linear(layer_size, vocab_size, weight_init=weight_init)
])

model.decay_epoch = decay_epoch
model.decay_rate = decay_rate

# optimizer
grad = MaxNorm(max_norm)
#decay = DecayEvery(decay_epoch * train_set.batches, decay_rate)
decay = None
optimizer = SGD(learning_rate, grad, decay)


if __name__ == '__main__':
    model.train(train_set, valid_set, test_set, optimizer, epochs)
    model.dump('ptb_medium_ref_results.pkl')
