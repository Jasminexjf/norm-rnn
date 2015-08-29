from ptb_large_ref import (vocab_size, layer_size,
                           optimizer, dataset, epochs,
                           drop_prob, weight_init)
from model import List
from layers import *

# model
model = List([
    Embed(vocab_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    LSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    Linear(layer_size, vocab_size, weight_init=weight_init)])

# train
model.compile(dataset, optimizer)
model.compile(dataset)
model.train(epochs)