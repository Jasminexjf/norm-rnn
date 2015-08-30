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


if __name__ == '__main__':
    # compile
    model.compile(dataset, optimizer)
    model.compile(dataset)

    # train
    model.train(epochs)
    model.dump('ptb_large_norm_results.pkl')