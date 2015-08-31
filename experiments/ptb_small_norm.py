from ptb_small_ref import (vocab_size, layer_size,
                           optimizer, dataset, epochs)
from model import List
from layers import *

# model
model = List([
    Embed(vocab_size, layer_size),
    BNLSTM(layer_size, layer_size),
    BNLSTM(layer_size, layer_size),
    Linear(layer_size, vocab_size)])


if __name__ == '__main__':
    # compile
    model.compile(dataset, optimizer)
    model.compile(dataset)

    # train
    model.train(epochs)
    model.dump('ptb_small_norm_results.pkl')
