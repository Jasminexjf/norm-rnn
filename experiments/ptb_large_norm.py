from ptb_large_ref import (vocab_size, layer_size, drop_prob, weight_init, decay_epoch, decay_rate,
                           optimizer, train_set, valid_set, test_set, epochs)
from model import List
from layers import *

# model
model = List([
    Embed(vocab_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    BNLSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    BNLSTM(layer_size, layer_size, weight_init=weight_init),
    Dropout(drop_prob),
    Linear(layer_size, vocab_size, weight_init=weight_init)])

model.decay_epoch = decay_epoch
model.decay_rate = decay_rate

if __name__ == '__main__':
    model.train(train_set, valid_set, test_set, optimizer, epochs)
    model.dump('ptb_large_norm_results.pkl')
