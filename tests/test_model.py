import numpy as np

from model import List
from layers import Embed, Softmax
from optimizers import SGD

batch_size = 2
time_steps = 3
input_size = 4
layer_size = 5
batches = 6


class Dataset(object):
    X = np.ones((batch_size * batches, time_steps), np.int32)
    y = np.ones((batch_size * batches, time_steps), np.int32)

    time_steps = time_steps
    batch_size = batch_size
    batches = batches


def test_train():
    model = List([Embed(input_size, layer_size),
                  Softmax()])
    model.train(Dataset(), Dataset(), SGD(), 10)
    model.dump('test.pkl')


def test_pickle():
    model = List([Embed(input_size, layer_size),
              Softmax()])
    model.pickle()

    from utils import unpickle_model
    unpickle_model('model.pkl')
    

test_pickle()