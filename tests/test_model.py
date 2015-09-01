import theano
import numpy as np

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
    from model import List
    from layers import Embed, Softmax
    from optimizers import SGD
    model = List([Embed(input_size, layer_size),
                  Softmax()])
    model.train(Dataset(), Dataset(), SGD(), 10)
    model.dump('test.pkl')


test_train()