import theano
import numpy as np

batch_size = 2
time_steps = 3
input_size = 4
layer_size = 5
batchs_per_epoch = 6

class Dataset(object):
    X_train = theano.shared(np.ones((batch_size * batchs_per_epoch, time_steps), np.int32))
    y_train = theano.shared(np.ones((batch_size * batchs_per_epoch, time_steps), np.int32))

    X_valid = theano.shared(np.ones((batch_size * batchs_per_epoch, time_steps), np.int32))
    y_valid = theano.shared(np.ones((batch_size * batchs_per_epoch, time_steps), np.int32))

    time_steps = time_steps
    batch_size = batch_size


def test_train():
    from model import List
    from layers import Embed, Softmax
    from optimizers import SGD
    model = List([Embed(input_size, layer_size),
                  Softmax()])
    model.compile(Dataset(), SGD())
    model.compile(Dataset(),)
    model.train(10)
    model.dump('test2.pkl')


test_train()