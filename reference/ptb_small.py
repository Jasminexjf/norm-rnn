import numpy as np
np.random.seed(0)

from tests.test_dataset import PennTreebank
from utils import ProgressBar
from norm_rnn import *
from layers import *

# hyper parameters
layer_size = 200
time_steps = 20
batch_size = 20
max_norm = 5
decay_rate = 0.5
decay_epoch = 4
learning_rate = 1
epochs = 5

# load dataset
dataset = PennTreebank(batch_size, time_steps)
vocab_size = len(dataset.vocab_map)
train_batches = len(dataset.X_train) / batch_size
valid_batches = len(dataset.X_valid) / batch_size
test_batches = len(dataset.X_test) / batch_size

# config model
model_name = 'Reference'
model = List([
    Embed(vocab_size, layer_size),
    LSTM(layer_size, layer_size),
    LSTM(layer_size, layer_size),
    Linear(layer_size, vocab_size)
])

# initialize shared memory for lstm states
# (find a way to push this into the layer)
for layer in model.layers:
    if isinstance(layer, LSTM):
        layer.set_state(batch_size)

# initialize optimizer
grad = MaxNorm(max_norm)
decay = DecayEvery(decay_epoch * train_batches, decay_rate)
optimizer = SGD(learning_rate, grad, decay)

# compile theano functions
fit = compile_model_lasagne(model, (dataset.X_train, dataset.y_train), optimizer)
val = compile_model_lasagne(model, (dataset.X_valid, dataset.y_valid))

# function to compute epoch and return loss
# (put this in model?)
def apply_to_batches(f, batches):
    perplexity_list = []
    accuracy_list = []
    train_progress = ProgressBar(range(batches))
    for batch in train_progress:
        perplexity, accuracy = f(batch)
        perplexity_list.append(perplexity)
        accuracy_list.append(accuracy)
        train_progress.perplexity = np.mean(perplexity_list)
        train_progress.accuracy = np.mean(accuracy_list)

    # reset LSTM layer
    for layer in model.layers:
        if isinstance(layer, LSTM):
            layer.set_state(batch_size)

    return np.mean(perplexity_list)

# train
train_costs = []
valid_costs = []
for epoch in range(1, epochs + 1):
    print 'epoch({})'.format(epoch)
    train_costs.append(apply_to_batches(fit, train_batches))
    valid_costs.append(apply_to_batches(val, valid_batches))


print range(len(train_costs))
print train_costs

print 'test type(static)'
val = compile_model_lasagne(model, (dataset.X_test, dataset.y_test))
static_cost = apply_to_batches(val, test_batches)


print 'test type(dynamic)'
fit = compile_model_lasagne(model, (dataset.X_test, dataset.y_test), optimizer)
dynamic_cost = apply_to_batches(fit, test_batches)


# plot results
import sys
import plotly.plotly as py
from plotly.graph_objs import *

train_trace = Scatter(x=range(len(train_costs)),
                 y=train_costs,
                 name='{} (Train))'.format(model_name))

valid_trace = Scatter(x=range(len(valid_costs)),
                 y=valid_costs,
                 name='{} (Valid))'.format(model_name),
                 line=Line(dash='dot'))

layout = Layout(xaxis=XAxis(title='Epochs'),
                yaxis=YAxis(title='Perplexity'))

data = Data([train_trace, valid_trace])
fig = Figure(data=data, layout=layout)
py.image.save_as(fig, '{}.png'.format(sys.modules[__name__].__name__))
plot_url = py.plot(data, filename='basic-line')
