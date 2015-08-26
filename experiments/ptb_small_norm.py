import sys
import numpy as np
np.random.seed(0)

# lasagne loader (merge into datasets)
from tests.test_dataset import PennTreebank
from utils import ProgressBar
from norm_rnn import *
from layers import *

# try to import plotly
try:
    import plotly.plotly as py
    from plotly.graph_objs import *
    plotly_is_installed = True
except ImportError:
    print 'Run \'pip install plotly\' to get a dope plot after training'
    plotly_is_installed = False

# models params
layer_size = 200

# data params
time_steps = 20
batch_size = 20
epochs = 15

# optimizer params
learning_rate = 1
decay_rate = 0.5
decay_epoch = 4
max_norm = 15

# load data
dataset = PennTreebank(batch_size, time_steps)
vocab_size = len(dataset.vocab_map)
train_batches = len(dataset.X_train) / batch_size
valid_batches = len(dataset.X_valid) / batch_size
test_batches = len(dataset.X_test) / batch_size

# models
models = {
    'Without Batch Normalization':
        List([Embed(vocab_size, layer_size),
              LSTM(layer_size, layer_size),
              LSTM(layer_size, layer_size),
              Linear(layer_size, vocab_size)]),

    'With Batch Normalization':
        List([Embed(vocab_size, layer_size),
              BNLSTM(layer_size, layer_size),
              BNLSTM(layer_size, layer_size),
              Linear(layer_size, vocab_size)])
}

# optimizer
grad = MaxNorm()
decay = DecayEvery(decay_epoch * train_batches, decay_rate)
optimizer = SGD(learning_rate, grad, decay)


# epoch helper (move to model)
def apply_to_batches(f, batches):
    perplexity_list = []
    accuracy_list = []
    progress = ProgressBar(range(batches))
    for batch in progress:
        perplexity, accuracy = f(batch)
        perplexity_list.append(perplexity)
        accuracy_list.append(accuracy)
        progress.perplexity = np.mean(perplexity_list)
        progress.accuracy = np.mean(accuracy_list)

    # reset LSTM layer
    for layer in model.layers:
        if isinstance(layer, LSTM):
            layer.set_state(batch_size)

    return np.mean(perplexity_list)


# plotly data
# (move plotly logic to utils)
data = []
colors = ['rgb(31, 119, 180)',
          'rgb(255, 127, 14)']


# train each model
for model_name, model in models.iteritems():
    # set LSTM and BN states
    for layer in model.layers:
        if isinstance(layer, LSTM):
            layer.set_state(batch_size, time_steps)

    # compile
    fit = compile_model_lasagne(model, (dataset.X_train, dataset.y_train), optimizer)
    val = compile_model_lasagne(model, (dataset.X_valid, dataset.y_valid))

    print
    print 'model({})'.format(model_name)
    print

    # train
    train_costs = []
    valid_costs = []
    for epoch in range(1, epochs + 1):
        print 'epoch({})'.format(epoch)
        train_costs.append(apply_to_batches(fit, train_batches))
        valid_costs.append(apply_to_batches(val, valid_batches))

    # static test validation
    print 'test type(static)'
    val = compile_model_lasagne(model, (dataset.X_test, dataset.y_test))
    static_cost = apply_to_batches(val, test_batches)

    # dynamic test validation
    print 'test type(dynamic)'
    fit = compile_model_lasagne(model, (dataset.X_test, dataset.y_test), optimizer)
    dynamic_cost = apply_to_batches(fit, test_batches)

    if plotly_is_installed:
        color = colors.pop()

        # plot train results
        train_trace = Scatter(x=range(len(train_costs)),
                              y=train_costs,
                              name='{} (Train))'.format(model_name),
                              line=Line(color=color))

        # plot valid results
        valid_trace = Scatter(x=range(len(valid_costs)),
                              y=valid_costs,
                              name='{} (Valid))'.format(model_name),
                              line=Line(color=color, dash='dot'))

        # collect traces
        data.extend([train_trace, valid_trace])


if plotly_is_installed:
    # x and y labels
    layout = Layout(xaxis=XAxis(title='Epochs'),
                    yaxis=YAxis(title='Perplexity'))

    # file name
    from os.path import splitext, basename
    file_name = splitext(basename(__file__))[0]

    # save plot and open in browser
    fig = Figure(data=Data(data), layout=layout)
    py.image.save_as(fig, '{}.png'.format(file_name))
    plot_url = py.plot(data, filename='basic-line')
else:
    print 'No plot for you!'
