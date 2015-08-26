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
init_range = 0.04
layer_size = 1500
drop_prob = 0.65

# data params
time_steps = 32
batch_size = 20
epochs = 55

# optimizer params
learning_rate = 1
decay_rate = 1 / 1.15
decay_epoch = 14
max_norm = 10

# load data
dataset = PennTreebank()
vocab_size = len(dataset.vocab_map)
train_batches = len(dataset.X_train) / batch_size
valid_batches = len(dataset.X_valid) / batch_size
test_batches = len(dataset.X_test) / batch_size

# weight init
weight_init = Uniform(init_range)

# models
models = {
    'Without Batch Normalization':
        List([Embed(vocab_size, layer_size, weight_init=weight_init),
              Dropout(drop_prob),
              LSTM(layer_size, layer_size, weight_init=weight_init),
              Dropout(drop_prob),
              LSTM(layer_size, layer_size, weight_init=weight_init),
              Linear(layer_size, vocab_size, Identity(), weight_init=weight_init),
              Dropout(drop_prob),
              Softmax()]),

    'With Batch Normalization':
        List([Embed(vocab_size, layer_size, weight_init=weight_init),
              BNLSTM(layer_size, layer_size, weight_init=weight_init),
              BNLSTM(layer_size, layer_size, weight_init=weight_init),
              Linear(layer_size, vocab_size, weight_init=weight_init)])
}

# optimizer
clip = MaxNorm(max_norm)
decay = DecayEvery(decay_epoch * train_batches, decay_rate)
optimizer = SGD(grad=clip, decay=decay)


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
data = []
colors = ['rgb(31, 119, 180)',
          'rgb(255, 127, 14)']


# train each model
for model_name, model in models.iteritems():
    # set LSTM states
    for layer in model.layers:
        if isinstance(layer, LSTM):
            layer.set_state(batch_size)

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
                         color=Line(color=color))

        # plot valid results
        valid_trace = Scatter(x=range(len(valid_costs)),
                         y=valid_costs,
                         name='{} (Valid))'.format(model_name),
                         line=Line(color=color, dash='dot'))

        # collect traces
        data.append(train_trace)
        data.append(valid_trace)


if plotly_is_installed:
    # x and y labels
    layout = Layout(xaxis=XAxis(title='Epochs'),
                    yaxis=YAxis(title='Perplexity'))

    # save plot and open in browser
    fig = Figure(data=Data(data), layout=layout)
    py.image.save_as(fig, '{}.png'.format(sys.modules[__name__].__name__))
    plot_url = py.plot(data, filename='basic-line')
else:
    print 'No plot for you!'
