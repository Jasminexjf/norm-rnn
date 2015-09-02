import sys
import timeit
import numpy as np


class TrainProgressBar(object):

    def __init__(self, epoch, fit_batches, val_batches):
        self.epoch = epoch
        self.fit_size = 60 * fit_batches / (fit_batches + val_batches)
        self.val_size = 60 * val_batches / (fit_batches + val_batches)

        self.fit_batches = fit_batches
        self.val_batches = val_batches

        self.fit_done = 0
        self.val_done = 0
        self.fit_left = self.fit_size
        self.val_left = self.val_size

        self.fit_perplexity = 0
        self.val_perplexity = 0

        self.begin = timeit.default_timer()

    def __str__(self):
        return '{} [{}{} | {}{}] pp({:.2f} | {:.2f}) time({:.2f}s)\r'.format(
            self.epoch,
            '#' * self.fit_done, '.' * self.fit_left,
            '#' * self.val_done, '.' * self.val_left,
            self.fit_perplexity, self.val_perplexity,
            timeit.default_timer() - self.begin
        )

    def fit_update(self, fit_perplexity):
        self.fit_perplexity = np.mean(fit_perplexity)

        self.fit_done = int(self.fit_size * len(fit_perplexity) / self.fit_batches)
        self.fit_left = self.fit_size - self.fit_done

        sys.stdout.write(str(self))
        sys.stdout.flush()

    def val_update(self, val_perplexity):
        self.val_perplexity = np.mean(val_perplexity)

        self.val_done = int(self.val_size * len(val_perplexity) / self.val_batches)
        self.val_left = self.val_size - self.val_done

        sys.stdout.write(str(self))
        sys.stdout.flush()


import cPickle
import plotly.plotly as py
from plotly.graph_objs import *


def plot_saved_perplexity(file_names, model_names, save_file_name):
    data = []
    colors = ['rgb(44, 160, 44)',
              'rgb(214, 39, 40)']

    for file_name, model_name in zip(file_names, model_names):
        with open(file_name) as save_file:
            fit_perplexity, val_perplexity = cPickle.load(save_file)

            train = []
            for epoch in fit_perplexity:
                train.extend(epoch)

            valid = []
            for epoch in val_perplexity:
                valid.append(np.mean(epoch))

            color = colors.pop()

            # plot train results
            train_trace = Scatter(x=range(len(fit_perplexity)),
                             y=train,
                             name='{} (Train)'.format(model_name),
                             line=Line(color=color, width=1))

            step = len(train) / len(valid)

            # plot valid results
            valid_trace = Scatter(x=range(0, len(train), step),
                             y=valid,
                             name='{} (Valid)'.format(model_name),
                             line=Line(color=color, width=1, dash='dot'))

            # collect traces
            data.append(train_trace)
            data.append(valid_trace)

    # x and y labels
    layout = Layout(xaxis=XAxis(title='Epochs'),
                    yaxis=YAxis(title='Perplexity'),
                    showlegend=True,
                    legend=Legend(x=0.6, y=1))

    # save plot and open in browser
    fig = Figure(data=Data(data), layout=layout)
    py.image.save_as(fig, '{}.png'.format(save_file_name))
    print py.plot(data, filename='basic-line')


plot_saved_perplexity(['ptb_small_ref_results.pkl', 'ptb_small_norm_results.pkl'],
                      ['Without Batch Normalization', 'With Batch Normalization'],
                      'results')