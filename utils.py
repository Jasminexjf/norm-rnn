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
        self.fit_accuracy = 0
        self.val_accuracy = 0

        self.begin = timeit.default_timer()

    def __str__(self):
        return '{} [{}{} | {}{}] pp({:.2f} | {:.2f}) acc({:.2f}% | {:.2f}%) time({:.2f}s)\r'.format(
            self.epoch,
            '#' * self.fit_done, '.' * self.fit_left,
            '#' * self.val_done, '.' * self.val_left,
            self.fit_perplexity, self.val_perplexity,
            self.fit_accuracy, self.val_accuracy,
            timeit.default_timer() - self.begin
        )

    def fit_update(self, fit_results):
        perplexity_list, accuracy_list = zip(*fit_results)
        self.fit_perplexity = np.mean(perplexity_list)
        self.fit_accuracy = np.mean(accuracy_list)

        self.fit_done = int(self.fit_size * len(fit_results) / self.fit_batches)
        self.fit_left = self.fit_size - self.fit_done

        sys.stdout.write(str(self))
        sys.stdout.flush()

    def val_update(self, val_results):
        perplexity_list, accuracy_list = zip(*val_results)
        self.val_perplexity = np.mean(perplexity_list)
        self.val_accuracy = np.mean(accuracy_list)

        self.val_done = int(self.val_size * len(val_results) / self.val_batches)
        self.val_left = self.val_size - self.val_done

        sys.stdout.write(str(self))
        sys.stdout.flush()


import cPickle


def plot_saved_perplexity(file_names):
    for file_name in file_names:
        with open(file_name) as save_file:
            fit_results, val_results = cPickle.load(save_file)
            fit_perplexity, fit_accuracy = zip(*fit_results)
            val_perplexity, val_accuracy = zip(*val_results)

            print '{} (Train)'.format(file_name)
            for p in fit_perplexity:
                print p
            print '{} (Valid)'.format(file_name)
            for p in val_perplexity:
                print p
