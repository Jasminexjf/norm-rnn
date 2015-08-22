import sys
import timeit


class ProgressBar(object):

    def __init__(self, iterable, size=60):
        self.iterable = iterable
        self.size = size
        self.remaining = size
        self.progress = 0

    def __str__(self):
        return '[{}{}] cost({:.2f}) time({:.2f})\r'.format(
            '#' * self.progress, '.' * self.remaining, self.cost, self.elapsed)

    def __iter__(self):
        self.begin = timeit.default_timer()
        self.cost = 0
        self.elapsed = 0
        self.display(0)

        for i, item in enumerate(self.iterable, 1):
            yield item
            self.elapsed = timeit.default_timer() - self.begin
            self.display(i)
        print

    def display(self, current):
        self.progress = int(self.size * current / len(self.iterable))
        self.remaining = self.size - self.progress

        sys.stdout.write(str(self))
        sys.stdout.flush()