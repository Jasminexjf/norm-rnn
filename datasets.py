import numpy as np


class PennTreebank(object):

    train_path = 'data/ptb.train.txt'
    valid_path = 'data/ptb.valid.txt'
    test_path = 'data/ptb.test.txt'

    def __init__(self, batch_size=20, time_steps=20, path=train_path,
                 vocab=None, tokenizer=str.split):
        text = open(path).read()

        if tokenizer is not None:
            # tokenizer = str.split gives words
            tokens = tokenizer(text)
        else:
            # tokenizer = None gives chars
            tokens = text

        # round off extra tokens
        extra_tokens = len(tokens) % (batch_size * time_steps)
        if extra_tokens:
            tokens = tokens[:-extra_tokens]

        # create vocab
        if vocab is None:
            vocab = sorted(set(tokens))

        self.token_to_index = dict((t, i) for i, t in enumerate(vocab))

        X = [self.token_to_index[t] for t in tokens]
        Y = X[1:] + X[-1:]

        # cast to int32 for gpu compatibility
        X = np.asarray(X, dtype=np.int32)
        Y = np.asarray(Y, dtype=np.int32)

        # reshape so that sentences are continuous across batches
        self.X = X.reshape(-1, batch_size, time_steps)
        self.Y = Y.reshape(-1, batch_size * time_steps)

        self.batch_size = batch_size
        self.time_steps = time_steps

    def __len__(self):
        # number of batches
        return len(self.X)

    @property
    def vocab(self):
        return sorted(self.token_to_index.keys())

    @property
    def index_to_token(self):
        return dict((v, k) for k, v in self.token_to_index.iteritems())
