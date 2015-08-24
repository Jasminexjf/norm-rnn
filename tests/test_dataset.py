import gzip
import numpy as np


# lasagne
def load_data(file_name, vocab_map, vocab_idx):
    """
    Loads Penn Tree files downloaded from https://github.com/wojzaremba/lstm
    Parameters
    ----------
    file_name : str
        Path to Penn tree file.
    vocab_map : dictionary
        Dictionary mapping words to integers
    vocab_idx : one element list
        Current vocabulary index.
    Returns
    -------
    Returns an array with each words specified in file_name as integers.
    Note that the function has the side effects that vocab_map and vocab_idx
    are updated.
    Notes
    -----
    This is python port of the LUA function load_data in
    https://github.com/wojzaremba/lstm/blob/master/data.lua
    """
    def process_line(line):
        line = line.lstrip()
        line = line.replace('\n', '<eos>')
        words = line.split(" ")
        if words[-1] == "":
            del words[-1]
        return words

    words = []
    with gzip.open(file_name, 'rb') as f:
        for line in f.readlines():
            words += process_line(line)

    n_words = len(words)
    print("Loaded %i words from %s" % (n_words, file_name))

    x = np.zeros(n_words)
    for wrd_idx, wrd in enumerate(words):
        if wrd not in vocab_map:
            vocab_map[wrd] = vocab_idx[0]
            vocab_idx[0] += 1
        x[wrd_idx] = vocab_map[wrd]
    return x.astype('int32')


# lasagne
def reorder(x_in, batch_size, model_seq_len):
    """
    Rearranges data set so batches process sequential data.
    If we have the dataset:
    x_in = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    and the batch size is 2 and the model_seq_len is 3. Then the dataset is
    reordered such that:
                   Batch 1    Batch 2
                 ------------------------
    batch pos 1  [1, 2, 3]   [4, 5, 6]
    batch pos 2  [7, 8, 9]   [10, 11, 12]
    This ensures that we use the last hidden state of batch 1 to initialize
    batch 2.
    Also creates targets. In language modelling the target is to predict the
    next word in the sequence.
    Parameters
    ----------
    x_in : 1D numpy.array
    batch_size : int
    model_seq_len : int
        number of steps the model is unrolled
    Returns
    -------
    reordered x_in and reordered targets. Targets are shifted version of x_in.
    """
    if x_in.ndim != 1:
        raise ValueError("Data must be 1D, was", x_in.ndim)

    if x_in.shape[0] % (batch_size*model_seq_len) == 0:
        print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
              "set to x_in = x_in[:-1]")
        x_in = x_in[:-1]

    x_resize =  \
        (x_in.shape[0] // (batch_size*model_seq_len))*model_seq_len*batch_size
    n_samples = x_resize // (model_seq_len)
    n_batches = n_samples // batch_size

    targets = x_in[1:x_resize+1].reshape(n_samples, model_seq_len)
    x_out = x_in[:x_resize].reshape(n_samples, model_seq_len)

    out = np.zeros(n_samples, dtype=int)
    for i in range(n_batches):
        val = range(i, n_batches*batch_size+i, n_batches)
        out[i*batch_size:(i+1)*batch_size] = val

    x_out = x_out[out]
    targets = targets[out]

    return x_out.astype('int32'), targets.astype('int32')


# lasagne
def traindata(model_seq_len, batch_size, vocab_map, vocab_idx):
    x = load_data("ptb.train.txt.gz", vocab_map, vocab_idx)
    return reorder(x, batch_size, model_seq_len)


# lasagne
def validdata(model_seq_len, batch_size, vocab_map, vocab_idx):
    x = load_data("ptb.valid.txt.gz", vocab_map, vocab_idx)
    return reorder(x, batch_size, model_seq_len)


# lasagne
def testdata(model_seq_len, batch_size, vocab_map, vocab_idx):
    x = load_data("ptb.test.txt.gz", vocab_map, vocab_idx)
    return reorder(x, batch_size, model_seq_len)


# lasagne
class LasagneLoader():

    def __init__(self):
        self.vocab_map = {}
        self.vocab_idx = [0]

        self.X_train, self.y_train = traindata(20, 20, self.vocab_map, self.vocab_idx)
        self.X_valid, self.y_valid = validdata(20, 20, self.vocab_map, self.vocab_idx)
        self.X_test, self.y_test = testdata(20, 20, self.vocab_map, self.vocab_idx)


def test_penn_treebank():
    from datasets import PennTreebank
    ptb = PennTreebank(path='../data/ptb.train.txt')

    print ptb.X[0][0]
