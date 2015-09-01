from datasets import *

# data
train_set = Text(ptb_train_path, 10, 10)
valid_set = Text(ptb_valid_path, 10, 10, train_set.vocab_map, train_set.vocab_idx)
test_set = Text(ptb_test_path, 10, 10, train_set.vocab_map, train_set.vocab_idx)

# check vocab size
vocab_size = 10000
assert len(train_set.vocab_map) == vocab_size
assert len(valid_set.vocab_map) == vocab_size
assert len(test_set.vocab_map) == vocab_size

# first 10 words of each split
desired_train = 'aer banknote berlitz calloway centrust cluett fromstein gitano guterman hydro-quebec'
desired_valid = 'consumers may want to move their telephones a little closer'
desired_test = 'no it was n\'t black monday <eos> but while the'

# check first 10 words loaded
reverse_vocab_map = {i: v for v, i in train_set.vocab_map.iteritems()}
actual_train = [reverse_vocab_map[i] for i in train_set.X[0][:10]]
actual_valid = [reverse_vocab_map[i] for i in valid_set.X[0][:10]]
actual_test = [reverse_vocab_map[i] for i in test_set.X[0][:10]]

assert ' '.join(actual_train) == desired_train
assert ' '.join(actual_valid) == desired_valid
assert ' '.join(actual_test) == desired_test


