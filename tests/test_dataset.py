import gzip
import numpy as np


def test_penn_treebank():
    from datasets import PennTreebank
    ptb = PennTreebank(path='../data/ptb.train.txt')

    print ptb.X[0][0]


def split_hutter_prize():
    with open('enwik8') as text_file:
        text = text_file.read()

    valid_start = int(0.9 * len(text))
    valid_end = int(0.94 * len(text))

    train = text[:valid_start]
    valid = text[valid_start:valid_end]
    test = text[valid_end:]

    for split_name, data in zip(['train', 'valid', 'test'],
                                [train, valid, test]):
        print len(data)
        with open('enwik8_{}'.format(split_name), 'w') as text_file:
            text_file.write(data)
