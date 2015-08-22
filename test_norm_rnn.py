def test_penn_treebank():
    from norm_rnn import PennTreebank
    ptb = PennTreebank()

    x1, y1 = ptb.__iter__().next()
    x2, y2 = ptb.__iter__().next()