#### Batch Normalized Recurrent Neural Networks

Theano code for the Penn Treebank language model experiments in the paper [Batch Normalized Recurrent Neural Networks](http://arxiv.org/abs/1510.01378). The baseline LSTMs reproduce the results from the paper [Recurrent Neural Network Regularization](http://arxiv.org/pdf/1409.2329v4.pdf).

The name of the repo is, of course, based off of Karpathy's [char-rnn](https://github.com/karpathy/char-rnn).  

#### Requirements

**Theano** is required for running the experiments:

```python
pip install Theano
```

**Plotly** is optional and is used to generate plots after training:

```python
pip install plotly
```
 
#### Experiments

To run the small reference model on CPU:
```python
python experiments/ptb_small_ref.py
```

To run the small normalized model on GPU:
```python
THEANO_FLAGS=device=gpu python experiments/ptb_small_norm.py
```

#### References <a name="refs"></a>

1. Ioffe, S., & Szegedy, C. (2015). [Batch normalization: Accelerating deep network training by reducing internal covariate shift](http://arxiv.org/pdf/1502.03167v3.pdf).

2. Zaremba, W., Sutskever, I., & Vinyals, O. (2014). [Recurrent neural network regularization](http://arxiv.org/pdf/1409.2329v5.pdf).
