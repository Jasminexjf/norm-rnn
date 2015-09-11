## Normalized Recurrent Neural Networks

#### Abstract

Theano implementation of Normalized Recurrent Neural Networks (link to paper).

| Model  | Valid PP  | Test PP | Iterations | Iterations (BN) |
| :----- | :-------: | :-----: | ---------: | --------------: |
| Small  | 120  	 | 114 	   | 10K 		| 5K			  |
| Medium | 86  	 	 | 82 	   | 30K 		| 10K			  |
| Large  | 82  	 	 | 78 	   | 40K 		| 15K			  |

(The numbers in the table are just place holders)

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
