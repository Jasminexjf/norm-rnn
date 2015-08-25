import theano
import theano.tensor as T
import numpy as np


class Uniform(object):

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, shape):
        return np.asarray(
            np.random.uniform(-self.scale, self.scale, shape), dtype=np.float32)


class Softmax(object):

    def __call__(self, x):
        return T.nnet.softmax(x.reshape((-1, x.shape[-1])))


class Identity(object):

    def __call__(self, x):
        return x


class Linear(object):

    def __init__(self, input_size, output_size, activation=Softmax(),
                 weight_init=Uniform()):
        self.W = theano.shared(weight_init((input_size, output_size)))
        self.b = theano.shared(weight_init(output_size))
        self.params = [self.W, self.b]

        self.activation = activation

    def __call__(self, x):
        return self.activation(T.dot(x, self.W) + self.b)


class Embed(object):

    def __init__(self, input_size, output_size, weight_init=Uniform()):
        self.W = theano.shared(weight_init((input_size, output_size)))
        self.params = [self.W]

    def __call__(self, x):
        return self.W[x]


class LSTM(object):

    # stripped down LSTM from Keras

    def __init__(self, input_size, layer_size, activation=T.tanh,
                 inner_activation=T.nnet.sigmoid, weight_init=Uniform()):
        W_shape = input_size, layer_size
        U_shape = layer_size, layer_size
        b_shape = layer_size

        self.W_i = theano.shared(weight_init(W_shape))
        self.U_i = theano.shared(weight_init(U_shape))
        self.b_i = theano.shared(weight_init(b_shape))

        self.W_f = theano.shared(weight_init(W_shape))
        self.U_f = theano.shared(weight_init(U_shape))
        self.b_f = theano.shared(weight_init(b_shape))

        self.W_c = theano.shared(weight_init(W_shape))
        self.U_c = theano.shared(weight_init(U_shape))
        self.b_c = theano.shared(weight_init(b_shape))

        self.W_o = theano.shared(weight_init(W_shape))
        self.U_o = theano.shared(weight_init(U_shape))
        self.b_o = theano.shared(weight_init(b_shape))

        self.params = [
            self.W_i, self.U_i, self.b_i,
            self.W_c, self.U_c, self.b_c,
            self.W_f, self.U_f, self.b_f,
            self.W_o, self.U_o, self.b_o,
        ]

        self.activation = activation
        self.inner_activation = inner_activation

        self.shared_state = False
        self.layer_size = layer_size

    def __call__(self, x):
        x = x.dimshuffle((1, 0, 2))

        xi = T.dot(x, self.W_i) + self.b_i
        xf = T.dot(x, self.W_f) + self.b_f
        xc = T.dot(x, self.W_c) + self.b_c
        xo = T.dot(x, self.W_o) + self.b_o

        # if set_state hasn't been called, use temporary zeros (stateless)
        if not self.shared_state:
            self.h = self._alloc_zeros_matrix(x.shape[1], xi.shape[2])
            self.c = self._alloc_zeros_matrix(x.shape[1], xi.shape[2])

        [outputs, memories], updates = theano.scan(self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[self.h, self.c],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
        )

        if self.shared_state:
            self.updates = [(self.h, outputs[-1]), (self.c, memories[-1])]

        return outputs.dimshuffle((1, 0, 2))

    def _step(self,
        xi_t, xf_t, xo_t, xc_t,
        h_tm1, c_tm1,
        u_i, u_f, u_o, u_c):

        i_t = self.inner_activation(xi_t + T.dot(h_tm1, u_i))
        f_t = self.inner_activation(xf_t + T.dot(h_tm1, u_f))
        c_t = f_t * c_tm1 + i_t * self.activation(xc_t + T.dot(h_tm1, u_c))
        o_t = self.inner_activation(xo_t + T.dot(h_tm1, u_o))
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def _alloc_zeros_matrix(self, *dims):
        return T.alloc(np.cast[theano.config.floatX](0.), *dims)

    def set_state(self, batch_size):
        # make hidden and cells a shared variable so that state is persistent
        self.h = theano.shared(np.zeros((batch_size, self.layer_size), dtype=theano.config.floatX))
        self.c = theano.shared(np.zeros((batch_size, self.layer_size), dtype=theano.config.floatX))
        self.shared_state = True


class BN(object):

    def __init__(self, input_size, epsilon=1e-6):
        self.gamma = theano.shared(np.ones(input_size, dtype=np.float32))
        self.beta = theano.shared(np.zeros(input_size, dtype=np.float32))
        self.params = [self.gamma, self.beta]

        self.epsilon = epsilon

    def __call__(self, x):
        m = x.mean(axis=0)
        std = std = T.mean((x - m) ** 2 + self.epsilon, axis=0) ** 0.5
        x = (x - m) / (std + self.epsilon)
        return self.gamma * x + self.beta

class Dropout(object):

    def __init__(self, p):
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        self.srng = RandomStreams(seed=np.random.randint(10e6))
        self.p = p

    def __call__(self, x):
            return x * self.srng.binomial(x.shape, p=1-self.p, dtype=theano.config.floatX)


class BNLSTM(LSTM):

    def __init__(self, input_size, layer_size, activation=T.tanh,
                 inner_activation=T.nnet.sigmoid, weight_init=Uniform()):
        super(BNLSTM, self).__init__(input_size, layer_size, activation,
                 inner_activation, weight_init)

        # remove biases
        self.params.pop(2)
        self.params.pop(4)
        self.params.pop(6)
        self.params.pop(8)

        # add batch norm layers
        self.norm_xi = BN(layer_size)
        self.norm_xf = BN(layer_size)
        self.norm_xc = BN(layer_size)
        self.norm_xo = BN(layer_size)

        self.params.extend(self.norm_xi.params + self.norm_xf.params +
                           self.norm_xc.params + self.norm_xo.params)

    def __call__(self, x):
        xi = self.norm_xi(T.dot(x, self.W_i))
        xf = self.norm_xf(T.dot(x, self.W_f))
        xc = self.norm_xc(T.dot(x, self.W_c))
        xo = self.norm_xo(T.dot(x, self.W_o))

        # (time_steps, batch_size, layer_size)
        x = x.dimshuffle((1, 0, 2))

        # if set_state hasn't been called, use temporary zeros (stateless)
        if not self.shared_state:
            self.h = self._alloc_zeros_matrix(x.shape[1], xi.shape[2])
            self.c = self._alloc_zeros_matrix(x.shape[1], xi.shape[2])

        [outputs, memories], updates = theano.scan(self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[self.h, self.c],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
        )

        if self.shared_state:
            self.updates = [(self.h, outputs[-1]), (self.c, memories[-1])]

        return outputs.dimshuffle((1, 0, 2))
