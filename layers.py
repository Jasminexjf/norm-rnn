import theano
import theano.tensor as T
import numpy as np


class Uniform(object):

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, shape):
        return np.asarray(
            np.random.uniform(-self.scale, self.scale, shape), dtype=np.float32)


class LSTM(object):

    # stripped down LSTM from Keras

    def __init__(self, input_size, output_size, activation=T.tanh,
                 inner_activation=T.nnet.softmax, weight_init=Uniform()):
        W_shape = input_size, output_size
        U_shape = output_size, output_size
        b_shape = output_size

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

    def __call__(self, x):
        x = x.dimshuffle((1, 0, 2))

        xi = T.dot(x, self.W_i) + self.b_i
        xf = T.dot(x, self.W_f) + self.b_f
        xc = T.dot(x, self.W_c) + self.b_c
        xo = T.dot(x, self.W_o) + self.b_o

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(self._alloc_zeros_matrix(x.shape[1], xi.shape[2]), 1),
                T.unbroadcast(self._alloc_zeros_matrix(x.shape[1], xi.shape[2]), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
        )

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


class NormalizedLSTM(LSTM):

    def __init__(self, input_size, output_size, activation=T.tanh,
                 inner_activation=T.nnet.softmax, weight_init=Uniform()):
        super(NormalizedLSTM, self).__init__(input_size, output_size, activation, inner_activation, weight_init)

        # input-to-hidden batch-normalization
        self.i_norm = BatchNormalization(output_size, weight_init)
        self.f_norm = BatchNormalization(output_size, weight_init)
        self.c_norm = BatchNormalization(output_size, weight_init)
        self.o_norm = BatchNormalization(output_size, weight_init)

        def is_matrix(p):
            return p.ndim == 2

        # remove biases
        self.params = filter(is_matrix, self.params)

        # add batch-norm params
        self.params.extend(self.i_norm.params + self.f_norm.params +
                           self.c_norm.params + self.o_norm.params)

    def __call__(self, x):
        x = x.dimshuffle((1, 0, 2))

        # apply batch-norm after input-to-hidden dot product
        xi = self.i_norm(T.dot(x, self.W_i))
        xf = self.f_norm(T.dot(x, self.W_f))
        xc = self.c_norm(T.dot(x, self.W_c))
        xo = self.o_norm(T.dot(x, self.W_o))

        [outputs, memories], updates = theano.scan(
            self._step,
            sequences=[xi, xf, xo, xc],
            outputs_info=[
                T.unbroadcast(self._alloc_zeros_matrix(x.shape[1], xi.shape[2]), 1),
                T.unbroadcast(self._alloc_zeros_matrix(x.shape[1], xi.shape[2]), 1)
            ],
            non_sequences=[self.U_i, self.U_f, self.U_o, self.U_c],
        )

        return outputs.dimshuffle((1, 0, 2))


class BatchNormalization():

    def __init__(self, input_size, weight_init=Uniform, epsilon=1e-6):
        self.gamma = theano.shared(weight_init(input_size))
        self.beta = theano.shared(weight_init(input_size))
        self.params = [self.gamma, self.beta]

        self.epsilon = epsilon

    def __call__(self, x):
        # axis 1 is batch since x is dimshuffled in LSTM
        m = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        x = (x - m) / (std + self.epsilon)
        return self.gamma * x + self.beta


class Softmax(object):

    def __call__(self, x):
        return T.nnet.softmax(x.reshape((-1, x.shape[-1])))


class Linear(object):

    def __init__(self, input_size, output_size, activation=Softmax(),
                 weight_init=Uniform()):
        self.W = theano.shared(weight_init((input_size, output_size)))
        self.b = theano.shared(weight_init(output_size))
        self.params = [self.W, self.b]

        self.activation = activation

    def __call__(self, x):
        return self.activation(T.dot(x, self.W) + self.b)


