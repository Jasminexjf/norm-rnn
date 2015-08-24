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

    def __init__(self, input_size, batch_size=None, time_steps=None, momentum=0.9, epsilon=1e-6):
        self.gamma = theano.shared(np.ones(input_size, dtype=np.float32))
        self.beta = theano.shared(np.zeros(input_size, dtype=np.float32))
        self.params = [self.gamma, self.beta]

        # if dimensions are None, we normalize over them
        # the resuling mean shape is all dimensions that were specified
        # (e.g. to get standard batch-norm on a sequence, we need to
        #  specify time-steps since our shared running averages needs this
        #  to be initialized.)
        running_shape = ()
        self.axes = ()
        if batch_size is not None:
            running_shape += (batch_size, )
        else:
            self.axes += (0, )

        if time_steps is not None:
            running_shape += (time_steps, )
        else:
            self.axes += (1, )

        running_shape += (input_size, )

        self.running_shape = running_shape
        self.reset_state()

        self.momentum = momentum
        self.epsilon = epsilon

    def __call__(self, x):
        # mini-batch statistics
        m = x.mean(axis=self.axes)
        std = std = T.mean((x - m) ** 2 + self.epsilon, axis=self.axes) ** 0.5

        # updates for shared state
        mean_update = self.momentum * self.running_mean + (1-self.momentum) * m
        std_update = self.momentum * self.running_std + (1-self.momentum) * std
        self.updates = [(self.running_mean, mean_update), (self.running_std, std_update)]

        # normalize
        x = (x - self.running_mean) / (self.running_std + self.epsilon)
        return self.gamma * x + self.beta

    def reset_state(self):
        self.running_mean = theano.shared(np.zeros(self.running_shape, np.float32))
        self.running_std = theano.shared(np.zeros(self.running_shape, np.float32))


class Dropout(object):

    def __init__(self, p):
        from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
        self.srng = RandomStreams(seed=np.random.randint(10e6))
        self.p = p

    def __call__(self, x):
            return x * self.srng.binomial(x.shape, p=1-self.p, dtype=theano.config.floatX)