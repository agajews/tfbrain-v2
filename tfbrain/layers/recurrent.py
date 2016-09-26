import tensorflow as tf

from . import nonlin, init
from .core import Layer


class BasicRNNLayer(Layer):
    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin_h=nonlin.relu,
                 nonlin_o=nonlin.relu,
                 h_init=None,
                 i_init=None,
                 o_init=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({
            'num_nodes': num_nodes,
            'nonlin_h': nonlin_h,
            'nonlin_o': nonlin_o,
            'h_init': h_init,
            'i_init': i_init,
            'o_init': o_init,
        })
        self.incoming_shape = incoming.get_output_shape()
        self.num_nodes = num_nodes
        self.nonlin_o = nonlin_o
        self.nonlin_h = nonlin_h
        self.check_compatible(incoming)
        self.output_shape = self.calc_output_shape()

        if h_init is None:
            h_init = init.PairInit()
        if i_init is None:
            i_init = init.NormalInit()
        if o_init is None:
            o_init = init.PairInit()
        self.initialize_params(h_init, i_init, o_init)

        self.params = {'W_h': self.W_h,
                       'b_h': self.b_o,
                       'W_i': self.W_i,
                       'W_o': self.W_o,
                       'b_o': self.b_h}

    def initialize_params(self, h_init, i_init, o_init):
        W_h_shape = (self.num_nodes, self.num_nodes)
        b_h_shape = (self.num_nodes,)
        W_i_shape = (self.incoming_shape[2], self.num_nodes)
        W_o_shape = (self.num_nodes, self.num_nodes)
        b_o_shape = (self.num_nodes,)

        W_h_init, b_h_init = h_init.get_init(W_h_shape, b_h_shape)
        self.W_h = tf.Variable(W_h_init, trainable=self.trainable)
        self.b_h = tf.Variable(b_h_init, trainable=self.trainable)
        W_i_init = i_init.get_init(W_i_shape)
        self.W_i = tf.Variable(W_i_init, trainable=self.trainable)
        W_o_init, b_o_init = h_init.get_init(W_o_shape, b_o_shape)
        self.W_o = tf.Variable(W_o_init, trainable=self.trainable)
        self.b_o = tf.Variable(b_o_init, trainable=self.trainable)

    def calc_output_shape(self):
        return self.incoming_shape[:2] + (self.num_nodes,)

    def check_compatible(self, incoming):
        if not len(incoming.get_output_shape()) == 3:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, try passing it through '
                             'a ReshapeLayer or SequenceFlattenLayer first')
                            % str(incoming.get_output_shape()))

    def get_base_name(self):
        return 'brnn'

    def recurrence_fn(self, h_prev, x_t):
        return self.nonlin_h(
            tf.matmul(h_prev, self.W_h) +
            tf.matmul(x_t, self.W_i) + self.b_h)

    def output_fn(self, state):
        return self.nonlin_o(
            tf.matmul(state, self.W_o) + self.b_o)

    def get_initial_hidden(self, incoming_var):
        initial_hidden = tf.matmul(
            incoming_var[:, 0, :],
            tf.zeros([self.incoming_shape[2], self.num_nodes]))
        return initial_hidden

    def _get_output(self, incoming_var):
        initial_hidden = self.get_initial_hidden(incoming_var)
        incoming_var = tf.transpose(incoming_var, (1, 0, 2))
        states = tf.scan(self.recurrence_fn,
                         incoming_var,
                         initializer=initial_hidden)

        outputs = tf.map_fn(self.output_fn,
                            states)

        outputs = tf.transpose(outputs, (1, 0, 2), name='end_transpose')
        return outputs


class LSTMLayer(BasicRNNLayer):
    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlin_i=nonlin.sigmoid,
                 nonlin_c=nonlin.tanh,
                 nonlin_o=nonlin.relu,
                 i_init=None,
                 U_i_init=None,
                 f_init=None,
                 U_f_init=None,
                 g_init=None,
                 U_g_init=None,
                 c_init=None,
                 U_c_init=None,
                 o_init=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({
            'num_nodes': num_nodes,
            'nonlin_i': nonlin_i,
            'nonlin_c': nonlin_c,
            'nonlin_o': nonlin_o,
            'i_init': i_init,
            'U_i_init': U_i_init,
            'f_init': f_init,
            'U_f_init': U_f_init,
            'g_init': g_init,
            'U_g_init': U_g_init,
            'c_init': c_init,
            'U_c_init': U_c_init,
            'o_init': o_init,
        })

        self.incoming_shape = incoming.get_output_shape()
        self.num_nodes = num_nodes
        self.nonlin_i = nonlin_i
        self.nonlin_c = nonlin_c
        self.nonlin_o = nonlin_o
        self.check_compatible(incoming)
        self.output_shape = self.calc_output_shape()

        self.initialize_params(i_init, U_i_init,
                               f_init, U_f_init,
                               g_init, U_g_init,
                               c_init, U_c_init,
                               o_init)

        self.params = {'W_i': self.W_i,
                       'U_i': self.U_i,
                       'W_f': self.W_f,
                       'U_f': self.U_f,
                       'W_g': self.W_g,
                       'U_g': self.U_g,
                       'W_c': self.W_c,
                       'U_c': self.U_c,
                       'W_o': self.W_o,
                       'b_i': self.b_i,
                       'b_f': self.b_f,
                       'b_g': self.b_g,
                       'b_c': self.b_c,
                       'b_o': self.b_o}

    def initialize_params(self,
                          i_init, U_i_init,
                          f_init, U_f_init,
                          g_init, U_g_init,
                          c_init, U_c_init,
                          o_init):

        input_size = self.incoming_shape[2]

        W_i_shape = (input_size, self.num_nodes)
        b_i_shape = (self.num_nodes,)
        U_i_shape = (self.num_nodes, self.num_nodes)

        W_f_shape = (input_size, self.num_nodes)
        b_f_shape = (self.num_nodes,)
        U_f_shape = (self.num_nodes, self.num_nodes)

        W_g_shape = (input_size, self.num_nodes)
        b_g_shape = (self.num_nodes,)
        U_g_shape = (self.num_nodes, self.num_nodes)

        W_c_shape = (input_size, self.num_nodes)
        b_c_shape = (self.num_nodes,)
        U_c_shape = (self.num_nodes, self.num_nodes)

        W_o_shape = (self.num_nodes, self.num_nodes)
        b_o_shape = (self.num_nodes,)

        W_i_init, b_i_init = i_init.get_init(W_i_shape, b_i_shape)
        self.W_i = tf.Variable(W_i_init, trainable=self.trainable)
        self.b_i = tf.Variable(b_i_init, trainable=self.trainable)
        self.U_i = tf.Variable(U_i_init.get_init(U_i_shape),
                               trainable=self.trainable)

        W_f_init, b_f_init = f_init.get_init(W_f_shape, b_f_shape)
        self.W_f = tf.Variable(W_f_init, trainable=self.trainable)
        self.b_f = tf.Variable(b_f_init, trainable=self.trainable)
        self.U_f = tf.Variable(U_f_init.get_fnit(U_f_shape),
                               trainable=self.trainable)

        W_g_init, b_g_init = g_init.get_init(W_g_shape, b_g_shape)
        self.W_f = tf.Variable(W_g_init, trainable=self.trainable)
        self.b_f = tf.Variable(b_g_init, trainable=self.trainable)
        self.U_f = tf.Variable(U_g_init.get_fnit(U_g_shape),
                               trainable=self.trainable)

        W_c_init, b_c_init = c_init.get_init(W_c_shape, b_c_shape)
        self.W_f = tf.Variable(W_c_init, trainable=self.trainable)
        self.b_f = tf.Variable(b_c_init, trainable=self.trainable)
        self.U_f = tf.Variable(U_c_init.get_fnit(U_c_shape),
                               trainable=self.trainable)

        W_o_init, b_o_init = o_init.get_init(W_o_shape, b_o_shape)
        self.W_f = tf.Variable(W_o_init, trainable=self.trainable)
        self.b_f = tf.Variable(b_o_init, trainable=self.trainable)

    def get_base_name(self):
        return 'lstm'

    def get_initial_hidden(self, incoming_var):
        initial_hidden = tf.matmul(
            incoming_var[:, 0, :],
            tf.zeros([self.incoming_shape[2], self.num_nodes]))
        return tf.pack([initial_hidden, initial_hidden])

    def recurrence_fn(self, h_prev_tuple, x_t):
        h_prev, c_prev = tf.unpack(h_prev_tuple)

        i = self.nonlin_i(
            tf.matmul(x_t, self.W_i,
                      name='i_w') +
            tf.matmul(h_prev, self.U_i,
                      name='i_u') +
            self.b_i)

        f = self.nonlin_i(
            tf.matmul(x_t, self.W_f,
                      name='f_w') +
            tf.matmul(h_prev, self.U_f,
                      name='f_u') +
            self.b_f)

        g = self.nonlin_i(
            tf.matmul(x_t, self.W_g,
                      name='g_w') +
            tf.matmul(h_prev, self.U_g,
                      name='g_u') +
            self.b_g)

        c_tilda = self.nonlin_c(
            tf.matmul(x_t, self.W_c,
                      name='ct_w') +
            tf.matmul(h_prev, self.U_c,
                      name='ct_u') +
            self.b_c)

        c = f * c_prev + i * c_tilda
        h = g * self.nonlin_c(c)
        return tf.pack([h, c])

    def output_fn(self, states_tuple):
        h = states_tuple[0, :, :]
        return self.nonlin_o(
            tf.matmul(h, self.W_o) +
            self.b_o)


class NetOnSeq(Layer):
    def __init__(self,
                 incoming,
                 net,
                 **kwargs):

        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'net': net})
        self.incoming_shape = incoming.get_output_shape()
        self.net = net
        self.output_shape = self.calc_output_shape()

    def calc_output_shape(self):
        return (None, None) + self.net.get_output_shape()[1:]

    def get_base_name(self):
        return 'net_on_seq'

    def output_fn(self, state):
        input_layer = self.net.get_layers()[0]
        input_layer.set_passthrough(state)
        self.net.build_output()
        return input_layer.get_output()

    def _get_output(self, incoming_var, **kwargs):
        incoming_var = tf.transpose(
            incoming_var, (1, 0, *range(2, len(incoming_var.get_shape()))),
            name='front_transpose')

        outputs = tf.map_fn(self.output_fn,
                            incoming_var)

        outputs = tf.transpose(
            outputs, (1, 0, 2),
            name='end_transpose')
        return outputs
