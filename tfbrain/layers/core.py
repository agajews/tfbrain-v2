import tensorflow as tf

from tfbrain import nonlin, init
from tfbrain.helpers import flatten_params


class Layer(object):

    def __init__(self, incoming_list, sess=None, name=None, trainable=True):
        if len(incoming_list) > 0:
            self.sess = incoming_list[0].get_sess()
            for layer in incoming_list[1:]:
                assert layer.get_sess() is self.get_sess()
        elif sess is not None:
            self.sess = sess
        else:
            raise ValueError('Please specify a session (this is a leaf layer)')
        self.incoming = incoming_list
        self.trainable = trainable
        self.chosen_name = name
        self.gen_name(0)
        self.params = {}
        self.config = {'name': name}

    def get_sess(self):
        return self.sess

    def get_params(self):
        return self.params

    def get_output_shape(self):
        return self.output_shape

    def get_base_name(self):
        return 'layer'

    def gen_name(self, num_global_layers):
        num_local_layers = 0
        for incoming in self.incoming:
            incoming.gen_name(num_global_layers)
            num_local_layers += incoming.num_local_layers
            num_global_layers += incoming.num_local_layers
        num_local_layers += 1  # this layer
        self.layer_num = num_global_layers + 1
        self.num_local_layers = num_local_layers
        if self.chosen_name is None:
            self.name = '%s_%d' % (self.get_base_name(), self.layer_num)
        else:
            self.name = self.chosen_name

    def get_name(self):
        return self.name

    def _get_output(self, *incoming_vars):
        '''This method takes a TF variable
        (presumably the output from the current
        layer's incoming layer, but not necessarily)
        and uses it to get the output of the current
        layer'''
        raise NotImplementedError()

    def build_output(self):
        for layer in self.incoming:
            layer.build_output()
        self.output = self._get_output(
            *[l.get_output() for l in self.incoming])

    def get_output(self):
        return self.output

    def eval(self, **input_vars):
        feed_dict = self.get_feed_dict(input_vars, train=False)
        return self.sess.run(self.output, feed_dict=feed_dict)

    def get_feed_dict(self, input_vals, train=False):
        feed_dict = {p: input_vals[n] for (n, p)
                     in self.get_input_vars().items()}
        feed_dict.update(self.get_supp_feed_dict(train))
        return feed_dict

    def clone(self, keep_params=False, **kwargs):
        args = {}
        args.update(self.config)
        args.update(kwargs)
        if keep_params:
            args.update(self.params)
        if len(self.incoming) == 0:  # InputLayer
            args['sess'] = self.sess
            return self.__class__(**args)
        else:  # not InputLayer
            if len(self.incoming) == 1:
                incoming = self.incoming[0].clone(keep_params, **kwargs)
            else:  # multiple inputs (e.g. MergeLayer)
                incoming = map(
                    lambda i: i.clone(**kwargs),
                    self.incoming)
            return self.__class__(incoming, **args)

    def get_all_params(self, just_trainable=False,
                       eval_values=False, flatten=False):
        layers = self.get_layers_dict()
        params = {l_n: l.get_params() for (l_n, l) in layers.items()}
        if just_trainable:
            params = {l_n: l_p for (l_n, l_p) in params.items()
                      if layers[l_n].trainable}
        if eval_values:
            params = {l_n: {p_n: self.sess.run(p) for (p_n, p) in l_p.items()}
                      for (l_n, l_p) in params.items()}
        if flatten:
            params = flatten_params(params)
        return params

    def set_all_params(self, src_params, just_trainable=False,
                       flattened=False, run_ops=False):
        if not flattened:
            dest_dict = self.get_all_params(just_trainable)
            assert dest_dict.keys() == src_params.keys()
            for layer_name, layer_params in dest_dict.items():
                assert layer_params.keys() == src_params[layer_name].keys()
            src_params = flatten_params(src_params)
        ops = []
        dest_params = self.get_all_params(just_trainable, flatten=True)
        for dest, src in zip(dest_params, src_params):
            ops.append(dest.assign(src))
        if run_ops:
            self.sess.run(ops)
        return ops

    def set_all_params_from(self, src_net, just_trainable=False,
                            run_ops=False):
        return self.set_all_params(src_net.get_all_params(just_trainable),
                                   run_ops=run_ops)

    def _get_layers_list(self, layers):
        for layer in self.incoming:
            layers = layer._get_layers_list(layers)
        layers.append(self)
        return layers

    def get_layers_list(self):
        '''Returns topologically sorted list
        of layers leading up to and including this one'''
        return self._get_layers_list([])

    def get_layers_dict(self):
        return {l.get_name(): l for l in self.get_layers_list()}

    def get_input_vars(self):
        input_vars = {}
        for layer in self.get_layers_list():
            if len(layer.incoming) == 0:  # InputLayer
                input_vars[layer.get_name()] = layer.get_input_var()
        return input_vars

    def get_supp_feed_dict(self, train=False):
        supp_feed_dict = {}
        for layer in self.get_layers_list():
            supp_feed_dict.update(layer._get_supp_feed_dict(train))
        return supp_feed_dict

    def _get_supp_feed_dict(self, train=False):
        '''Returns any supplementary feed_dict
        items (for filling TF placeholders)
        (e.g. a dropout layer might set keep_prob)'''
        return {}

    def get_input_hidden_var(self, **kwargs):
        return None

    def get_output_hidden_var(self, **kwargs):
        return None

    def get_init_hidden(self):
        return None

    def get_assign_hidden_op(self, **kwargs):
        return None


class InputLayer(Layer):

    def __init__(self,
                 shape,
                 sess,
                 dtype=tf.float32,
                 **kwargs):
        Layer.__init__(self, [], sess=sess, **kwargs)
        self.dtype = dtype
        self.output_shape = shape
        self.initialize_placeholder()
        self.config.update({'shape': shape,
                            'dtype': dtype})

    def set_passthrough(self, var):
        var_shape = var.get_shape().as_list()
        assert len(var_shape) == len(self.output_shape)
        for var_dim, out_dim in zip(var_shape, self.output_shape):
            if var_dim is not None and out_dim is not None:
                assert var_dim == out_dim
        self.var = var

    def get_input_var(self):
        return self.var

    def get_base_name(self):
        return 'input'

    def initialize_placeholder(self):
        self.var = tf.placeholder(self.dtype,
                                  shape=self.output_shape,
                                  name=self.name)

    def _get_output(self):
        return self.var


class FullyConnectedLayer(Layer):

    def __init__(self,
                 incoming,
                 num_nodes,
                 nonlinearity=nonlin.relu,
                 initializer=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'num_nodes': num_nodes,
                            'nonlinearity': nonlinearity,
                            'initializer': initializer})

        self.check_compatible(incoming)
        self.incoming_shape = incoming.get_output_shape()

        self.num_nodes = num_nodes
        self.nonlin = nonlinearity

        if initializer is None:
            initializer = init.PairInit()
        self.initialize_params(initializer)
        self.params = {'W': self.W,
                       'b': self.b}

        self.output_shape = (None, num_nodes)

    def get_base_name(self):
        return 'fc'

    def check_compatible(self, incoming):
        if not len(incoming.get_output_shape()) == 2:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, try passing it through '
                             'a FlattenLayer first')
                            % str(incoming.get_output_shape()))

    def initialize_params(self, initializer):
        W_shape = (self.incoming_shape[1], self.num_nodes)
        b_shape = (self.num_nodes,)

        W_init, b_init = initializer.get_init(W_shape, b_shape)
        self.W = tf.Variable(W_init, trainable=self.trainable)
        self.b = tf.Variable(b_init, trainable=self.trainable)

    def _get_output(self, incoming_var):
        return self.nonlin(
            tf.matmul(incoming_var, self.W) + self.b)
