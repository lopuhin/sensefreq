import contextlib
from math import sqrt, ceil

from keras import backend as K
from keras import initializations
from keras.engine.topology import Layer
from keras.engine import InputSpec
from theano.tensor.nnet import h_softmax


def repeat_iter(it, *args, **kwargs):
    while True:
        for item in it(*args, **kwargs):
            yield item


@contextlib.contextmanager
def printing_done(msg: str):
    print(msg, end=' ', flush=True)
    yield
    print('done')


class HierarchicalSoftmax(Layer):
    def __init__(self, output_dim, init='glorot_uniform', **kwargs):
        self.init = initializations.get(init)
        self.output_dim = output_dim
        l1 = ceil(sqrt(output_dim))
        l2 = ceil(output_dim / l1)
        self.n_classes, self.n_outputs_per_class = int(l1), int(l2)
        super(HierarchicalSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = self.input_spec[0].shape[-1]
        self.W1 = self.init((input_dim, self.n_classes),
                            name='{}_W1'.format(self.name))
        self.b1 = K.zeros((self.n_classes,), name='{}_b1'.format(self.name))
        self.W2 = self.init(
            (self.n_classes, input_dim, self.n_outputs_per_class),
            name='{}_W2'.format(self.name))
        self.b2 = K.zeros((self.n_classes, self.n_outputs_per_class),
                          name='{}_b2'.format(self.name))

        self.trainable_weights = [self.W1, self.b1, self.W2, self.b2]

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, X, mask=None):
        # TODO - target?
        Y = h_softmax(X, K.shape(X)[0], self.output_dim,
                      self.n_classes, self.n_outputs_per_class,
                      self.W1, self.b1, self.W2, self.b2)
        return Y

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__}
        base_config = super(HierarchicalSoftmax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
