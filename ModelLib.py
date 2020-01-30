
from keras.models import Model
from keras.layers import Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
from keras import backend as K
from keras import activations as Activ
from keras import initializers as Init
from keras.engine.topology import Layer


def activate_function(name, parameter=0.2):
    if name == 'relu':
        return K.relu

    elif name == 'softmax':
        return K.softmax

    elif name == 'identity':
        def identity(x):
            return x
        return identity


class Constraint(object):
    """
    Constraint template
    """
    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class MinMax(Constraint):

    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return K.clip(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value,
                'max_value': self.max_value}


class Expand(Layer):

    def __init__(self, **kwargs):
        super(Expand, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Expand, self).build(input_shape)

    def call(self, x):
        return K.expand_dims(x, axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], 1)


class ShrinkMod(Layer):

    def __init__(self, **kwargs):
        super(ShrinkMod, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name='W', shape=(1, 1, 1, 4),
                                 initializer=Init.Constant(0.5),
                                 constraint=MinMax(),
                                 trainable=True)
        super(ShrinkMod, self).build(input_shape)

    def call(self, x):

        a = 1.0 - self.W
        alpha = K.concatenate((a, self.W), axis=-1)
        x = K.sum(x * alpha, axis=-1)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2])


class Attention1(Layer):

    def __init__(self, parameter=0.2,
                 bias_layerwise=False,
                 kernel_regularizer=None,
                 kernel_constraint=None, batch_size=256, trainable_=True,
                 kernel_constraint1=None,
                 b=0.2, **kwargs):
        self.bias_layerwise = bias_layerwise
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.kernel_constraint1 = kernel_constraint1
        self.batch_size = 256
        self.train_W = trainable_
        self.b = b
        super(Attention1, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name='W', shape=(input_shape[2], input_shape[2]),
                                 initializer=Init.Constant(1.0/input_shape[2]),
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 trainable=True)
        self.in_shape = input_shape

        super(Attention1, self).build(input_shape)

    def call(self, x):

        w = self.W-self.W*K.eye(self.in_shape[2], dtype='float32')\
            + K.eye(self.in_shape[2], dtype='float32')/float(self.in_shape[2])

        e = K.dot(x, w)

        x = Activ.softmax(e, axis=-1)

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class Mode2(Layer):

    def __init__(self, output_dim, activation, parameter=0.2,
                 bias_layerwise=False,
                 kernel_regularizer=None,
                 kernel_constraint=None, batch_size=256, trainable_=True,
                 kernel_constraint1=None,
                 b=0.2, **kwargs):
        self.output_dim = output_dim
        self.activate = activate_function(activation, parameter)
        self.bias_layerwise = bias_layerwise
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.kernel_constraint1 = kernel_constraint1
        self.batch_size = 256
        self.train_W = trainable_
        self.b = b
        super(Mode2, self).__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(name='W', shape=(input_shape[2], self.output_dim),
                                      initializer='he_uniform',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True)
        self.bias = self.add_weight(name='bias', shape=(input_shape[1], self.output_dim),
                                    initializer='zeros', trainable=True)
        self.in_shape = input_shape

        super(Mode2, self).build(input_shape)

    def call(self, x):

        x = K.dot(x, self.W)+self.bias
        x = self.activate(x)

        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)


class FirstMode(Layer):

    """
    Calculate the 1-mode product using 10 x 10 input for 3-class weighting.
    Columns=Time dimension, rows=Features
    """

    def __init__(self, output_dim, activation, parameter=0.2,
                 bias_layerwise=False,
                 kernel_regularizer=None,
                 kernel_constraint=None, batch_size=256, trainable_=True,
                 kernel_constraint1=None,
                 b=0.2, **kwargs):
        self.output_dim = output_dim
        self.activate = activate_function(activation, parameter)
        self.bias_layerwise = bias_layerwise
        self.kernel_regularizer = kernel_regularizer
        self.kernel_constraint = kernel_constraint
        self.kernel_constraint1 = kernel_constraint1
        self.batch_size = 256
        self.train_W = trainable_
        self.b = b
        super(FirstMode, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(self.output_dim, input_shape[2]),  # 3 x 10
                                 initializer='he_uniform',
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 trainable=True)

        self.in_shape = input_shape

        super(FirstMode, self).build(input_shape)

    def call(self, x):

        x = K.permute_dimensions(x, (0, 2, 1))
        x = K.dot(self.W, x)
        x = K.permute_dimensions(x, (1, 0, 2))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim, input_shape[1]  # ? x 3 x 10


# Modified Temporal Attention for 4 attention masks
def get_subattention(kernel_regularizer=None, kernel_constraint=None):

    """
    :param kernel_regularizer:
    :param kernel_constraint:
    :return: Modified model with 4 masks
    """

    in_ap, in_av, in_bp, in_bv = Input([10, 10]), Input([10, 10]), Input([10, 10]), Input([10, 10])

    """ 
        Inputs X of size 10 x 10 must come in form (?, time, features)
        The 1-mode product is done to get the changes over time, keeping features - levels - intact
        First mode projection with K.dot(self.W, x) 
    """

    # Feature specific 1-mode products
    out_ap = FirstMode(3, 'identity', kernel_regularizer=kernel_regularizer,
                       kernel_constraint=kernel_constraint)(in_ap)

    out_av = FirstMode(3, 'identity', kernel_regularizer=kernel_regularizer,
                       kernel_constraint=kernel_constraint)(in_av)

    out_bp = FirstMode(3, 'identity', kernel_regularizer=kernel_regularizer,
                       kernel_constraint=kernel_constraint)(in_bp)

    out_bv = FirstMode(3, 'identity', kernel_regularizer=kernel_regularizer,
                       kernel_constraint=kernel_constraint)(in_bv)

    # Calculate the attention masks
    att_ap = Attention1(kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint)(out_ap)
    att_av = Attention1(kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint)(out_av)
    att_bp = Attention1(kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint)(out_bp)
    att_bv = Attention1(kernel_regularizer=kernel_regularizer, kernel_constraint=kernel_constraint)(out_bv)

    att_ap = Expand()(att_ap)
    att_av = Expand()(att_av)
    att_bp = Expand()(att_bp)
    att_bv = Expand()(att_bv)

    out_ap = Expand()(out_ap)
    out_av = Expand()(out_av)
    out_bp = Expand()(out_bp)
    out_bv = Expand()(out_bv)

    outputs = Concatenate(axis=-1)([out_ap, out_av, out_bp, out_bv,
                                    att_ap, att_av, att_bp, att_bv])

    # Calculate the attended values
    outputs = ShrinkMod()(outputs)

    # 2-mode product over the combined values
    outputs = Mode2(1, 'identity', kernel_regularizer=kernel_regularizer,
                    kernel_constraint=kernel_constraint)(outputs)

    outputs = Flatten()(outputs)
    outputs = Activation('softmax')(outputs)

    model = Model(inputs=[in_ap, in_av, in_bp, in_bv], outputs=outputs)
    adam = Adam(lr=0.01)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

