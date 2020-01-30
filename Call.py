import keras
from keras import backend as k
import numpy as np


class MaskHistories(keras.callbacks.Callback):
    """
    Custom Callback object to catch changes in weights
    """

    def __init__(self, data):
        self.data = data

    def on_train_begin(self, logs={}):
        self.all_outputs = []
        self.derivatives = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        inputs = self.model.input

        in_ap = inputs[0]
        in_av = inputs[1]
        in_bp = inputs[2]
        in_bv = inputs[3]

        out_ap = [self.model.layers[8].output]
        out_av = [self.model.layers[9].output]
        out_bp = [self.model.layers[10].output]
        out_bv = [self.model.layers[11].output]

        functor1 = k.function([in_ap] + [k.learning_phase()], out_ap)
        functor2 = k.function([in_av] + [k.learning_phase()], out_av)
        functor3 = k.function([in_bp] + [k.learning_phase()], out_bp)
        functor4 = k.function([in_bv] + [k.learning_phase()], out_bv)

        tmp0 = np.mean(functor1([self.data[0], 1.0])[0], axis=0)
        tmp1 = np.mean(functor1([self.data[1], 1.0])[0], axis=0)
        tmp2 = np.mean(functor1([self.data[2], 1.0])[0], axis=0)

        tmp3 = np.mean(functor2([self.data[3], 1.0])[0], axis=0)
        tmp4 = np.mean(functor2([self.data[4], 1.0])[0], axis=0)
        tmp5 = np.mean(functor2([self.data[5], 1.0])[0], axis=0)

        tmp6 = np.mean(functor3([self.data[6], 1.0])[0], axis=0)
        tmp7 = np.mean(functor3([self.data[7], 1.0])[0], axis=0)
        tmp8 = np.mean(functor3([self.data[8], 1.0])[0], axis=0)

        tmp9 = np.mean(functor4([self.data[9], 1.0])[0], axis=0)
        tmp10 = np.mean(functor4([self.data[10], 1.0])[0], axis=0)
        tmp11 = np.mean(functor4([self.data[11], 1.0])[0], axis=0)

        self.all_outputs.append([tmp0, tmp1, tmp2,
                                 tmp3, tmp4, tmp5,
                                 tmp6, tmp7, tmp8,
                                 tmp9, tmp10, tmp11])

        self.derivatives.append(self.model.layers[21].get_weights()[0])
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return