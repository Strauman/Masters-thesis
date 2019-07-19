import tensorflow as tf
# tf.enable_eager_execution()
from sys import exit as xit
import numpy as np
from tensorflow.keras.layers import Lambda, Input, InputLayer
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.training.checkpointable.data_structures import _ListWrapper as ListWrapper
from tensorflow.python.util import tf_inspect
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine.network import Network

class Remember(object):
    """
    Purpose: Give a function the Remember-instance.
    Usage:
        @Remember
        def myfunc(a,s,d,f):
            pass
        isinstance(myfunc,Remember) # -> True
    """

    def __init__(self, layer):
        self.layer=layer

    def __call__(self, useq_instance):
        # assert isinstance(useq_instance,USequential), "Remember can only be used with USequential"
        useq=useq_instance
        useq._output_idx.append(len(useq._layers))
        return self.layer


class USequential(object):
    def __init__(self):
        # self._sequential=Sequential()
        self._layers = []
        self._output_idx = []
        self._extra_outputs = []


    def add(self, *layers):
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            layers = layers[0]
        if not isinstance(layers, (list, tuple)):
            layers = [layers]

        for l in layers:
            if isinstance(l, Remember):
                #pylint: disable=E1102
                l = l(self)
            self._layers.append(l)
            # self._sequential.add(l)

    def build(self, inputs, training=None, mask=None):
        # if not isinstance(self._layers[0], InputLayer):
            # x = Input(tensor=inputs)
        # else:
            # x = inputs
            # x = self._layers[0](inputs)
        x = Input(tensor=inputs)
        _extra_outputs = []
        for i, l in enumerate(self._layers):
            layer_kwargs = {}
            layer_argspec = tf_inspect.getfullargspec(l.call).args
            if 'training' in layer_argspec:
                layer_kwargs['training'] = training
            if 'mask' in layer_argspec:
                layer_kwargs['mask'] = training
            # if isinstance(l, ListWrapper):
                # continue
            # print(isinstance(l, Network))
            if not l.built:
                with ops.name_scope(l._name_scope()):
                    l._maybe_build(x)
                l.built = True
            # y = l.call(x, **layer_kwargs)
            y=l(x, **layer_kwargs)
            if i in self._output_idx:
                # print("Output saved after evaluating", l.name)
                _extra_outputs.append(y)
            x = y

        mod = Model(inputs=[inputs], outputs=[x, *_extra_outputs])
        return mod

    def __call__(self, inputs, training=None, mask=None):
        main_output, *copies = self.build(inputs, training, mask)(inputs)
        return main_output, copies

if __name__ == '__main__':
    seq = USequential()
    seq.add([
        Lambda(lambda x:x * 2, name="0_x2"),
        Remember(Lambda(lambda x:x / 2, name="1_div_2")),
        Lambda(lambda x:x * 4, name="2_x4")
    ])
    # seq.add(Lambda(lambda x: x*2))
    X_ = np.arange(3 * 3).reshape((3, 3))
    X = tf.placeholder_with_default(X_, (3, 3))
    sess = tf.InteractiveSession()
    # Yhat,Z = seq(X)
    # mod=seq.build(X)
    Yhat, Z = seq(X)
    print("Yhat:\n", Yhat.eval())
    print("Z:\n", Z[0].eval())
