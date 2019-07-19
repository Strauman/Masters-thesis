from tfmodels import TFModel
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Conv2DTranspose, Lambda, InputLayer, concatenate, BatchNormalization,Dropout, Dense
from tensorflow.keras.models import Sequential, Model
from . import util
from tfmodels import USequential, Remember
from sys import exit as xit
import tensorflow as tf
import keras
Input = tf.keras.Input
dropout_rate = 0.4


# class NestedLayers():
#     """
#     https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html#decorator-functions-with-decorator-arguments
#     - __init__ gets the arguments sent to decorator
#     - __call__ is called with myfunc as arguments,
#       and is expected to return the function which will be called.
#     """
#     def __init__(self, **decorator_kwargs):
#         for k,v in decorator_kwargs.items():
#             setattr(self,k,v)
#     def __call__(self, func):
#         def wrapper():
#           #do stuff with things
#           return func
#         return wrapper()
#         # Or just plain return func

class LayerList(object):
    """
    https://python-3-patterns-idioms-test.readthedocs.io/en/latest/PythonDecorators.html#review-decorators-without-arguments
    No args can be sent to this decorator
    - __init__ gets myfunc as argument
    - __call__ gets called when myfunc(asdf) is called. Called is passed any arguments sent to myfunc
    """

    def __init__(self, function):
        self.func = function

    def __call__(self, *args, **kwargs):
        layer_list = self.func(*args, **kwargs)

        def _con_layer(input):
            x = input
            for l in layer_list:
                x = l(x)
            return x
        return _con_layer
# Dropout(rate=dropout_rate)(Z, training=is_training)
@LayerList
def conv_layer(filters, **kwargs):
    # default_opts = dict(
    #     kernel_size=[3, 3],
    #     strides=(1, 1),
    #     padding="SAME",
    #     activation=Activation("relu")
    # )
    # default_opts.update(kwargs)
    # Z = tf.layers.conv2d(Z, filters=filters, kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lambda2))
    # return Conv2D(filters=int(filters), **default_opts)
    return [
        # Dropout(dropout_rate),
        Conv2D(filters=int(filters), kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=None),
        Activation("relu")
    ]


def bn_layer(**kwargs):

    return BatchNormalization(**kwargs)
    # return BatchNormalization(
    #     axis=-1,
    #     momentum=0.99,
    #     epsilon=0.001,
    #     center=True,
    #     scale=True,
    #     beta_initializer='zeros',
    #     gamma_initializer='ones',
    #     moving_mean_initializer='zeros',
    #     moving_variance_initializer='ones',
    #     beta_regularizer=None,
    #     gamma_regularizer=None,
    #     beta_constraint=None,
    #     gamma_constraint=None
    # )


def pooling_layer(**kwargs):
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME', data_format=None, **kwargs)

@LayerList
def tr_conv_layer(filters, **kwargs):
    opts = dict(activation=None, kernel_size=(1, 1), strides=(2, 2), padding='SAME')
    # opts.update(kwargs)
    return [
        # Dropout(dropout_rate),
        Conv2DTranspose(int(filters), **opts),
        Activation("relu")
    ]
    # return lambda Z: tf.layers.conv2d_transpose(Z, filters=int(filters), kernel_size=(1, 1), strides=2, padding="SAME", activation=tf.nn.relu, **kwargs)


def encoder_model(X, cfg):
    depth = cfg.depth
    filters = cfg.filters
    copies = []
    X_in = Input(shape=X.shape[1:], name="Enc_input")
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    # First
    for _ in range(depth):
        filters *= 2
        x = conv_layer(filters)(x)
        x = conv_layer(filters)(x)
        copies.append(x)
        # print(x.shape)
        x = bn_layer()(x)
        x = pooling_layer()(x)
        # print(x.shape)
    filters *= 2
    # Bottom
    x = conv_layer(filters)(x)
    x = conv_layer(filters)(x)
    Z = x
    # print(Z.shape,filters)
    # print("--")
    # xit()
    mod = Model(inputs=[X_in], outputs=[Z, *copies], name="Encoder")
    return mod


def decoder_model(Z, cp_tensors, cfg, get_layer=False):
    depth = cfg.depth
    filters = cfg.filters * (2**(depth + 1))
    # Set filters to the same as encoder
    cp_ins = [Input(shape=c.shape[1:], name="cpy{}".format(i)) for i, c in enumerate(cp_tensors[::-1])]
    # print(Z.shape)
    # xit()
    # print(cp_ins[0].shape, cp_ins[1].shape)
    # cp_in=Input(shape=cp_ts.shape[1:], name="cp_ts_in")
    Z_in = Input(shape=Z.shape[1:], name="Dec_input")
    # print(Z.shape,filters)
    # for c in cp_ins:
    # print(c.shape)
    # xit()
    # First up to align with layers
    # x=Lambda(lambda x: x)(Z_in)
    x = Z_in
    for d in range(depth):
        x = tr_conv_layer(filters)(x)
        filters /= 2
        # print(x.shape,cp_ins[d].shape)
        # y = lambda z, cpy_t: concatenate([cpy_t, z], -1)
        x = concatenate([cp_ins[d], x])
        # x = y(x, cp_ins[d])
        x = conv_layer(filters)(x)
        x = conv_layer(filters)(x)
        x = bn_layer()(x)
    # x = conv_layer(1, name="last_decode_map", activation=Activation('sigmoid'))(x)
    x = Conv2D(1, kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation="sigmoid")(x)
    # x = conv_layer(1, name="last_decode_map", activation=Activation('sigmoid'))(x)
    # x=Activation('sigmoid')(x)
    # x = Lambda(lambda z: tf.nn.sigmoid(z))(x)
    x = Lambda(lambda x: tf.squeeze(x, -1), name="Decoder_squeeze")(x)
    # mod.summary()
    # xit()
    mod = Model(inputs=[Z_in, *cp_ins], outputs=[x], name="Decoder")
    retlist=[mod]
    if get_layer:
        retlist.append(x)

    return retlist

def dense_discriminator(Z, cp_tensors, cfg):
    depth = cfg.depth
    filters = cfg.filters * (2**(depth + 1))
    cp_ins = [Input(shape=c.shape[1:], name="cpy{}".format(i)) for i, c in enumerate(cp_tensors)]
    Z_in = Input(shape=Z.shape[1:], name="Dec_input")
    Z_tensors=[Z_in]+cp_ins
    outputs=[]
    for V in Z_tensors:
        outputs=[Dense(1,input_shape=V.shape[1:])(V)]
    out=Lambda(lambda x:tf.nn.sigmoid(tf.reduce_sum(x)))(outputs)
    mod=Model(inputs=Z_tensors, outputs=[out])
    return mod
