from . import TFModel
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Conv2DTranspose, Lambda, InputLayer, concatenate, BatchNormalization,Dropout, Dense,Flatten
from tensorflow.keras.layers import Reshape, LeakyReLU, ReLU
# from tensorflow.keras.models import Sequential
from . import UpdModel as Model
from . import util
# from tfmodels import USequential, Remember
from ..helpers.cprints import color_print
from sys import exit as xit
import tensorflow as tf
import keras
from . import LayerList, register_kernel_L2_on_scope
Input = tf.keras.Input
dropout_rate = 0.5
dropout_discriminator_rate=0.5
import typing
from ..configs import ADDA_CFG
l_val=0.001
conv_args=lambda: dict(kernel_initializer=tf.keras.initializers.he_normal(), bias_initializer=tf.keras.initializers.he_normal())
# conv_args=lambda: {}
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

# Dropout(rate=dropout_rate)(Z, training=is_training)

class clayer(object):
    """docstring for clayer."""

    def __init__(self, filters):
        self.filters=int(filters)
    def __call__(self, inputs, *args, **kwargs):
        x=Conv2D(filters=int(self.filters), kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=None, **conv_args())(inputs)
        x=Activation("relu")(x)
        return x
@LayerList
def conv_layer(filters, **kwargs):
    # return clayer(filters)
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
        Conv2D(filters=int(filters), kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=None, **conv_args()),
        Activation("relu")
    ]


from pprint import pprint as pp
class BatchNormalizationUPD(BatchNormalization):
    def __init__(self, *args, **kwargs):
        super(BatchNormalizationUPD,self).__init__(*args,**kwargs)
    def __call__(self, inputs, *args, **kwargs):
        call_result=super(BatchNormalizationUPD,self).__call__(inputs, *args, **kwargs)
        # updates=self.get_updates_for(inputs)
        # updates=self.updates
        # pp(self.updates)
        # xit()
        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updates[0])
        return call_result


def bn_layer(name=None,**kwargs):
    return BatchNormalizationUPD(**kwargs)
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
from contextlib import contextmanager
@contextmanager
def null_context(*args, **kwargs):
    yield None

def down_sampling_layer(filters,activation=None,**kwargs):
    return Conv2D(filters=filters, kernel_size=[2, 2], strides=(2, 2), padding="SAME", activation=activation, **conv_args())

def down_sampling_layer_mpool(filters,**kwargs):
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

def conv_classifier_first(Z,main_cfg,get_layer=False,is_training=False):
    bn_training=True
    X_in=Input(shape=Z.shape[1:])
    x=X_in
    # x = Flatten()(x)
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    bn_training=True
    x = Conv2D(1, kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=None, **conv_args())(x)
    # x=LeakyReLU()(x)
    # x=ReLU()(x)
    # x = Activation("relu")(x)
    x=bn_layer()(x, training=bn_training)
    x=Conv2D(1, kernel_size=[3,3], strides=(1,1), padding="SAME", activation=None, **conv_args())(x)
    x = Lambda(lambda x: tf.squeeze(x, -1), name="clf_squeeze")(x)
    mod=Model(inputs=[X_in], outputs=[x], name="m_conv_classifier_first")
    return mod

def encoder_model_hidden(X, main_cfg: ADDA_CFG, role_cfg,get_layer=False,is_training=False):
    vs_name=tf.get_variable_scope().name
    cfg=role_cfg
    bn_training=True
    depth = cfg.depth
    filters = cfg.filters
    copies = []
    X_in = Input(shape=main_cfg.input_shape, name="Enc_input")
    # X_in= Input(tensor=X, name="Enc_input")
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    # First
    #pylint: disable=E1102
    # x=Dropout(dropout_rate)(x)
    # x=Conv2D(filters=3, kernel_size=[28, 28], strides=(1, 1), padding="SAME", activation=None)(x)
    for depth_count in range(depth):
        filters *= 2
        with tf.variable_scope("L2_m_unet_full"):
        # with null_context():
            x = conv_layer(filters)(x, training=is_training)
            x = conv_layer(filters)(x, training=is_training)
            x = bn_layer()(x, training=bn_training)
        # x = Dropout(dropout_rate)(x, training=is_training)
        copies.append(x)
        x = down_sampling_layer(filters, activation=None)(x)
        x = Activation("relu")(x)
    filters *= 2
    # Bottom
    x = bn_layer()(x, training=bn_training)
    # x = Dropout(dropout_rate)(x)
    with tf.variable_scope("L2_m_unet_full"):
    # with null_context():
        x = conv_layer(filters)(x, training=is_training)
        x = conv_layer(filters)(x, training=is_training)
    # x = Dropout(dropout_rate)(x, training=is_training)
    copies=copies[::-1]
    for d in range(depth):
        x = tr_conv_layer(filters)(x, training=is_training)
        filters /= 2
        x = concatenate([copies[d], x])
        x = bn_layer()(x, training=bn_training)
        with tf.variable_scope("L2_m_unet_full"):
        # with null_context():
            x = conv_layer(filters)(x, training=is_training)
            x = conv_layer(filters)(x, training=is_training)
        # x = Dropout(dropout_rate)(x, training=is_training)
        # x = Flatten()(x)
        # x = Dense(main_cfg.input_shape[1]*main_cfg.input_shape[0],activation=None)(x)
        # x = Dense(main_cfg.input_shape[1]*main_cfg.input_shape[0],activation=None)(x)
        # x = Reshape(main_cfg.input_shape)(x)
    # register_kernel_L2_on_scope("L2_m_unet_full")
    x=conv_layer(1)(x, training=is_training)
    x = Lambda(lambda x: tf.squeeze(x, -1), name="unet_enc_squeeze")(x)
    mod = Model(inputs=[X_in], outputs=[x], name="m_unet_full_interm")
    # print(X)
    # x=mod_a(X_in)
    # print(tf.shape(x))
    return mod

def encoder_model(X, main_cfg: ADDA_CFG, role_cfg,get_layer=False,is_training=False,max_pool=False):
    vs_name=tf.get_variable_scope().name
    cfg=role_cfg
    bn_training=True
    depth = cfg.depth
    filters = cfg.filters
    copies = []
    X_in = Input(shape=main_cfg.input_shape, name="Enc_input")
    do_dropout=False
    do_l2=False
    l_val=0
    dropout_rate=0
    if hasattr(role_cfg, "dropout"):
        color_print("DOING dropout", style="warning")
        do_dropout=True
        dropout_rate=role_cfg.dropout
    else:
        color_print("NOT doing dropout", style="danger")
    if hasattr(role_cfg, "l2"):
        color_print("DOING l2", style="warning")
        l_val=role_cfg.l2
        do_l2=True
    # X_in= Input(tensor=X, name="Enc_input")
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    # First
    #pylint: disable=E1102
    # x=Dropout(dropout_rate)(x)
    # x=Conv2D(filters=3, kernel_size=[28, 28], strides=(1, 1), padding="SAME", activation=None)(x)
    for depth_count in range(depth):
        filters *= 2
        with tf.variable_scope("L2_m_unet_full"):
        # with null_context():
            x = conv_layer(filters)(x, training=is_training)
            x = conv_layer(filters)(x, training=is_training)
            x = bn_layer()(x, training=bn_training)
        if do_dropout:
            x = Dropout(dropout_rate)(x, training=is_training)
        copies.append(x)
        if max_pool:
            x = down_sampling_layer_mpool(filters)(x)
        else:
            x = down_sampling_layer(filters, activation=None)(x)
        x = Activation("relu")(x)
    filters *= 2
    # Bottom
    x = bn_layer()(x, training=bn_training)
    # x = Dropout(dropout_rate)(x)
    with tf.variable_scope("L2_m_unet_full"):
    # with null_context():
        x = conv_layer(filters)(x, training=is_training)
        x = conv_layer(filters)(x, training=is_training)
    if do_dropout:
        x = Dropout(dropout_rate)(x, training=is_training)
    copies=copies[::-1]
    for d in range(depth):
        x = tr_conv_layer(filters)(x, training=is_training)
        filters /= 2
        x = concatenate([copies[d], x])
        x = bn_layer()(x, training=bn_training)
        with tf.variable_scope("L2_m_unet_full"):
        # with null_context():
            x = conv_layer(filters)(x, training=is_training)
            x = conv_layer(filters)(x, training=is_training)
        # x = Dropout(dropout_rate)(x, training=is_training)
    with tf.variable_scope("L2_m_unet_full"):
        x = Conv2D(1, kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=None, **conv_args())(x)
        # x=LeakyReLU()(x)
        # x=ReLU()(x)
        # x = Activation("relu")(x)
        x=bn_layer()(x, training=bn_training)
        x=Conv2D(1, kernel_size=[3,3], strides=(1,1), padding="SAME", activation=None, **conv_args())(x)
        x = Lambda(lambda x: tf.squeeze(x, -1), name="unet_squeeze")(x)
        # x = Flatten()(x)
        # x = Dense(main_cfg.input_shape[1]*main_cfg.input_shape[0],activation=None)(x)
        # x = Dense(main_cfg.input_shape[1]*main_cfg.input_shape[0],activation=None)(x)
        # x = Reshape(main_cfg.input_shape)(x)
    # register_kernel_L2_on_scope("L2_m_unet_full")
    mname="m_unet_full_interm"
    if max_pool:
        mname+="_mpool"
    mod = Model(inputs=[X_in], outputs=[x], name=mname)
    # print(X)
    # x=mod_a(X_in)
    # print(tf.shape(x))
    return mod

def encoder_model_mpool(X, main_cfg: ADDA_CFG, role_cfg,get_layer=False,is_training=False):
    return encoder_model(X, main_cfg, role_cfg,get_layer,is_training, max_pool=True)


def classifier(Z,main_cfg,get_layer=False,is_training=False):
    bn_training=True
    X_in=Input(shape=Z.shape[1:])
    x=X_in
    # x = Flatten()(x)
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    with tf.variable_scope("L2_m_unet_classifier"):
    # with null_context():
        # x = Dense(Z.shape[1]*Z.shape[2],activation=None)(x)
        # x = Dense(Z.shape[1]*Z.shape[2],activation=None)(x)
        x=bn_layer()(x, training=bn_training)
        x=Conv2D(1, kernel_size=[3,3], strides=(1,1), padding="SAME", activation=None, **conv_args())(x)
        # x=LeakyReLU()(x)
    x = Lambda(lambda x: tf.squeeze(x, -1), name="clf_squeeze")(x)
        # x = Lambda(lambda z: z)(x)
    # x = Reshape(Z.shape[1:])(x)
    mod=Model(inputs=[X_in], outputs=[x], name="m_unet_classifier")
    if get_layer:
        return mod,x
    # register_kernel_L2_on_scope("L2_m_unet_classifier", l=l_val)
    return mod


def dense_layer(channels,**kwargs):
    return Dense(channels, **kwargs)
from typing import Callable
@LayerList
def reg_dense(*args, **kwargs):
    return [
        # Dropout(rate=0.3),
        dense_layer(*args,**kwargs)
    ]
def auto_dense_same(*args, **kwargs):
    def _auto_dense(inputs):
        input_shape=inputs.shape[1:]
        x=Flatten()(inputs)
        dense_shape=x.shape[1]
        try:
            x=Dense(dense_shape,*args,**kwargs)(x)
        except BaseException as e:
            print("DSHAPE:", dense_shape)
            raise

        x=Reshape(input_shape)(x)
        return x
    return _auto_dense

def conv_discriminator(Z, main_cfg: ADDA_CFG, get_layer=False, is_training=False, mpool=False):
    if mpool:
        down_sampler=down_sampling_layer_mpool
    else:
        down_sampler=down_sampling_layer
    bn_training=True
    X_in=Input(shape=Z.shape[1:], name="discriminator_input")
    x=X_in
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(x)
    x = Dropout(rate=0.3)(x)
    x = bn_layer()(x)
    # x = conv_layer(2)(x)
    with tf.variable_scope("L2_m_unet_discriminator"):
        x = down_sampler(2)(x)
        x = down_sampler(4)(x)
        x=LeakyReLU()(x)
        x = bn_layer()(x)
        x = down_sampler(4)(x)
        x = down_sampler(2)(x)
        x = LeakyReLU()(x)
        x = bn_layer()(x)
        x = down_sampler(1)(x)
        x = down_sampler(1)(x)
        x=LeakyReLU()(x)
        x = bn_layer()(x)
        x=Conv2D(filters=1, kernel_size=[2, 2], strides=(2,1), padding="SAME", activation=None, **conv_args())(x)
        x = bn_layer()(x)
        x = Flatten()(x)
    register_kernel_L2_on_scope("L2_m_unet_discriminator", l=l_val)
    mod=Model(inputs=[X_in], outputs=[x], name="unet_discriminator")
    if get_layer:
        return mod,x

    return mod
def conv_discriminator_mpool(Z, main_cfg: ADDA_CFG, get_layer=False, is_training=False):
    return conv_discriminator(Z=Z, main_cfg=main_cfg, get_layer=get_layer, is_training=is_training, mpool=True)

def conv_discriminator_PT_T2(Z, main_cfg: ADDA_CFG, get_layer=False, is_training=False):
    bn_training=True
    X_in=Input(shape=Z.shape[1:], name="discriminator_input")
    x=X_in
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(x)
    x = Dropout(rate=0.3)(x)
    x = bn_layer()(x)
    # x = conv_layer(2)(x)
    with tf.variable_scope("L2_m_unet_discriminator"):
        x = down_sampling_layer(2)(x)
        x = down_sampling_layer(4)(x)
        x=LeakyReLU()(x)
        x = bn_layer()(x)
        x = down_sampling_layer(4)(x)
        x = down_sampling_layer(2)(x)
        x = LeakyReLU()(x)
        x = bn_layer()(x)
        x = down_sampling_layer(1)(x)
        x = down_sampling_layer(1)(x)
        x=LeakyReLU()(x)
        x = bn_layer()(x)
        x=Conv2D(filters=1, kernel_size=[2, 2], strides=(2,1), padding="SAME", activation=None, **conv_args())(x)
        x = bn_layer()(x)
        x = Flatten()(x)
    register_kernel_L2_on_scope("L2_m_unet_discriminator", l=l_val)
    mod=Model(inputs=[X_in], outputs=[x], name="unet_discriminator_PT_T2")
    if get_layer:
        return mod,x

    return mod
