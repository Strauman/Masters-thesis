from . import TFModel
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Conv2DTranspose, Lambda, InputLayer, concatenate, BatchNormalization,Dropout, Dense,Flatten
from tensorflow.keras.layers import Reshape
# from tensorflow.keras.models import Sequential
from . import UpdModel as Model
from . import util
# from tfmodels import USequential, Remember
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

def shallow_dense_encoder(X, main_cfg: ADDA_CFG,role_cfg=None, get_layer=False,is_training=False):
    vs_name=tf.get_variable_scope().name
    cfg=main_cfg
    X_in = Input(shape=main_cfg.input_shape, name="Enc_input")
    x=Flatten()(X_in)
    # x=Dense(cfg.input_shape[1]*cfg.input_shape[0],activation="sigmoid")(x)
    x=Dense(cfg.input_shape[1]*cfg.input_shape[0],activation=None)(x)
    x = Reshape(main_cfg.input_shape)(x)
    mod = Model(inputs=[X_in], outputs=[x], name="m_dense_encoder")
    return mod

def naive_dense_discriminator(Z, main_cfg: ADDA_CFG,role_cfg=None, get_layer=False, is_training=False):
    bn_training=True
    X_in=Input(shape=Z.shape[1:], name="discriminator_input")
    x=X_in
    # x = Lambda(lambda Z: tf.expand_dims(Z, 3))(x)
    # x = Dropout(rate=0.3)(x)
    # x = bn_layer()(x)
    # x = conv_layer(2)(x)
    with tf.variable_scope("L2_m_unet_discriminator"):
        x = Flatten()(x)
        x = Dropout(dropout_rate)(x, training=is_training)
        # x = bn_layer(name=f"bn_disc_1")(x, training=bn_training)
        # x = Lambda(lambda Z: tf.expand_dims(Z, 2))(x)
        # x = conv_layer(8)(x, training=is_training)#pylint: disable=E1102
        # x = Lambda(lambda z: tf.squeeze(z, -1), name="unet_squeeze")(x)
        x = Dense(64*64, activation='sigmoid')(x)
        # x = Dense(128*128, activation='sigmoid')(x)
        # x = bn_layer()(x, training=bn_training)
        # x = Dense(14*14, activation='sigmoid')(x)
        x = Dense(2,activation=None)(x)
    register_kernel_L2_on_scope("L2_m_unet_discriminator", l=l_val)
    mod=Model(inputs=[X_in], outputs=[x], name="unet_discriminator")
    if get_layer:
        return mod,x

    return mod

def identity_model(Z,main_cfg,role_cfg=None,get_layer=False,is_training=False):
    X_in=Input(shape=Z.shape[1:])
    x=X_in
    x = Flatten()(x)
    x = Lambda(lambda z: z)(x)
    x = Reshape(Z.shape[1:])(x)
    mod=Model(inputs=[X_in], outputs=[x], name="m_unet_classifier")
    if get_layer:
        return mod,x
    # register_kernel_L2_on_scope("L2_m_unet_classifier", l=l_val)
    return mod
