from tfmodels import TFModel
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Conv2DTranspose, Lambda, InputLayer, concatenate, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from . import util
from tfmodels import USequential, Remember
from sys import exit as xit
import tensorflow as tf

Input=tf.keras.Input

def conv_layer(filters,**kwargs):
    # Z = tf.layers.conv2d(Z, filters=filters, kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(lambda2))
    return Conv2D(filters=int(filters), kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=Activation("relu"),**kwargs)


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
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME', data_format=None,**kwargs)


def tr_conv_layer(filters,**kwargs):
    return Conv2DTranspose(int(filters), kernel_size=(1, 1), strides=2, padding='SAME',activation=Activation("relu"),**kwargs)
    # return lambda Z: tf.layers.conv2d_transpose(Z, filters=int(filters), kernel_size=(1, 1), strides=2, padding="SAME", activation=tf.nn.relu, **kwargs)
def encoder_model(X, cfg):
    depth = cfg.depth
    filters = cfg.filters
    copies=[]
    X_in=Input(shape=X.shape[1:], name="Enc_input")
    x=Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    # First
    x=conv_layer(filters)(x)
    x=conv_layer(filters)(x)
    copies.append(x)
    x=bn_layer()(x)
    x=pooling_layer()(x)
    # Second
    filters*=2
    x=conv_layer(filters)(x)
    x=conv_layer(filters)(x)
    copies.append(x)
    x=bn_layer()(x)
    x=pooling_layer()(x)
    # Bottom
    filters*=2
    x=conv_layer(filters)(x)
    x=conv_layer(filters)(x)
    Z=x
    print(Z.shape)
    xit()
    mod=Model(inputs=[X_in], outputs=[Z,*copies], name="Encoder")
    return mod

def decoder_model(Z, cp_tensors,cfg):
    depth = cfg.depth
    filters = cfg.filters* (2**(depth))
    # Set filters to the same as encoder
    print(Z.shape)
    cp_ins=[Input(shape=c.shape[1:], name="cpy{}".format(i)) for i,c in enumerate(cp_tensors[::-1])]
    # print(Z.shape)
    # print(cp_ins[0].shape, cp_ins[1].shape)

    # cp_in=Input(shape=cp_ts.shape[1:], name="cp_ts_in")
    Z_in=Input(shape=Z.shape[1:], name="Dec_input")
    # First up to align with layers
    filters/=2
    x=tr_conv_layer(filters)(Z_in)
    # print(x.shape,cp_ins[0].shape, cp_ins[1].shape)
    # xit()
    y=concatenate([cp_ins[0], x],-1)
    x=conv_layer(filters, name="first1")(y)
    x=conv_layer(filters, name="first2")(x)
    x=bn_layer()(x)
    # 2nd up to align with layers
    filters/=2
    x=tr_conv_layer(filters, name="scnd_up")(x)
    x=concatenate([cp_ins[1], x],-1)
    x=conv_layer(filters, name="scnd1")(x)
    x=conv_layer(filters, name="scnd2")(x)
    x=bn_layer()(x)
    x=conv_layer(1, name="last_decode_map")(x)
    x=Lambda(lambda x: tf.squeeze(x,-1), name="Decoder_squeeze")(x)
    mod=Model(inputs=[Z_in,*cp_ins], outputs=[x], name="Decoder")
    # mod.summary()
    # xit()
    return mod
