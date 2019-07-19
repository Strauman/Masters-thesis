import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda, Input, Flatten, InputLayer
from tensorflow.keras.layers import MaxPooling2D, Activation
from . import UpdModel as Model
from . import LayerList, register_kernel_L2_on_scope
from ..configs import ADDA_CFG
from typing import Callable
from sys import exit as xit
from pprint import pprint as pp
dropout_rate = 0.4
l_val = 0.001
l2_reg = lambda l2=l_val: tf.keras.regularizers.l2(l_val)

@LayerList
def conv_layer(filters,kernel_size,strides=(1,1), **kwargs) -> Callable:
    return [
        # Dropout(dropout_rate),
        Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="VALID", activation=None),
        Activation("relu")
    ]
def pooling_layer(pool_size,strides=None, **kwargs):
    return MaxPooling2D(pool_size=pool_size, strides=strides, padding='SAME', data_format=None, **kwargs)

def lenet_encoder(X, main_cfg: ADDA_CFG, is_training=False, role_cfg=None, also_classify=False):
    X_in = Input(shape=X.shape[1:], name="lenet_input")
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    with tf.variable_scope("Lenet_L2"):
        x=Conv2D(filters=20,kernel_size=5)(x)
        x=pooling_layer(2,2)(x)
        x=Activation("relu")(x)
        x=Conv2D(filters=50,kernel_size=5)(x)
        x=Dropout(0.5)(x, training=is_training)
        x=pooling_layer(2,2)(x)
        x=Activation("relu")(x)
        x=Flatten()(x)
    x=Dense(500, activation=None)(x)
    if also_classify:
        x=Activation("relu")(x)
        x=Dense(10, activation=None)(x)
    register_kernel_L2_on_scope("Lenet_L2", l=l_val)
    return Model(inputs=[X_in], outputs=[x], name="m_lenet")
def lenet_classify_encoder(X, main_cfg: ADDA_CFG, is_training=False, role_cfg=None):
    return lenet_encoder(X=X, main_cfg=main_cfg, is_training=is_training, role_cfg=role_cfg, also_classify=True)


def identity(X, main_cfg: ADDA_CFG, get_layer=False, is_training=False):
    X_in = Input(shape=X.shape[1:], name="identity_input")
    x = Lambda(lambda Z: Z)(X_in)
    return Model(inputs=[X_in], outputs=[x], name="m_identity")

def dense_classifier(X, main_cfg: ADDA_CFG, get_layer=False, is_training=False):
    X_in = Input(shape=X.shape[1:], name="dense_classifier_input")
    x=Dense(500, activation=None)(X_in)
    x=Activation("relu")(x)
    x=Dense(500, activation=None)(x)
    x=Activation("relu")(x)
    x=Dense(10, activation=None)(X_in)
    return Model(inputs=[X_in], outputs=[x], name="m_adda_discriminator")

def discriminator(Z, main_cfg: ADDA_CFG, get_layer=False, is_training=False):
    X_in = Input(shape=Z.shape[1:], name="disc_input")
    x=X_in
    with tf.variable_scope("Lenet_disc_L2"):
        x=Dense(500, activation=None)(X_in)
        x=Activation("relu")(x)
        x=Dense(500, activation=None)(x)
        x=Activation("relu")(x)
        x=Dense(2, activation=None)(x)
    register_kernel_L2_on_scope("Lenet_disc_L2", l=l_val)
    return Model(inputs=[X_in], outputs=[x], name="m_adda_discriminator")

if __name__ == '__main__':
    from ..configs.MNIST_SVHN_LBL import MAIN_CFG as cfg
    Xp = tf.reshape(tf.placeholder(tf.float32), (-1, *cfg.input_shape))
    enc = lenet_encoder(Xp, cfg)
    hidden_shape=enc(Xp).shape[1:]
    # print(hidden_shape)
    # xit()
    Zp = tf.reshape(tf.placeholder(tf.float32), (-1, *hidden_shape))
    clf = dense_classifier(Zp, cfg)
    Z = enc(Xp)
    Y = clf(Z)
    enc.summary()
    clf.summary()
    dsc = discriminator(Z, cfg)
    dsc.summary()

    # clf =
