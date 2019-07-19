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
def conv_layer(filters, **kwargs) -> Callable:
    return [
        Dropout(dropout_rate),
        Conv2D(filters=int(filters), kernel_size=[3, 3], strides=(1, 1), padding="SAME", activation=None),
        Activation("relu")
    ]


def pooling_layer(**kwargs):
    return MaxPooling2D(pool_size=(2, 2), strides=None, padding='SAME', data_format=None, **kwargs)


def bn_layer(**layer_kwargs):
    def wrap(*call_args, **call_kwargs):
        return BatchNormalization(**layer_kwargs)(*call_args, training=True, **call_kwargs)
    return wrap


def encoder(X, main_cfg: ADDA_CFG, get_layer=False, is_training=False):
    X_in = Input(shape=X.shape[1:], name="Enc_input")
    x = Lambda(lambda Z: tf.expand_dims(Z, 3))(X_in)
    #pylint: disable=E1102
    x = conv_layer(8)(x, training=is_training)
    x = conv_layer(8)(x, training=is_training)
    x = pooling_layer()(x)
    x = bn_layer()(x)
    x = conv_layer(1)(x, training=is_training)
    x = pooling_layer()(x)
    x = bn_layer()(x)
    x = Flatten()(x)
    # x = pooling_layer()(x)
    return Model(inputs=[X_in], outputs=[x], name="m_encoder")


def classifier(Z, main_cfg, get_layer=False, is_training=False):
    X_in = Input(shape=Z.shape[1:], name="Enc_input")
    x = X_in
    x = Dense(49, activation='sigmoid')(x)
    x = Dense(main_cfg.trainers.pretrain.num_classes, activation=None)(x)
    return Model(inputs=[X_in], outputs=[x], name="m_classifier")


def discriminator(Z, main_cfg: ADDA_CFG, get_layer=False, is_training=False):
    X_in = Input(shape=Z.shape[1:], name="disc_input")
    x = X_in
    with tf.name_scope("disc_L2"):
        x = Dense(main_cfg.trainers.pretrain.num_classes, activation="sigmoid")(x)
    x = Dropout(dropout_rate)(x, training=is_training)
    with tf.name_scope("disc_L2"):
        x = Dense(main_cfg.trainers.pretrain.num_classes // 2, activation="sigmoid")(x)
        x = Dense(2, activation=None)(x)
    register_kernel_L2_on_scope("disc_L2", l=l_val)

    return Model(inputs=[X_in], outputs=[x], name="m_discriminator")


if __name__ == '__main__':
    from ..configs.MNIST_SVHN_LBL import MAIN_CFG as cfg
    Xp = tf.reshape(tf.placeholder(tf.float32), (-1, *cfg.input_shape))
    Zp = tf.reshape(tf.placeholder(tf.float32), (-1, *cfg.hidden_shape))
    enc = encoder(Xp, cfg)
    clf = classifier(Zp, cfg)
    Z = enc(Xp)
    Y = clf(Z)
    enc.summary()
    clf.summary()
    dsc = discriminator(Z, cfg)
    dsc.summary()

    # clf =
