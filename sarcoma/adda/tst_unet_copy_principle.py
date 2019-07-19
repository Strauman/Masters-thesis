# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import InputLayer,Lambda,Dense
# from tensorflow.python.keras.engine import training_utils
# from tfmodels import USequential, Remember

from sys import exit as xit
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Dense, Flatten
from tensorflow.keras.layers import InputLayer, concatenate, Activation

import tensorflow as tf
import numpy as np
from tensorflow.keras import Input

from . import datasets, train
DS = datasets.get_dataset("MNIST_LBL", binary=True)
DS.send_tr(batch=1)
DS.send_val(batch=1)
DS.send_val("repeat")
tf_it = DS.make_iterators()

X,Y=tf_it.get_next()
# X = tf.placeholder(tf.float32, shape=(1, 3, 3))
# Y = tf.ones_like(X)
# Goal is to concatenate X and Y in the middle of moddeling (the Model API)
# The inputs below should generate placeholders, right?


def mod_a():
    i_X = Input(shape=X.shape[1:])
    x=Flatten()(i_X)
    x = Dense(10,activation=Activation('sigmoid'))(x)
    x = Dense(1,activation=Activation('sigmoid'))(x)
    z_out=x
    x = Dense(1,activation=Activation('sigmoid'))(x)
    v_out=x
    # x = Lambda(lambda x: tf.squeeze(x,-1))(x)

    enc = Model(inputs=[i_X], outputs=[z_out, v_out], name="enc")
    return enc


def mod_b(z_in,v_in):
    i_Z = Input(shape=z_in.shape[1:])
    i_V = Input(shape=v_in.shape[1:])
    x = Lambda(lambda x: x)(i_Z)
    x = concatenate([x,i_V])
    x = Dense(1, activation=Activation('sigmoid'))(x)
    x = Lambda(lambda x: tf.squeeze(x,-1))(x)
    dec = Model(inputs=[i_Z,i_V], outputs=[x])
    return dec


ma = mod_a()

Z, V = ma(X)

# Yhat=V
mb = mod_b(Z,V)
Yhat=mb([Z,V])

loss = tf.losses.mean_squared_error(Yhat, Y)
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Yhat0 = mb(V0)
# Yhat1 = mb(V1)

# loss0 = tf.losses.mean_squared_error(Yhat0, Y)
# optimizer0 = tf.train.AdamOptimizer().minimize(loss0)
#
# loss1 = tf.losses.mean_squared_error(Yhat1, Y)
# optimizer1 = tf.train.AdamOptimizer().minimize(loss1)


def accuracy(Yhat, Y):
    Ythr = tf.where(tf.greater(Yhat, 0.5), tf.ones_like(Yhat), tf.zeros_like(Yhat))
    N = tf.cast(tf.shape(Y)[0], tf.float32)
    return tf.reduce_sum(1 - tf.abs(Y - Ythr)) / N

Zs=tf.squeeze(Z,-1)
Vs=tf.squeeze(V,-1)
accZ=accuracy(Z,Y)
accV=accuracy(V,Y)
acc = accuracy(Yhat, Y)
# DS=DataSetObject()
# DS.ds_send_tr(batch=10)
# iter=DS.make_iterators()
# # tf iterator handle is now available as DS.tf_it_handle
# X,Y=iter.get_next()
# sess=tf.Session()
# DS.init_handles(sess) # Initialize handles to send do the tf_handle
# DS.it_init_tr(sess) # run initializer on the tr iterator
# x,y=sess.run([X,Y], DS.feed_tr)

def init_cb(sess):
    # print("Starting!")
    DS.init_it_val(sess)
    printerval_cb(sess)
def epoch_cb(sess):
    DS.init_it_tr(sess)
    # printerval_cb(sess)
    # DS.init_it_tr(sess)


def iter_cb(sess):
    sess.run(optimizer, DS.feed_tr)

def printerval_cb(sess):
    # DS.init_it_val(sess)
    v_acc,v_loss=sess.run([acc,loss], DS.feed_val)
    tr_acc,tr_loss=sess.run([acc,loss], DS.feed_tr)
    print("val acc, val loss")
    print(v_acc,v_loss)
    print("tr_acc","tr_loss")
    print(tr_acc,tr_loss)
    # DS.init_it_val(sess)
    print("ZVAcc")
    # print(sess.run([accZ,accV,Y], DS.feed_val))
    z,v,yh,y=sess.run([Z,V,Yhat,Y], DS.feed_val)
    print(y,yh,z,v)


init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    DS.init_handles(sess)
    DS.init_it_tr(sess)
    train.train(
        sess,
        printerval=1000,
        epoch_cb=epoch_cb,
        iter_cb=iter_cb,
        init_cb=init_cb,
        printerval_cb=printerval_cb
    )


# Ymod=mod_b(cps)
# Yhat=Ymod([Z,cps[0]])
# i_Y=Input(shape=Y.shape[1:])


# X_in=InputLayer(input_shape=X.shape[1:])
# cc_input=InputLayer(input_shape=Y.shape[1:])

# seq.add(mod)
# Concatinate X and Y
# aux=Lambda(lambda x: x)
# conc=concatenate([seq.output,aux.output])
# seq2=Sequential([conc])
# Make model, which will not work this way

# Now I want to send in my X-values
# X_=np.arange(3*3).reshape(1,3,3).astype(np.float32)
# Yhat=mod.call([X,Y])
# sess=tf.InteractiveSession()
# print(sess.run(Yhat, {X: X_}))
