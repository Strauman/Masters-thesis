# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.layers import InputLayer,Lambda,Dense
# from tensorflow.python.keras.engine import training_utils
# from tfmodels import USequential, Remember

from sys import exit as xit
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import InputLayer, concatenate

import tensorflow as tf
import numpy as np
from tensorflow.keras import Input
X=tf.placeholder(tf.float32,shape=(1,3,3))
Y=tf.ones_like(X)
# Goal is to concatenate X and Y in the middle of moddeling (the Model API)
# The inputs below should generate placeholders, right?
def mod_a():
    i_X=Input(shape=X.shape[1:])
    first=Lambda(lambda x: x*2, name="1st")(i_X)
    second=Lambda(lambda x: x*2, name="2nd")(first)
    third=Lambda(lambda x: x*2, name="3rd")(second)
    copies=[first]
    enc=Model(inputs=[i_X], outputs=[third,*copies], name="enc")
    return enc

def mod_b(cps):
    cpl=[Input(shape=c.shape[1:]) for c in cps]
    i_Y=Input(shape=Y.shape[1:])
    x=i_Y
    x=tf.keras.layers.concatenate([cpl[0], x], name="cc")
    # for i,cp_l in enumerate(cps):
        # x=tf.keras.layers.concatenate([cp_l, x], name="cc{}".format(i))
    dec=Model(inputs=[i_Y, *cpl], outputs=[x], name="dec")
    return dec

Z,*cps=mod_a()([X])
Ymod=mod_b(cps)
Yhat=Ymod([Z,cps[0]])
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
X_=np.arange(3*3).reshape(1,3,3).astype(np.float32)
# Yhat=mod.call([X,Y])
sess=tf.InteractiveSession()
print(sess.run(Yhat, {X: X_}))
