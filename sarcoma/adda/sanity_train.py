from . import datasets
import tensorflow as tf
from train import train
from sys import exit as xit
DS = datasets.get_dataset("MNIST_LBL", tf_handle_name="handle_source")
DS.ds_send_tr(batch=10)
tf_it_handle, tf_iterator = DS.make_iterators()
X, Y = tf_iterator.get_next()

DSt = datasets.get_dataset("MNIST_LBL", tf_handle_name="handle_target")
DSt.ds_send_tr(batch=10)
tf_itt_handle, tf_titerator = DSt.make_iterators()
Xt, Yt = tf_titerator.get_next()


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Activation,Input, Lambda
from tfmodels.sanity_model import TFModel as Classifier
clf=Classifier()
Yhat=clf(X)
Yhat_t=clf(Xt)

loss = tf.losses.mean_squared_error(Yhat,Y)
optimizer = tf.train.AdamOptimizer().minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_op=optimizer
# train_op = tf.group([optimizer, update_ops])
Y=tf.cast(Y, tf.float32)
def accuracy(Yhat,Y):
    Ythr=tf.where(tf.greater(Yhat,0.5),tf.ones_like(Yhat),tf.zeros_like(Yhat))
    N=tf.cast(tf.shape(Y)[0], tf.float32)
    return tf.reduce_sum(1-tf.abs(Y-Ythr))/N

acc=accuracy(Yhat,Y)
acc_t=accuracy(Yhat_t,Yt)

def epoch_cb(sess):
    DS.it_init_tr(sess)
    DSt.it_init_tr(sess)
    # DS.it_init_val(sess)
    # print(DS.feed_val)
    # feed=DS.feed_tr
    # print(feed[list(feed.keys())[0]])
    # xit()
    feed={}
    print("acc:", sess.run(acc, DS.feed_tr))
    print("acc target:", sess.run(acc_t, DSt.feed_tr))
    DS.it_init_tr(sess)
    DSt.it_init_tr(sess)
    # print(sess.run(acc, DS.feed_val))
    # print("--"*10)
    # sess.run(tr)
def iter_cb(sess):
    sess.run(train_op, DS.feed_tr)

def init_cb(sess):
    sess.run([init_op, tf.local_variables_initializer()])
    DS.init_handles(sess)
    DSt.init_handles(sess)
    DS.it_init_tr(sess)
    DSt.it_init_tr(sess)


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # x, y = sess.run([X, Y], DS.feed_tr)
    # print(x.shape, y.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(x[0], cmap="bone")
    # print(y[0])
    # plt.show()
    train(
        sess=sess, printerval=50, epoch_cb=epoch_cb, iter_cb=iter_cb, printerval_cb=None, init_cb=init_cb
    )
