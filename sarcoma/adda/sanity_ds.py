from . import datasets
import tensorflow as tf
from train import train
from sys import exit as xit
DS = datasets.get_dataset("MNIST_LBL")
# DS_tr=DS.dataset_tr
DS.ds_send_tr(batch=10, shuffle=20)
tf_it_handle,tf_iterator=DS.make_iterators()
X, Y = tf_iterator.get_next()

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run([init_op, tf.local_variables_initializer()])
    DS.init_handles(sess)
    DS.it_init_tr(sess)
    x, y = sess.run([X, Y], DS.feed_tr)
    print(x.shape, y.shape)
    import matplotlib.pyplot as plt
    plt.imshow(x[0], cmap="bone")
    print(y[0])
    plt.show()
