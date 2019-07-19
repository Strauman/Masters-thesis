from . import datasets
import tensorflow as tf
from train import train
from sys import exit as xit
DS = datasets.get_dataset("MNIST_LBL")
# DS.ds_set_tr("batch")(10)
DS_tr = DS.dataset_(1)
# DS.ds_send_tr(batch=10, shuffle=100)
# DS_tr=DS_tr.batch(10)
# DS.datasets[1]=DS_tr
# DS.datasets[1]=getattr(DS.datasets[1], "batch")(10)
# Auto:
def auto():
    #pylint: disable=W0621
    tr_it=DS.iterator_tr
    tf_it_handle,tf_iterator=DS.tf_iterator_tr
    return tr_it,tf_it_handle,tf_iterator
# Manual

def manual():
    #pylint: disable=W0621
    tr_it = DS_tr.make_initializable_iterator()
    tf_it_handle = tf.placeholder(tf.string, shape=[])
    tf_iterator = tf.data.Iterator.from_string_handle(
        tf_it_handle,
        tr_it.output_types,
        output_shapes=tr_it.output_shapes)
    return tr_it,tf_it_handle,tf_iterator

tr_it,tf_it_handle,tf_iterator=auto()
# tr_it,tf_it_handle,tf_iterator=manual()

X, Y = tf_iterator.get_next()
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run([init_op, tf.local_variables_initializer()])
    h_tr_iter=sess.run(tr_it.string_handle())
    tr_feed = {tf_it_handle: h_tr_iter}
    sess.run(tr_it.initializer, tr_feed)
    x, y = sess.run([X, Y], tr_feed)
    print(x.shape, y.shape)
