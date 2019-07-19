import tensorflow as tf
import os
from . import util


def _train(sess, printerval=50, epoch_cb=None, iter_cb=None, printerval_cb=None, init_cb=None):
    # with tf.get_default_session() as sess:
    info=util.Dottify()
    info.step=0
    info.epoch=1
    if callable(init_cb):
        init_cb(sess,info)
        print("Init done")
    # sess.run(tf.local_variables_initializer())
    while True:
        if callable(epoch_cb):
            info.epoch+=1
            if epoch_cb(sess,info) is False:
                break
        while True:
            try:
                if callable(iter_cb):
                    iter_cb(sess,info)
                if info.step % printerval == 0:
                    if callable(printerval_cb):
                        printerval_cb(sess,info)
                info.step += 1
            except tf.errors.OutOfRangeError:
                break

def train(sess, printerval=50, epoch_cb=None, iter_cb=None, printerval_cb=None, init_cb=None):
    try:
        _train(sess, printerval, epoch_cb, iter_cb, printerval_cb, init_cb)
    except KeyboardInterrupt as e:
        print("KeyboardInterrupt")
        return True
    except StopTraining as e:
        print("Requested to stop training")
        print(str(e))
        return True

#

# if "--test" in sys.argv:
#     test_n_store()
#     xit()
