from .build_graph import *
import tr_discriminator as DS
import tr_targetmap as TM
def train_discriminator(sess):
    return train(
        sess=sess, printerval=50, epoch_cb=DS.epochs, iter_cb=DS.iters, printerval_cb=DS.printer, init_cb=DS.initer
    )
def train_targtemap(sess):
    return train(
        sess=sess, printerval=50, epoch_cb=TM.epochs, iter_cb=TM.iters, printerval_cb=TM.printer, init_cb=TM.initer
    )
if __name__ == '__main__':
    from contextlib import contextmanager

    @hlp.SessionSaver
    def session_saver(sess):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("Restoring")
        savers.source_map.restore(sess)
        savers.discriminator.restore(sess)
        savers.target_map.restore(sess)
        savers.classifier.restore(sess)
        yield sess
        print("Saving")
        savers.target_map.do_save(sess)
        savers.discriminator.do_save(sess)

    with session_saver() as sess:
        import time
        DS_t.init_handles(sess)
        DS_s.init_handles(sess)
        # init_it_trs(sess)
        # print(sess.run([Xs,Xt],feeds_tr()))
        # xit()
        tr_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        print("Training discriminator")
        while True:
            train_discriminator(sess)
            print("switching to target map")
            print("waiting for interrupt")
            time.sleep(0.5)
            train_targtemap(sess)
            print("switching to discriminator")
            print("waiting for interrupt")
            time.sleep(0.5)

        if "--test" in sys.argv:
            test_n_store(sess)
