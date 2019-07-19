from . import trainhelp as hlp
from .build_graph import *
from trainables.discriminator import Discriminator
from trainables.targetmap import Targetmap
if __name__ == '__main__':
    from contextlib import contextmanager

    @hlp.SessionSaver
    def session_saver(sess):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("Restoring")
        # savers.discriminator.restore(sess)
        # savers.target_map.restore(sess)
        savers.source_map.restore(sess)
        # savers.classifier.restore(sess)
        yield sess
        print("Saving")
        savers.target_map.do_save(sess)
        savers.discriminator.do_save(sess)
    disc_trainer=Discriminator(DS_s=DS_s, DS_t=DS_t, settings=dict(stop_val_acc=0.7))
    tm_trainer=Targetmap(DS_s=DS_s, DS_t=DS_t, settings=dict(stop_val_acc=0.7))
    with session_saver() as sess:
        import time
        disc_trainer.sess_enter(sess)
        tm_trainer.sess_enter(sess)
        # init_it_trs(sess)
        # print(sess.run([Xs,Xt],feeds_tr()))
        # xit()
        # tr_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "train"), sess.graph)
        # val_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        print("Training discriminator")
        while True:
            tm_trainer.train()
            print("switching to discriminator")
            print("waiting for interrupt")
            time.sleep(0.5)
            disc_trainer.train()
            print("switching to target map")
            print("waiting for interrupt")
            time.sleep(0.5)

        if "--test" in sys.argv:
            test_n_store(sess)
