# import sys
# import os
# parentpath=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
# sys.path.append(parentpath)
from .build_graph import *

dice=metrics.tf_dice_score(Ys,Yhat)
loss=metrics.pixelw_cross_entropy(Ys,Yhat)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)
train_op = tf.group([optimizer, update_ops])

sm_loss=tf.summary.scalar(name="loss", tensor=loss)
sm_dice=tf.summary.scalar(name="dice", tensor=dice)
merged_summaries = tf.summary.merge_all()

stop_val_dice=0.785
def pretrain_epoch(sess, info):
    DS_s.it_init_tr(sess)
    print("dice: {}, loss: {}".format(*sess.run([dice, loss], DS_s.feed_tr)))
    DS_s.it_init_val(sess)
    val_dice_, val_loss_=sess.run([dice, loss], DS_s.feed_val)
    print("val_dice: {}, val_loss: {}".format(val_dice_,val_loss_))
    if val_dice_ > stop_val_dice:
        raise StopTraining("Reached satisfactory dice {}".format(val_dice_))

def pretrain_printval(sess, info):
    DS_s.it_init_val(sess)
    val_dice_, val_loss_, v_summaries_ = sess.run([dice, loss, merged_summaries], DS_s.feed_val)
    print("dice: {}, loss: {}".format(*sess.run([dice, loss], DS_s.feed_tr)))
    print("s_val_dice: {}, s_val_loss: {}".format(val_dice_, val_loss_))
    if val_dice_ > stop_val_dice:
        raise StopTraining("Reached satisfactory dice {}".format(val_dice_))

# color_print("RESETTING training iterator every iter.", style="danger")
def pretrain_iter(sess, info):
    # DS_s.it_init_tr(sess)
    _, _, summaries_ = sess.run([train_op, loss, merged_summaries], DS_s.feed_tr)
    if summaries_ is not None:
        tr_writer.add_summary(summaries_, info.step)


def pretrain_init(sess, info):
    DS_s.it_init_val(sess)
    v_summaries_ = sess.run(merged_summaries, DS_s.feed_val)
    if v_summaries_ is not None:
        val_writer.add_summary(v_summaries_, info.step)


def test_n_store(sess):
    import numpy as np
    """ Loads test set, runs test data and generates downloadable npz-file which can show data by use of test_result_vis.py """
    DS_t.init_handles(sess)
    DS_s.it_init_tst(sess)
    X_, Y_, Yhat_, dice_ = sess.run([Xs, Ys, Yhat, dice], feed_dict=DS_s.feed_tst)
    print("Soft standard dice")
    print(dice_)
    # if util.ask_user("Save np?"):
    np.savez("/root/data/test-results", X=X_, Y=Y_, Yhat=Yhat_, dice=dice_)
if __name__ == '__main__':
    from contextlib import contextmanager

    @hlp.SessionSaver
    def session_saver(sess):
        print("Restoring")
        sess.run(init_op)
        savers.classifier.restore(sess)
        savers.source_map.restore(sess)
        yield sess
        print("Saving")
        savers.classifier.do_save(sess)
        savers.source_map.do_save(sess)

    with session_saver() as sess:
        DS_s.init_handles(sess)
        DS_t.init_handles(sess)
        metrics.pixelw_cross_entropy
        tr_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        train(
            sess=sess, printerval=50, epoch_cb=pretrain_epoch, iter_cb=pretrain_iter, printerval_cb=pretrain_printval, init_cb=pretrain_init
        )

        if "--test" in sys.argv:
            test_n_store(sess)
