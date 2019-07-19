# import sys
# import os
# parentpath=os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
# sys.path.append(parentpath)
import tensorflow as tf
with tf.get_default_graph().as_default():
    from .build_graph import *
stop_val_acc = 0.95
disc_accs = tf.metrics.accuracy(
    disc_labels,
    Dhat,
    # weights=None,
    # metrics_collections=None,
    # updates_collections=None,
    name="Discriminator_acc"
)
disc_acc=tf.reduce_mean(disc_accs)
lbls=tf.expand_dims(disc_labels,-1)
# print(Dhat.shape,lbls.shape)
# xit()
disc_losses = tf.nn.sigmoid_cross_entropy_with_logits(
    labels=lbls,
    logits=Dhat,
    name="Discriminator_loss"
)
disc_loss=tf.reduce_mean(disc_losses)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(disc_loss, name="Discriminator_optim")
train_op = tf.group([optimizer, update_ops])

sm_loss = tf.summary.scalar(name="discriminator_loss", tensor=disc_loss)
sm_acc = tf.summary.scalar(name="discriminator_accuracy", tensor=disc_acc)
# merged_summaries = tf.summary.merge_all()
color_print("No summaries", style="warning")
merged_summaries=tf.no_op()



def feeds_tr():
    return merge_dicts(DS_s.feed_tr, DS_t.feed_tr)

def feeds_val():
    return merge_dicts(DS_s.feed_val, DS_t.feed_val)

def init_it_trs(sess):
    DS_s.init_it_tr(sess)
    DS_t.init_it_tr(sess)

def init_it_vals(sess):
    DS_s.init_it_val(sess)
    DS_t.init_it_val(sess)

def merge_dicts(*dicts):
    fd = {}
    for d in dicts:
        fd.update(d)
    return fd

def epochs(sess, info):
    print("Discriminator")
    init_it_trs(sess)
    init_it_vals(sess)
    val_acc_ = sess.run(disc_acc, feeds_val())
    if val_acc_ > stop_val_acc:
        raise StopTraining("Reached satisfactory acc {}".format(val_acc_))

def printer(sess, info):
    init_it_vals(sess)
    val_acc_, val_loss_, v_summaries_ = sess.run([disc_acc, disc_loss, merged_summaries], feeds_val())
    # print("Discriminator")
    # try:
    print("acc: {}, loss: {}".format(*sess.run([disc_acc, disc_loss], feeds_tr())))
    # except OutOfRangeError:
        # init_it_trs()
        # print("acc: {}, loss: {}".format(*sess.run([disc_acc, disc_loss], feeds_tr())))

    print("s_val_dice: {}, s_val_loss: {}".format(val_acc_, val_loss_))
    if val_acc_ > stop_val_acc:
        raise StopTraining("Reached satisfactory acc {}".format(val_acc_))

# color_print("RESETTING training iterator every iter.", style="danger")
def iters(sess, info):
    # DS_s.it_init_tr(sess)
    # _, _, summaries_ = sess.run([train_op, disc_loss, merged_summaries], feeds_tr())
    _,disc_loss_=sess.run([train_op,disc_loss], feeds_tr())
    summaries_=None
    if summaries_ is not None:
        tr_writer.add_summary(summaries_, info.step)


def initer(sess, info):
    print("Initing discriminator")
    init_it_trs(sess)
    init_it_vals(sess)
    v_summaries_ = sess.run(merged_summaries, feeds_val())
    if v_summaries_ is not None:
        val_writer.add_summary(v_summaries_, info.step)


def test_n_store(sess):
    import numpy as np
    """ Loads test set, runs test data and generates downloadable npz-file which can show data by use of test_result_vis.py """
    DS_t.init_handles(sess)
    DS_s.init_handles(sess)
    X_, Y_, Yhat_, dice_ = sess.run([Xs, Ys, Yhat, dice], feed_dict=DS_s.feed_tst)
    print("Soft standard dice")
    print(dice_)
    # if util.ask_user("Save np?"):
    np.savez("/root/data/test-results", X=X_, Y=Y_, Yhat=Yhat_, dice=dice_)
if __name__ == '__main__':
    from contextlib import contextmanager

    @hlp.SessionSaver
    def session_saver(sess):
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print("Restoring")
        savers.source_map.restore(sess)
        # savers.discriminator.restore(sess)
        # savers.target_map.restore(sess)
        # savers.classifier.restore(sess)
        yield sess
        print("Saving")
        # savers.target_map.do_save(sess)
        savers.discriminator.do_save(sess)

    with session_saver() as sess:
        DS_t.init_handles(sess)
        DS_s.init_handles(sess)
        # init_it_trs(sess)
        # print(sess.run([Xs,Xt],feeds_tr()))
        # xit()
        tr_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "train"), sess.graph)
        val_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        train(
            sess=sess, printerval=50, epoch_cb=epochs, iter_cb=iters, printerval_cb=printer, init_cb=initer
        )

        if "--test" in sys.argv:
            test_n_store(sess)
