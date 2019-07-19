import tensorflow as tf
from .build_graph import *
diceS=metrics.tf_dice_score(Ys,Yhat)
diceT=metrics.tf_dice_score(Yt,YhatT)
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.global_variables_initializer()])
    DS_s.init_handles(sess)
    DS_t.init_handles(sess)
    DS_s.init_it_val(sess)
    DS_t.init_it_val(sess)
    savers.classifier.restore(sess)
    savers.source_map.restore(sess)
    savers.target_map.restore(sess)

    print(sess.run(diceT,{DS_t.tf_it_handle: DS_s.handle_val}))
    DS_s.init_it_val(sess)
    DS_t.init_it_val(sess)
    print(sess.run(diceS,{DS_t.tf_it_handle: DS_t.handle_val}))
