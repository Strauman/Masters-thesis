import sys
import tensorflow as tf
# import train
from prettytable import PrettyTable
from sys import exit
import numpy as np
from . import metrics
from .metrics import zero_guard

class _Slicer(object):
    def __getitem__(self, val):
        return val
Slicer=_Slicer()

class InsaneError(Exception):
    pass


# class Slicer(object):
#     """docstring for Slicer."""
#     def __init__(self, arg):
#         super(Slicer, self).__init__()
#         self.arg = arg

init_op = tf.group([tf.global_variables_initializer(), tf.local_variables_initializer()])


def test_zero_guard(tensors, assertive_values):
    test_passed = True
    np.set_printoptions(precision=4, floatmode='maxprec')
    with tf.Session() as sess:
        sess.run(init_op)
        for i, (t_, ep) in enumerate(tensors):
            t = tf.cast(t_, tf.float32)
            kw = {}
            if ep is not None:
                kw['epsilon'] = tf.constant(ep)
            result = sess.run(zero_guard(t, **kw))
            correct_result = np.allclose(result, assertive_values[i]) and result > 0
            if not correct_result:
                test_passed = False
            print(
                "{}\t~=\t{}\t:".format(
                    np.array2string(result),
                    assertive_values[i]),
                correct_result)
    if not test_passed:
        raise InsaneError("Sanity check failed for zero_guard")
    else:
        print("zero_guard test succeeded")

def test_scores(fnc):
    # test_shape = (8, 10, 12)
    test_shape = (10, 100, 100)
    ON = tf.ones(test_shape)
    ZO = tf.zeros(test_shape)
    # HLF: Half of pixels are 1's
    hlf = np.ones(test_shape)
    half_slices = Slicer[:, :, test_shape[2] // 2:]
    hlf[half_slices] = 0
    # hlf_half_shape=hlf[half_slices].shape
    # hlf_shape=hlf.shape
    # print(hlf_half_shape, np.prod(hlf_half_shape))
    # print(hlf.shape, np.prod(hlf_shape))
    # print(hlf[hlf==1].shape, hlf[hlf==0].shape)
    # exit()
    # print(np.sum(np.where(hlf == 1)))
    # print(np.sum(np.where(hlf == 0)))
    print("#0 vs #1 in hlf")
    print(np.unique(hlf, return_counts=True))
    inverted = 1 - hlf
    INV = tf.constant(inverted.astype(np.float32))
    HLF = tf.constant(hlf.astype(np.float32))
    # q_ON -> a quarter are ones
    num_ones = np.prod(test_shape) // 4
    random_ones = [np.random.randint(0, shp, num_ones) for shp in test_shape]
    quarter_ones = np.zeros(test_shape)
    quarter_ones[random_ones] = 1
    q_ON = tf.constant(quarter_ones.astype(np.float32))
    q_ZO = 1 - q_ON
    tbl = PrettyTable()
    tbl.field_names = ["Label", "Pediction", "Score/loss"]

    def mixmatch_scores(a, b, a_tx, b_tx):
        tbl.add_row([a_tx, a_tx, sess.run(fnc(a, a))])
        tbl.add_row([b_tx, b_tx, sess.run(fnc(b, b))])
        tbl.add_row([a_tx, b_tx, sess.run(fnc(a, b))])
        tbl.add_row([b_tx, a_tx, sess.run(fnc(b, a))])
    def format_score(score_ts):
        score=sess.run(score_ts)
        score_str=str(score)
        if score < 0 or score > 1:
            score_str+=" OOB"
        return score_str

    with tf.Session() as sess:
        sess.run(init_op)
        print("lbl,pred")
        # tbl.add_row("Half", "Correct")
        tbl.add_row(["100%", "100%", format_score(fnc(ON, ON))])
        tbl.add_row(["0%", "0%", format_score(fnc(ZO, ZO))])

        tbl.add_row(["100%", "0%", format_score(fnc(ON, ZO))])
        tbl.add_row(["0%", "100%", format_score(fnc(ZO, ON))])
        # mixmatch_scores(ON,ZO,"1","0%")

        tbl.add_row(["50%", "100%", format_score(fnc(HLF, ON))])
        tbl.add_row(["50%", "0%", format_score(fnc(HLF, ZO))])

        tbl.add_row(["100%", "50%", format_score(fnc(ON, HLF))])
        tbl.add_row(["0%", "50%", format_score(fnc(ZO, HLF))])

        tbl.add_row(["25%", "100%", format_score(fnc(q_ON, ON))])
        tbl.add_row(["25%", "0%", format_score(fnc(q_ON, ZO))])

        tbl.add_row(["100%", "25%", format_score(fnc(ON, q_ON))])
        tbl.add_row(["0%", "25%", format_score(fnc(ZO, q_ON))])

        tbl.add_row(["75%", "100%", format_score(fnc(q_ZO, ON))])
        tbl.add_row(["75%", "0%", format_score(fnc(q_ZO, ZO))])
        tbl.add_row(["100%", "75%", format_score(fnc(ON, q_ZO))])
        tbl.add_row(["0%", "75%", format_score(fnc(ZO, q_ZO))])
        tbl.add_row(["0.5: 100%", "0", format_score(fnc(ON*0.5, ZO))])
        tbl.add_row(["0.5: 100%", "1", format_score(fnc(ON*0.5, ON))])
        tbl.add_row(["0","0.5: 100%", format_score(fnc(ZO,ON*0.5))])
        tbl.add_row(["1","0.5: 100%", format_score(fnc(ON,ON*0.5))])
        tbl.add_row(["0","0.1: 100%", format_score(fnc(ZO,ON*0.1))])
        tbl.add_row(["1","0.1: 100%", format_score(fnc(ON,ON*0.1))])
        tbl.add_row(["0","0.9: 100%", format_score(fnc(ZO,ON*0.9))])
        tbl.add_row(["1","0.9: 100%", format_score(fnc(ON,ON*0.9))])
        tbl.add_row(["inv", "50%", format_score(fnc(INV, HLF))])
        tbl.add_row(["hlf", "inv", format_score(fnc(HLF, INV))])
        print("Table units: ratio of data that is equalt to 1")
        print(tbl)
        # print("hlf,1", sess.run(fnc(HLF, ON)))


def sanity_check_all():
    eps = 1e-10
    test_zero_guard(
        [(tf.constant(1), None),
         (tf.constant(1), 0.5),
         (tf.constant(0.4), 0.5),
         (tf.constant(10), None),
         (tf.constant(0), None),
         (tf.constant(-10), None)
         ], [
            1, 1, 0.5, 10, eps, eps, eps
        ])


if __name__ == '__main__':
    # Sanity check zero_guard
    # sanity_check_all()
    # test_scores(train.weighted_cross_entropy)
    # test_scores(tf_naive_weighted_dice)
    # test_scores(metrics.weighted_cross_entropy)
    # test_scores(metrics.tf_dice_score)
    test_scores(metrics.confusion_score)
    # test_scores(metrics.hard_dice)
    # test_scores(metrics.weighted_dice)
