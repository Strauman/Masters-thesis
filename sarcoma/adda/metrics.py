import tensorflow as tf
from sys import exit
from .helpers.trainhelp import sum_epoch
import numpy as np
epsilon=tf.constant(1e-10)


def nancheck(tensor, tx=""):
    return tf.Print(tensor, [tx, "Nans?",tf.is_numeric_tensor(tensor)])

def tfprint(tensor, **kwargs):
    return tf.Print(tensor, [tensor], **kwargs)

def zero_guard(expr, epsilon=1e-10):
    e=tf.cast(tf.constant(epsilon), expr.dtype)
    return tf.nn.relu(expr - e) + e

def one_ratio(X):
    N=X.get_shape()[1]*X.get_shape()[2]
    return tf.reduce_mean(tf.reduce_sum(X, axis=(1,2))/N)

def accuracy(labels,predictions,threshold=0.5):
    # labels=tf.cast(labels, tf.int32)
    # predictions=tf.cast(predictions, tf.int32)
    Ythr=tf.where(tf.greater(predictions,threshold),tf.ones_like(predictions),tf.zeros_like(predictions))
    # N=tf.cast(tf.shape(labels)[0], tf.float32)
    return 1-tf.abs(labels-Ythr)

def binary_crossentropy(labels, predictions):
    labels=tf.cast(labels,tf.float32)
    return -tf.reduce_sum(labels*tf.log(zero_guard(predictions))+(1-labels)*tf.log(zero_guard(1-predictions)), axis=(1,2))

def pixelw_cross_entropy_custom(labels, preds):
    # ll_mtx=tf.ones_like(labels)
    # Remove all zero values
    # ll_mtx=preds*ll_mtx
    # eql=tf.equal(labels,c_one)
    float_labels = tf.cast(labels, tf.float32)
    cross_entropy = -1 * (float_labels * tf.log(tf.clip_by_value(preds, 1e-5, 1.0)) + ((1 - float_labels) * tf.log(1 - tf.clip_by_value(preds, 0, (1.0) - (1e-5)))))
    # cross_entropy=-1*(1-float_labels)*tf.log(1-tf.clip_by_value(preds,0,(1.0)-(1e-5)))
    return tf.reduce_sum(cross_entropy,axis=(1,2))


def weighted_cross_entropy(int_labels, preds):
    epsilon=1e-5
    labels = tf.cast(int_labels, tf.float32)
    # H,W=labels.get_shape()[1], labels.get_shape()[2]
    H,W=tf.shape(labels)[1],tf.shape(labels)[2]
    N = H*W
    N = tf.cast(N, tf.float32)
    predsum = zero_guard(tf.reduce_sum(preds,axis=(1,2)))
    W = ((N / predsum) - 1.)
    W = tf.expand_dims(tf.expand_dims(W, 1), 2)
    true_class_log = W * labels * tf.log(tf.clip_by_value(preds, epsilon, 1.0))
    false_class_log = (1 - labels) * tf.log(1 - tf.clip_by_value(preds, 0, (1.0) - (epsilon)))
    false_class_log=tf.check_numerics(false_class_log,"FLX")
    cross_entropy = -1 / N * (true_class_log + false_class_log)
    cross_entropy=tf.check_numerics(cross_entropy,"WCE")
    # cross_entropy= tf.Print(cross_entropy, ["zrona", "Nans?",tf.reduce_any(tf.is_nan(tf.ones((1,2,3))/tf.zeros((1))))])
    return tf.reduce_mean(cross_entropy)


def weighted_dice(label_,pred,debug=False):
    label=tf.cast(label_,tf.float32)
    label,pred=(pred,label)
    if debug:
        label=assert_in_range(label)
        pred=assert_in_range(pred)
    # https://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/loss_segmentation.html
    # https://arxiv.org/pdf/1707.03237.pdf
    # Calculcate the weights
    # H,W=labels.get_shape()[1], labels.get_shape()[2]
    # H,W=tf.shape(label)[1],tf.shape(label)[2]
    # N = H*W
    # r = tf.reshape(label, (None, N))
    # For class 1 (l=1)
    W1 = 1 / zero_guard(tf.reduce_sum(label, axis=(1, 2))**2)
    p1RP = tf.reduce_sum(label * pred, axis=(1, 2))
    s1RP = tf.reduce_sum(label + pred, axis=(1, 2))

    nom = W1 * p1RP
    denom = W1 * s1RP
    # For class 1 (l=0)
    W0 = 1 / zero_guard(tf.reduce_sum(1 - label, axis=(1, 2))**2)
    p0RP = tf.reduce_sum((1 - label) * (1 - pred), axis=(1, 2))
    s0RP = tf.reduce_sum((1 - label) + (1 - pred), axis=(1, 2))
    nom += W0 * p0RP
    denom += W0 * s0RP
    score = 2 * nom / zero_guard(denom)
    return tf.reduce_mean(score)
    # print(W1.get_shape())
    # W0 = W1 / zero_guard((W1 * N - 1)**2)
    # W0 = W1 / zero_guard((W1*N-1)*(W1*N-1))
    # Weight for class 0 and 1

def assert_in_range(arr,min=0.,max=1.):
    assertions=[]
    assertions.append(tf.assert_less_equal(tf.reduce_max(arr), tf.ones_like(arr)*tf.cast(max, arr.dtype)))
    assertions.append(tf.assert_greater_equal(tf.reduce_min(arr), tf.ones_like(arr)*tf.cast(min, arr.dtype)))
    with tf.control_dependencies(assertions):
        arr=tf.identity(arr)
    return arr

def tf_dice_score(A, B, debug=False):
    # if debug:
    #     tf.control_dependencies([tf.assert_less_equal(x, y)]).__enter__()
    # if debug:
    #     tf.control_dependencies([tf.assert_less_equal(x, y)]).__enter__()
    if debug:
        A=assert_in_range(A)
        B=assert_in_range(B)
    # with tf.control_dependencies([tf.less_equal(tf.reduce_max(A), min), [A],tf.greater_equal(tf.reduce_min(B), max), [B]]):
    A = tf.cast(A, tf.float32)
    B = tf.cast(B, tf.float32)
    ApB=tf.reduce_sum(A, axis=(1, 2)) + tf.reduce_sum(B, axis=(1, 2))
    AtB=tf.reduce_sum(A * B, axis=(1, 2))
    epsilon=1e-10
    dice_score=tf.reduce_mean(zero_guard(2 *AtB) / zero_guard(ApB))
    # if debug: assert_in_range(dice_score)
    # dice_score=assert_in_range(dice_score)
    return dice_score
def tf_hard_dice_score(labels, predictions, threshold=0.5):
    hard_predictions=tf.where(tf.greater(predictions,threshold), tf.ones_like(predictions), tf.zeros_like(predictions))
    return tf_dice_score(labels, predictions)
def generalized_dice_loss(labels, logits, debug=False):
    if debug:
        labels=assert_in_range(labels)
        logits=assert_in_range(logits)
    smooth = 1e-17
    shape = tf.TensorShape(logits.shape).as_list()
    depth = int(shape[-1])
    labels = tf.one_hot(labels, depth, dtype=tf.float32)
    logits = tf.nn.softmax(logits)
    weights = 1.0 / (tf.reduce_sum(labels, axis=[0, 1, 2])**2)

    numerator = tf.reduce_sum(labels * logits, axis=[0, 1, 2])
    numerator = tf.reduce_sum(weights * numerator)

    denominator = tf.reduce_sum(labels + logits, axis=[0, 1, 2])
    denominator = tf.reduce_sum(weights * denominator)

    loss = 1.0 - 2.0*(numerator + smooth)/(denominator + smooth)
    return loss

def hard_dice(labels, logits, threshold=0.5, debug=False):
    logits=tf.where(tf.greater(logits,threshold), tf.ones_like(logits), tf.zeros_like(logits))
    return tf.reduce_mean(generalized_dice_loss(labels,logits, debug=debug))
# def hard_dice(labels, predictions, threshold=0.5, debug=False):
#     # if debug:
#     #     tf.control_dependencies([tf.assert_less_equal(x, y)]).__enter__()
#     # if debug:
#     #     tf.control_dependencies([tf.assert_less_equal(x, y)]).__enter__()
#
#     if debug:
#         labels=assert_in_range(labels)
#         predictions=assert_in_range(predictions)
#     # with tf.control_dependencies([tf.less_equal(tf.reduce_max(A), min), [A],tf.greater_equal(tf.reduce_min(B), max), [B]]):
#     A=labels
#     B=tf.where(tf.greater(predictions,threshold), tf.ones_like(labels), tf.zeros_like(labels))
#     # A = tf.cast(A, tf.float32)
#     # B = tf.cast(B, tf.float32)
#     ApB=tf.reduce_sum(A, axis=(1, 2)) + tf.reduce_sum(B, axis=(1, 2))
#     AtB=tf.reduce_sum(A * B, axis=(1, 2))
#     epsilon=1e-10
#     dice_score=tf.reduce_mean(zero_guard(2 *AtB) / zero_guard(ApB))
#     # if debug: assert_in_range(dice_score)
#     # dice_score=assert_in_range(dice_score)
#     return dice_score
def soft_dice(A,B,debug=False):
    if debug:
        A=assert_in_range(A)
        B=assert_in_range(B)
    A = tf.cast(A, tf.float32)
    B = tf.cast(B, tf.float32)
    nominator=2*tf.reduce_sum(A * B, axis=(1, 2))
    denominator=tf.reduce_sum(A**2, axis=(1, 2)) + tf.reduce_sum(B**2, axis=(1, 2))
    return tf.reduce_mean(nominator/zero_guard(denominator))


def tf_positive_weighted_dice(label,pred,alpha=0.0):
    """
    Weights dice score as (#positive/N)*(1-alpha)+alpha.
    That is: weighted based on how many pixels in positive classs.
    If only negative class in labels then highest possible score is alpha
    which is attained only if all correctly classified as 0.
    If all labels are positive class, the highest attainable score is 1.0.
    """
    lbl = tf.cast(label, tf.float32)
    Pred = tf.cast(pred, tf.float32)
    lbl_sum=tf.reduce_sum(lbl, axis=(1, 2))
    N=tf.cast(label.get_shape()[1]*label.get_shape()[2], tf.float32)
    lblpPred=lbl_sum + tf.reduce_sum(Pred, axis=(1, 2))
    lbltPred=tf.reduce_sum(lbl * Pred, axis=(1, 2))
    epsilon=1e-10
    W=alpha+(tf.reduce_sum(lbl, axis=(1,2))*(1-alpha))/N
    dices=(tf.nn.relu(2 *lbltPred-epsilon)+epsilon) / (tf.nn.relu(lblpPred-epsilon)+epsilon)
    dice=dices*W
    return tf.reduce_mean(dice)

def f1_score(labels, soft_predictions, threshold=0.5):
    labels=tf.cast(labels, tf.float32)
    is_negative=1-labels

    h_predictions=tf.where(tf.greater_equal(soft_predictions, threshold), tf.ones_like(soft_predictions), tf.zeros_like(soft_predictions))
    num_positives=tf.reduce_sum(labels)

    num_true_positives=tf.reduce_sum(tf.where(tf.equal(labels*2, h_predictions+1), tf.ones_like(labels), tf.zeros_like(labels)))
    num_false_positives=tf.reduce_sum(tf.where(tf.equal(is_negative*2, h_predictions+1), tf.ones_like(h_predictions), tf.zeros_like(h_predictions)))

    return 2*num_true_positives/(zero_guard(num_true_positives+num_positives+num_false_positives))
def indiv_f1(labels,soft_predictions, threshold=0.5):
    labels=tf.cast(labels, tf.float32)
    is_negative=1-labels

    h_predictions=tf.where(tf.greater_equal(soft_predictions, threshold), tf.ones_like(soft_predictions), tf.zeros_like(soft_predictions))
    num_positives=tf.reduce_sum(labels,axis=[1,2])

    num_true_positives=tf.reduce_sum(tf.where(tf.equal(labels*2, h_predictions+1), tf.ones_like(labels), tf.zeros_like(labels)),axis=[1,2])
    num_false_positives=tf.reduce_sum(tf.where(tf.equal(is_negative*2, h_predictions+1), tf.ones_like(h_predictions), tf.zeros_like(h_predictions)),axis=[1,2])
    f1_scores=2*num_true_positives/(zero_guard(num_true_positives+num_positives+num_false_positives))

    # f1_scores=tf.reshape(f1_scores, (labels.get_shape()[0],))
    return f1_scores

def indiv_acc(labels, soft_predictions, threshold=0.5):
    labels=tf.cast(labels, tf.float32)
    h_predictions=tf.where(tf.greater_equal(soft_predictions, threshold), tf.ones_like(soft_predictions), tf.zeros_like(soft_predictions))
    Ns=tf.reduce_sum(tf.ones_like(labels), axis=[1,2])
    num_correct=tf.reduce_sum(tf.where(tf.equal(labels, h_predictions), tf.ones_like(labels), tf.zeros_like(labels)),axis=[1,2])
    acc=num_correct/Ns
    # acc=tf.reshape(acc,(labels.get_shape()[0],))
    return acc



def f1_for_thresholds():
    pass

def confusion_matrix(labels,predictions, dtype_out=tf.float32):
    _,fp=tf.metrics.false_positives(labels, predictions)
    _,fn=tf.metrics.false_negatives(labels, predictions)
    _,tp=tf.metrics.true_positives(labels, predictions)
    _,tn=tf.metrics.true_negatives(labels, predictions)
    fp=tf.cast(fp, dtype_out)
    fn=tf.cast(fn, dtype_out)
    tp=tf.cast(tp, dtype_out)
    tn=tf.cast(tn, dtype_out)
    return tp,tn,fp,fn

def confusion_score(labels,predictions):
    # pre_local=set(tf.local_variables())
    tp, tn, fp, fn = confusion_matrix(labels=labels, predictions=predictions)
    # post_local=set(tf.local_variables())-pre_local
    # tf.get_default_session().run(tf.variables_initializer(post_local))
    neg=fp+tn
    pos=tp+fn
    pos_pred=tp+fp
    neg_pred=fn+tn
    recall = tf.where(tf.greater(pos,0), tp / pos,0)
    specificiy = tf.where(tf.greater(neg,0), tn / neg,0)
    # tn_rate = tf.where(tf.greater(tp + fn,0), tn / (tp + fn),0)
    # confusion = (recall + specificiy)/2
    # confusion = (tf.abs(specificiy-0.5) + tf.abs(recall-0.5))
    # confusion = 2/((1/specificiy) + (1/recall))
    # return 2*tp/(zero_guard(tp+pos+fp))
    return tf.abs(recall-0.5)+tf.abs(specificiy-0.5)

def confusion_matrix_for_thresholds(labels,predictions,thresholds,feed):
    pre_local=set(tf.local_variables())
    _,b_true_positives = tf.metrics.true_positives_at_thresholds(labels, predictions, thresholds)
    _,b_true_negatives = tf.metrics.true_negatives_at_thresholds(labels, predictions, thresholds)
    _,b_false_positives = tf.metrics.false_positives_at_thresholds(labels, predictions, thresholds)
    _,b_false_negatives = tf.metrics.false_negatives_at_thresholds(labels, predictions, thresholds)
    cmtx_list=[b_true_positives,b_true_negatives,b_false_positives,b_false_negatives]
    post_local=set(tf.local_variables())-pre_local
    tf.get_default_session().run(tf.variables_initializer(post_local))
    # tf.get_default_session().run(tf.local_variables_initializer())
    true_positives,true_negatives,false_positives,false_negatives=sum_epoch(cmtx_list, feed)
    positives=true_positives+false_negatives
    negatives=true_negatives+false_positives
    return true_positives,true_negatives,false_positives,false_negatives,positives,negatives



def best_f1_for_thresholds(labels, predictions, num_thresholds, feed):
    thresholds=np.linspace(0,1,num_thresholds, dtype=np.float32)
    true_positives,_,false_positives,_,positives,_=confusion_matrix_for_thresholds(labels, predictions, thresholds, feed)
    eps=np.finfo(np.float32).eps
    f1_scores=2*true_positives/(np.clip(true_positives+positives+false_positives, eps, np.inf))
    th_idx=np.argmax(f1_scores)
    threshold=thresholds[th_idx]
    f1_max=f1_scores[th_idx]
    return f1_max, threshold

# def sigmoid_pixelw_cross_entropy(preds):
#     # float_labels=tf.cast(labels, tf.float32)
#     cross_entropy=
# def pixelwise_cross_entropy(y, yhat):
#     yshape=tf.shape(y)
#     logits_shape=[yshape[0], yshape[1]*yshape[2]]
#     labels_shape=logits_shape
#     logits=tf.reshape(self.Yhat, logits_shape)
#     labels=tf.reshape(self.Ys,labels_shape)
#     self.loss=tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
