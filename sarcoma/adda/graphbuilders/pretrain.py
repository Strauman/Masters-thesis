from ..trainables import _pretrain
from .. import metrics
import tensorflow as tf
from sys import exit as xit
# from pout import v as pp
from pprint import pprint as pp
from ..helpers.trainhelp import MeanAccumulator, ensure_list, TF_Bestie

class Pretrain(_pretrain._Pretrain):
    @property
    def outputs(self):
        return tf.where(tf.greater(self.Yhat, self.output_threshold), tf.ones_like(self.Yhat), tf.zeros_like(self.Yhat))
    @property
    def hidden(self):
        return tf.where(tf.greater(self.Zs, self.output_threshold), tf.ones_like(self.Zs), tf.zeros_like(self.Zs))

    def predict(self):
        self.init_it_val()
        return self._sess.run([self.inputs, self.Ys, self.outputs], self.feeds_val())

    def get_f1_at_thresh(self, thresh):
        return metrics.f1_score(labels=self.labels, soft_predictions=self.probabilities, threshold=thresh)

    def get_indiv_f1_at_out_at_thresh(self, thresh):
        return metrics.indiv_f1(labels=self.labels, soft_predictions=self.probabilities, threshold=thresh)

    def indiv_accuracy_at_threshold(self, thresh):
        return metrics.indiv_acc(labels=self.labels, soft_predictions=self.probabilities, threshold=thresh)

    def accuracy_at_threshold(self, thresh):
        return tf.reduce_mean(self.indiv_accuracy_at_threshold(thresh))

    @property
    def auc(self):
        return self.get_epoch_value_val(self._auc,max_summary_steps=self.opts.max_summary_steps)
    def build_graph(self):
        #pylint: disable=W0201
        self.average_losses = []
        self.output_threshold=0.5
        # self.loss=
        # yshape=tf.shape(self.Yhat)
        # labels_shape=[yshape[0], yshape[1]*yshape[2]]
        # labels=tf.reshape(self.Ys,labels_shape)
        # Softmax
        # logits_shape=[*labels_shape, 1]
        # logits=tf.reshape(self.Yhat, logits_shape)
        # logits=tf.concat([logits,1-logits], axis=-1)
        # self.dice = metrics.hard_dice(labels=labels,logits=self.Yhat)
        # loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=self.Yhat)
        # Sigmoid
        # self.dice = metrics.hard_dice(labels=self.Ys, predictions=tf.math.sigmoid(self.Yhat))
        # logits_shape=labels_shape
        # logits=tf.reshape(self.Yhat, logits_shape)
        yshape = tf.shape(self.Yhat)
        Yhat_pred = tf.math.sigmoid(self.Yhat)
        Yhat_hard_pred = tf.where(tf.greater(Yhat_pred, 0.5), tf.ones_like(Yhat_pred), tf.zeros_like(Yhat_pred))
        labels = tf.reshape(self.Ys, [-1, yshape[1] * yshape[2]])
        logits = tf.reshape(self.Yhat, [-1, yshape[1] * yshape[2]])

        soft_predictions = tf.math.sigmoid(self.Yhat)
        raveled_predictions = tf.math.sigmoid(logits)
        hard_predictions = tf.where(tf.greater(soft_predictions, 0.5), tf.ones_like(soft_predictions), tf.zeros_like(soft_predictions))

        self.labels=self.Ys
        self.probabilities=soft_predictions
        self.r_labels=labels
        self.predictions=Yhat_hard_pred
        self.r_probabilities=raveled_predictions

        # Custom loss
        # loss=metrics.weighted_dice(self.Ys, tf.math.sigmoid(self.Yhat))
        loss = metrics.pixelw_cross_entropy_custom(labels=self.Ys, preds=Yhat_pred)
        # loss=metrics.binary_crossentropy(labels=self.Ys, predictions=Yhat_pred)
        # loss=metrics.weighted_cross_entropy(int_labels=self.Ys, preds=Yhat_pred)
        # Sigmoid cross entropy loss
        # loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
        # Softmax cross entropy loss
        # loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.cast(labels, tf.float32), logits=logits)
        # loss = tf.losses.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)
        self.dice = metrics.tf_dice_score(self.Ys, hard_predictions)
        self.accuracy = tf.reduce_mean(tf.cast(tf.math.equal(tf.cast(self.Ys, tf.int32), tf.cast(Yhat_hard_pred, tf.int32)), tf.float32))
        _,_auc=tf.metrics.auc(labels=labels, predictions=raveled_predictions)
        self._auc=MeanAccumulator(_auc)
        # self._auc=_auc
        # self.f1,self.f1_update_op = tf.contrib.metrics.f1_score(
        #     self.Ys,
        #     Yhat_hard_pred,
        #     weights=None,
        #     num_thresholds=200,
        #     metrics_collections=None,
        #     updates_collections=None,
        #     name=None
        # )
        self.f1 = metrics.f1_score(labels=self.labels, soft_predictions=self.probabilities, threshold=0.5)
        self.indiv_f1 = metrics.indiv_f1(labels=self.labels, soft_predictions=self.probabilities, threshold=0.5)
        # self.f1 =tf.reduce_mean(self.indiv_f1)

        self.num_positives = tf.reduce_mean(tf.cast(tf.math.equal(tf.cast(Yhat_hard_pred, tf.int32), tf.ones_like(Yhat_hard_pred, tf.int32)), tf.float32))
        self.soft_dice = metrics.tf_dice_score(self.Ys, soft_predictions)
        # Both
        self.loss = tf.reduce_mean(loss)

        # self.dice = metrics.tf_dice_score(self.Ys, tf.math.softmax(logits)[...,0])
        # labels_1d=tf.squeeze(tf.reshape(labels, (-1,1)))
        # un,_,cnt=tf.unique_with_counts(labels_1d)
        # labels=tf.Print(labels,[un,cnt])

        # loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits)

        # self.loss=metrics.pixelw_cross_entropy_custom(self.Ys, tf.math.sigmoid(self.Yhat))
        # check_nans=lambda *tensors: [tf.math.reduce_any(tf.math.is_nan(tf.cast(t, tf.float32))) for t in tensors]
        # self.loss=tf.Print(self.loss,check_nans(logits,labels,loss,self.loss), message="NANS LOGITS,LABELS,LOSS,LOSS_MEAN?")
        # print(tf.s    hape(self.loss))

        # self.loss = metrics.pixelw_cross_entropy(self.Ys, self.Yhat)

        self.optimizer = self.get_optimizer()
        loss_sum = self.loss
        if self.model_losses:
            loss_sum += self.model_losses
        # print(f"{self.name} train vars")
        # pp(self._train_vars)
        self.minimizer = self.optimizer.minimize(loss_sum, global_step=self.global_step, var_list=self.train_vars)
        # pp(self.train_vars)
        # pp(self.model_losses)
        # xit()
        self.besties=[TF_Bestie(trainable=self, tf_op=self.f1, comparer=">", init_val=0, param_state="best", name="best_f1",verbose=True)]
        self.add_summary(self.dice, "dice")
        self.add_summary(self.accuracy, "accuracy")
        self.add_summary(self.soft_dice, "soft dice")
        self.add_summary(self.num_positives, "Positives fraction")
        self.add_summary(self.loss, "loss")
        self.add_summary(self.f1, "F1_score")
        # print(tf.shape(self.dice),self.dice.shape)
        # print(self.loss.shape)
        # xit()
        self.performance_measure = self.f1
        # self.ignore_use(self.update_ops)
        # pp(self.update_ops)
        self.train_op = tf.group([self.minimizer, self.update_ops])
