import tensorflow as tf
from ..trainables._targetmap import _Targetmap, TM_Bestie
from ..helpers.trainhelp import MeanAccumulator, optimal_auc_cutoff
from .. import metrics
class Targetmap(_Targetmap):  # pylint: disable=W0212
    # def build_graph(self):
    #     #pylint: disable=W0201
    #     # First the discriminator part
    #     self.tm_disc_labels=tf.cast(self.tm_disc_labels,tf.int32)
    #     disc_loss = tf.losses.sparse_softmax_cross_entropy(
    #         labels=self.tm_disc_labels,
    #         logits=self.Dhat_t
    #     )
    #     disc_acc = tf.reduce_mean(tf.cast(tf.math.equal(self.tm_disc_labels, tf.argmax(tf.cast(self.Dhat_t, tf.int64), axis=1)), tf.float32))
    #     self.loss=disc_loss
    #     self.disc_acc=disc_acc

    # def get_optimal_f1(self):
    #     self.init_it_val()
    #     optimal_cutoff=optimal_auc_cutoff(labels=self.labels, predictions=self.raveled_predictions, num_thresholds=200, feed=self.feeds_val())
    #     print("OPTIMAL thresh:", optimal_cutoff)
    #     self.init_it_val()
    #     f1=metrics.f1_score(labels=self.Yt, soft_predictions=self.soft_predictions, threshold=optimal_cutoff)
    #     optimal_f1=self._sess.run(f1, self.feeds_val())
    #     return optimal_f1

    @property
    def outputs(self):
        return tf.where(tf.greater(self.YhatT, self.output_threshold), tf.ones_like(self.YhatT), tf.zeros_like(self.YhatT))
    def get_indiv_f1_at_out_at_thresh(self, thresh):
        return metrics.indiv_f1(labels=self.labels, soft_predictions=self.probabilities, threshold=thresh)
    @property
    def indiv_f1_at_out_thresh(self):
        return metrics.indiv_f1(labels=self.labels, soft_predictions=self.probabilities, threshold=self.output_threshold)
    @property
    def hidden(self):
        return tf.where(tf.greater(self.Zs, self.output_threshold), tf.ones_like(self.Zs), tf.zeros_like(self.Zs))
    def clf_accuracy_at_threshold(self,threshold):
        return tf.reduce_mean(metrics.indiv_acc(labels=self.labels, soft_predictions=self.probabilities, threshold=threshold))
    def predict(self):
        self.init_it_val()
        return self._sess.run([self.inputs, self.Yt, self.outputs], self.feeds_val())

    def build_graph(self):
        #pylint: disable=W0201
        print("Building label graph")
        self.output_threshold=0.5
        # Setup discriminator training
        self.tm_disc_labels=tf.cast(self.tm_disc_labels,tf.int32)
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.tm_disc_labels,
            logits=self.Dhat_t
        )
        self.metric_names.avg_loss="discriminator_avg_loss"
        # self.disc_acc = tf.reduce_mean(tf.cast(tf.math.equal(self.tm_disc_labels, tf.argmax(tf.cast(self.Dhat_t, tf.int64), axis=1)), tf.float32))
        self.preds = tf.cast(tf.argmax(self.Dhat_t, axis=1),tf.int32)
        self.disc_acc = tf.reduce_mean(tf.cast(tf.math.equal(self.tm_disc_labels, self.preds), tf.float32))
        # self.disc_acc
        # Get the ground truth labels for performance measure
        yshape=tf.shape(self.YhatT)
        r_labels=tf.reshape(self.Yt,[-1,yshape[1]*yshape[2]])
        logits=tf.reshape(self.YhatT,[-1,yshape[1]*yshape[2]])

        self.raveled_predictions=raveled_predictions=tf.math.sigmoid(logits)
        self.YhatT_hard_pred=YhatT_hard_pred=tf.where(tf.greater(self.YhatT, 0.5), tf.ones_like(self.YhatT), tf.zeros_like(self.YhatT))
        soft_predictions=tf.math.sigmoid(self.YhatT)
        hard_predictions=tf.where(tf.greater(soft_predictions,0.5), tf.ones_like(soft_predictions), tf.zeros_like(soft_predictions))
        self.labels=self.Yt
        self.hard_predictions=hard_predictions
        self.probabilities=soft_predictions
        self.r_labels=r_labels
        self.r_probabilities=self.raveled_predictions

        _,_auc=tf.metrics.auc(labels=r_labels, predictions=raveled_predictions)
        self._auc=MeanAccumulator(_auc)
        self.dice=metrics.tf_dice_score(self.Yt, hard_predictions)
        # self.f1 = metrics.f1_score(labels=self.labels, soft_predictions=self.probabilities,threshold=self.output_threshold)
        self.indiv_f1 = metrics.indiv_f1(labels=self.labels, soft_predictions=self.probabilities, threshold=self.output_threshold)
        self.f1 = tf.reduce_mean(self.indiv_f1)
        tf.identity(self.f1, name="tm_perf")
        # self.classification_accuracy = tf.reduce_mean(tf.cast(tf.math.equal(tf.cast(self.Yt, tf.int32), tf.cast(YhatT_hard_pred, tf.int32)), tf.float32))
        self.classification_accuracy = tf.reduce_mean(metrics.indiv_acc(labels=self.labels, soft_predictions=self.probabilities, threshold=self.output_threshold))
        ## Both
        # self.disc_loss = tf.reduce_mean(self.disc_losses)
        self.add_summary(self.disc_acc, "1:discriminator accuracy")
        self.add_summary(self.loss, "2_loss")
        self.add_summary(self.dice, "4_dice_classification")
        self.add_summary(self.f1, "5_f1_score")
        self.add_summary(self.classification_accuracy, "6_classification_accuracy")
        self.performance_measure=self.f1

        std_vals={}
        self._current_f1=0
        std_vals["f1"]=lambda tr,_: tr._current_f1
        print_keys=list(std_vals.keys())
        self.best_value=TM_Bestie(self, tf_op=self.performance_measure, delay_steps=50, comparer=">", param_state="best_perf", name="best_f1", print_keys=print_keys)
        self.best_loss=TM_Bestie(self, tf_op=self.loss, delay_steps=50, comparer="<=", param_state="best_loss", name="best_loss", vals=std_vals, print_keys=print_keys)
        # self.best_wdiv_loss=Bestie(self, comparer="<=", param_state="best_wdiv_loss", name="best_wdiv_loss", vals=std_vals, print_keys=print_keys)
        # self._besties=[self.best_value,self.best_loss,self.best_wdiv_loss]
        self._besties=[self.best_value,self.best_loss]
        self.besties=self._besties
        self.param_state_names = [b.param_state for b in self._besties if b.param_state is not None]
        # Exclude variables that are in the discriminator scope
        # self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "^((?!discriminator).)*/")

        # self.optimizer = tf.train.AdamOptimizer(**self.opts.adam)
        self.optimizer = self.get_optimizer()
        loss_sum=self.loss
        if self.model_losses:
            loss_sum+=self.model_losses
        self.minimize = self.optimizer.minimize(loss_sum, name="Targetmap_optim", var_list=self.train_vars)
        self.train_op = tf.group([self.minimize, self.update_ops])
        # self.wdiv_loss=(10*self.loss)/(self.info.step+1)
