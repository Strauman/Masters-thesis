from ..trainables._targetmap import _Targetmap, TM_Bestie
import tensorflow as tf
from .. import metrics
from sys import exit as xit
import time
import pout
class Targetmap_LBL(_Targetmap): #pylint: disable=W0212
    def build_graph(self):
        #pylint: disable=W0201
        print("Building label graph")
        # Setup discriminator training
        self.tm_disc_labels=tf.cast(self.tm_disc_labels,tf.int32)
        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.tm_disc_labels,
            logits=self.Dhat_t
        )
        self._current_perf=0
        self.metric_names.avg_loss="discriminator_avg_loss"
        # self.disc_acc = tf.reduce_mean(tf.cast(tf.math.equal(self.tm_disc_labels, tf.argmax(tf.cast(self.Dhat_t, tf.int64), axis=1)), tf.float32))
        self.preds = tf.cast(tf.argmax(self.Dhat_t, axis=1),tf.int32)
        self.disc_acc = tf.reduce_mean(tf.cast(tf.math.equal(self.tm_disc_labels, self.preds), tf.float32))
        # self.disc_acc
        # Get the ground truth labels for performance measure
        labels=tf.cast(self.Yt, tf.int32)
        pred=tf.argmax(self.YhatT,axis=1)
        self.indiv_acc=tf.cast(tf.math.equal(labels, tf.cast(pred, tf.int32)), tf.float32)
        acc = tf.reduce_mean(self.indiv_acc)
        self.classification_accuracy=self.performance_measure=acc
        tf.identity(self.performance_measure, name="tm_perf")
        # self.f1=acc
        # self.disc_loss = tf.reduce_mean(self.disc_losses)
        self.add_summary(self.disc_acc, "1_discriminator accuracy")
        self.add_summary(self.loss, "2_loss")
        self.add_summary(acc, "6_classification_accuracy")
        # Exclude variables that are in the discriminator scope
        # self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "^((?!discriminator).)*/")

        # self.optimizer = tf.train.AdamOptimizer(**self.opts.adam)
        self.optimizer = self.get_optimizer()
        self.minimize = self.optimizer.minimize(self.loss+self.model_losses, name="Targetmap_optim", var_list=self.train_vars)
        self.train_op = tf.group([self.minimize, self.update_ops])
        std_vals={}
        std_vals["perf"]=lambda tr,_: tr._current_perf
        print_keys=list(std_vals.keys())
        self.best_value=TM_Bestie(self, tf_op=self.performance_measure, delay_steps=10, comparer=">", param_state="best_perf", name="best_acc", print_keys=None)
        self.best_loss=TM_Bestie(self, tf_op=self.loss, delay_steps=10, comparer="<=", param_state="best_loss", name="best_loss", vals=std_vals, print_keys=print_keys)
        self.besties=[self.best_value,self.best_loss]
        self.param_state_names = [b.param_state for b in self.besties if b.param_state is not None]
    def cb_printer(self, do_print=True, check_stopping=True, write_out=True):
        # print("HELLO?")
        info=self.info
        ret_str=""
        # pout.p("cb_printer")
        # if do_print:
        ret_str=super().cb_printer(do_print=do_print, check_stopping=check_stopping, write_out=write_out)
        # else:
            # self.update_besties()
        # pout.p()
        return ret_str
                # pout.p()
            # if val_loss_ < self.best_value:
            # if self.info.step>100:
                # self.best_loss.try_update(val_loss_)
                # self.best_wdiv_loss.try_update()
            # wloss=val_loss_/self.info.step
            # if self.best_weighted_loss and self.info.step>100:
                # self.save_param_state("best_weighted_loss")
                # self.best_weighted_loss=wloss
            # self.best_value.try_update(val_performance_)
            # best_perf=self.update_best_value(val_performance_,val_performance_,do_print=do_print)
