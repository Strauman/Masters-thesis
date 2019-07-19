import tensorflow as tf
from ..trainables import Trainable, TF_VAR, metrics, StopTraining, EarlyStopping
from abc import ABC, abstractmethod
from pprint import pprint as pp
from sys import exit as xit
import numpy as np
from .. import color_print
from ..trainables.trainable import MeanAccumulator
from ..helpers.trainhelp import Bestie, TF_Bestie, CB_Bestie

class window_bestie(TF_Bestie):
    # self.tr.confusion

    def __init__(self, *args, window_size,delay=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = []
        self.window_size = window_size
        self.delay=delay
        self.buffer = 2 * self.window_size
    def add_val(self,new_val=None):
        if new_val is None:
            new_val=self.run_op()
        self.array.append(new_val)

    def update(self, new_val=None):
        self.add_val(new_val)
        if len(self.array) >= self.window_size:
            if len(self.array) > 2 * self.window_size:
                buffer=self.array[-2 * self.window_size:]
                # del self.array
                self.array = buffer
            latest_confusions = self.array[-self.window_size:]
            cmean = np.mean(latest_confusions)
            if self.tr.info.step > self.delay:
                self._try_update(cmean)
        return self.summary()

    def summary(self, verbose=None):
        out_str = ""
        out_str += self.summary_head
        out_str += "\n"
        if len(self.array) >= self.window_size:
            out_str += super().summary(verbose=verbose)
        else:
            out_str += f"Waiting. ({len(self.array)} >= {self.window_size})"
        return out_str


class Discriminator(Trainable):
    #pylint: disable=W0201
    disc_labels = TF_VAR()
    Dhat = TF_VAR()
    acc_vals = []
    best_acc_var = None  # type: Bestie
    param_state_names = []

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.set_besties()

    def set_besties(self):
        std_vals={}
        print_keys=[]
        try:
            self.tm_perf= tm_perf = tf.get_default_graph().get_tensor_by_name("tm_perf:0")
            std_vals["target_perf"]= lambda _, __, self=self: self.get_epoch_value_val(tm_perf, max_summary_steps=self.opts.max_summary_steps)
            print_keys.append("target_perf")
        except KeyError as e:
            print("Couldn't get target f1")
        print_keys+=["step"]
        # self.best_acc_var=Bestie(trainable=self, comparer="<=", param_state="d_accuracy_variance", name="accuracy_variance", vals=std_vals, print_keys=print_keys)
        self.confusion_bestie=window_bestie(trainable=self, tf_op=self.abs_confusion, window_size=self.opts.acc_var_window, delay=2000, comparer="<=", param_state="abs_confusion", name="abs_confusion", vals=std_vals, print_keys=print_keys)
        # self.acc_var_bestie=CB_Bestie(trainable=self, callback=lambda bst: np.var(bst.trainable.confusion_bestie.array))
        self.besties = [
            self.confusion_bestie,
            # self.acc_var_bestie
            # window_bestie(trainable=self, tf_op=self.sq_confusion, window_size=self.opts.acc_var_window, comparer="<=", param_state=None, name="sq_confusion", vals=std_vals, print_keys=print_keys)
        ]
        def get_confusion(name):
            def conf_wrapper(self):
                f1=self.confusion_bestie.state_vals[name]
                if isinstance(f1, str):
                    return None
                else:
                    return f1
            return conf_wrapper

        def get_bestie_value(bestie):
            def val_wrapper(self):
                try:
                    v=bestie.value
                except:
                    v=0
                return v
            return val_wrapper
        if hasattr(self, "tm_perf"):
            self.add_value_summary("f1@Confusion score", get_confusion("target_perf"))
        self.add_value_summary("Best confusion score", get_bestie_value(self.confusion_bestie))
        # self.add_value_summary("Acc_variance", get_bestie_value(self.acc_var_bestie))
        # self.best_accuracy=Bestie(trainable=self, comparer="<=", param_state="d_accuracy", name="d_accuracy")
        self.param_state_names = [b.param_state for b in self.besties if b.param_state is not None]

    def sess_enter(self, sess):
        super().sess_enter(sess)
        self.DS_s.init_handles(self._sess)
        self.DS_t.init_handles(self._sess)

    def set_options(self, settings):
        default = dict(
            adam={},
            # scope_re="^((?!source_map|target_map|classifier).)*/"
            scope_re="discriminator",
            acc_var_window=50,
            acc_var_mean_thr=-1
        )
        default.update(settings)
        return default

    def build_graph(self):
        # graph = tf.get_default_graph()
        self.lbls = tf.cast(self.disc_labels, tf.int32)
        self.preds = tf.cast(tf.argmax(self.Dhat, axis=1), tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.math.equal(self.lbls, tf.cast(self.preds, tf.int32)), tf.float32))
        self.acc = acc
        self.performance_measure = self.acc
        # print(self.lbls.shape)
        # print(self.Dhat.shape)
        # xit()
        self.disc_loss = tf.losses.sparse_softmax_cross_entropy(
            labels=self.lbls,
            logits=self.Dhat
        )
        self.loss = self.disc_loss
        # self.loss = tf.reduce_mean(self.loss, axis=0)
        self.optimizer = self.get_optimizer()
        # pp(self.train_vars)
        # print("Model losses:")
        # pp(self.model_losses)
        # xit()
        loss_sum = self.loss
        if self.model_losses:
            loss_sum += self.model_losses
        self.minimize = self.optimizer.minimize(loss_sum, name="DiscMini", var_list=self.train_vars)
        self.abs_confusion = metrics.confusion_score(labels=self.lbls, predictions=self.preds)
        # pp(self.train_vars)
        # pp(self.model_losses)
        # pp(self.update_ops)
        # xit()
        self.add_summary(self.disc_loss, "2:loss")
        self.add_summary(self.acc, "1:accuracy")
        self.add_summary(self.abs_confusion, "CurrentConfusion")
        self.train_op = tf.group([self.minimize, self.update_ops])
        # self.cmat=tf.math.confusion_matrix(
        # self.lbls,
        # self.preds,
        # num_classes=2
        # )
        self.cmat = tf.no_op()
        # tp, tn, fp, fn = metrics.confusion_matrix(labels=self.lbls, predictions=self.preds)
        # self.fp_rate = tf.where(tf.greater(fp + tn,0), fp / (fp + tn),0)
        # self.tp_rate = tf.where(tf.greater(tp + fn,0), tp / (tp + fn),0)
        # self.sq_confusion = (self.fp_rate - (1 / 2))**2 + (self.tp_rate - (1 / 2))**2


        # self.sq_confusion = tf.abs(self.fp_rate - (1 / 2)) + tf.abs(self.tp_rate - (1 / 2))
        # self.tp_rate=tf.metrics.recall(labels=self.lbls, predictions=self.preds)
        # self.fp_rate=1-tf.metrics.specificity_at_sensitivity(labels=self.lbls, predictions=self.preds, self.tp_rate)

    def cb_epoch(self):
        self.step_epoch()
        self.run_summary_tr(do_print=False)
        self.cb_printer(do_print=False)
        self.summary_flush_loss()
        self.init_its_tr()
        self.update_besties()
        return True
    # def cb_iter(self):
    #     # train_op = tf.group([self.train_op, self._update_ops])
    #     _, loss_, tr_summary_ = self._sess.run([self.train_op, self.loss, self.summary_op_tr], self.feeds_tr())
    #     self.loss_list.append(loss_)
    #     self.stopping_loss_list.append(loss_)
    #     if self.local_init_loss is None:
    #         self.local_init_loss = loss_
    # def cb_iter(self):
    #     _, tr_summary_ = self._sess.run([self.train_op, self.summary_op_tr], self.feeds_tr())
    #     self.summary_writer_tr.add_summary(tr_summary_, info.step)
    #
    # def cb_iter(self,*args,**kwargs):
        # super(Discriminator,self).cb_iter(*args,**kwargs)
        # self.acc_vals.append(self.get_epoch_value_val(self.acc,max_summary_steps=self.opts.max_summary_steps))

    def cb_init(self):
        color_print("Training discriminator", style="notice")
        self.init_its_tr()
        # print("Resetting discriminator optimizer")
        # self._sess.run(tf.variables_initializer(self.optimizer.variables()))

    def check_early_stopping(self, val_perf_):
        super().check_early_stopping()
        if val_perf_ > self.opts.stop_val_perf and self.info.step > self.opts.min_steps and self.info.local_step > self.opts.min_local_steps:
            self.return_val = val_perf_
            raise EarlyStopping("Reached satisfactory performance measure {}".format(val_perf_), self)


    def cb_printer(self, do_print=True, check_stopping=True, write_out=True):
        info = self.info
        self.init_its_val()
        # self.run_summary_tr(self.acc)
        acc = self.acc
        # u_pred,_, c_pred=tf.unique_with_counts(self.preds)
        # acc=tf.Print(acc,[u_pred,c_pred], message="Predictions")
        # val_acc_,cmat = self.run_summary_val(acc,self.cmat)
        # self.run_summary_tr(do_print=do_print)
        # self.acc_vals.append(self.get_epoch_value_val(self.acc,max_summary_steps=self.opts.max_summary_steps))
        val_acc_, val_perf_ = self.run_summary_val(acc, self.performance_measure, do_print=do_print, write_out=write_out)
        out_str = f"disc, epoch: {info.epoch} ({info.epoch}), step: {info.step}, avg_delta_loss:{np.mean(np.diff(self.loss_list))}\n"
        out_str += self.update_besties()

        # if len(self.acc_vals) > 2:
        #     latest_accs=self.acc_vals[-self.opts.acc_var_window:]
        #     if len(self.acc_vals) > 2*self.opts.acc_var_window:
        #         self.acc_vals=self.acc_vals[-2*self.opts.acc_var_window:]
        #     acc_var=np.var(latest_accs)
        #     acc_mean=np.mean(latest_accs)
        #     out_str+=str(self.best_acc_var)
        #     out_str+=f"Curr acc_var: {acc_var}\n"

        # if len(self.acc_vals) >= self.opts.acc_var_window:
        #     if (self.opts.acc_var_mean_thr<=0 or np.abs(0.5-acc_mean)<=self.opts.acc_var_mean_thr):
        #         self.best_acc_var.try_update(acc_var)
        #         out_str+=color_print("{self.best_acc_var.name}", style="success", as_str=True)
        # else:
        #     print(len(self.acc_vals),"<" ,self.opts.acc_var_window)

        if do_print:
            print(out_str)
        # print("Confusion matrix:")
        # pp(cmat)
        if check_stopping:
            self.check_early_stopping(val_perf_)
        return out_str
