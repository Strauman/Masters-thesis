import tensorflow as tf
from . import Trainable, TF_VAR, metrics, StopTraining, EarlyStopping
from abc import ABC, abstractmethod
from sys import exit as xit
from pprint import pprint as pp
from ..helpers.ansi_print import write_top_right
import numpy as np
from .. import metrics, color_print
from ..helpers.trainhelp import get_graph_params, restore_graph_params, Bestie, TF_Bestie as _TF_Bestie
from ..helpers.ansi_print import print_notifications
# class BestieScore(Bestie):
#     """ """
#     def __init__(self):
#         super(BestieScore, self).__init__(*args,**kwargs)
#     def update(self)
class TM_Bestie(_TF_Bestie):
    def __init__(self, *args, delay_steps=None, **kwargs):
          super(TM_Bestie,self).__init__(*args,**kwargs)
          self.delay_steps=delay_steps
    @property
    def extra_conditions(self):
        cond1=(self.delay_steps is None or self.tr.info.step > self.delay_steps)
        return cond1

    def update(self, *args, **kwargs):
        if self.extra_conditions:
            return super(TM_Bestie, self).update(*args, **kwargs)
        return self.summary()

    def summary(self, *args, **kwargs):
        if self.extra_conditions:
            return super(TM_Bestie, self).summary(*args, **kwargs)
        out_str="\n"+self.summary_head
        out_str+=f"Waiting for {self.delay_steps} (< current: {self.tr.info.step})"
        return out_str


class _Targetmap(Trainable):
    #pylint: disable=W0201
    tm_disc_labels = TF_VAR()
    Dhat_t = TF_VAR()
    YhatT = TF_VAR()
    Yt = TF_VAR()
    inputs= TF_VAR("Xt")
    predictions=TF_VAR("YhatT")
    param_state_names=[]
    # Zt=TF_VAR()
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)

    # def setup(self, *args, **kwargs):
        # super(_Targetmap, self).setup(*args, **kwargs)

    def sess_enter(self, sess):
        super().sess_enter(sess)
        self.handle = self.DS_s.init_handles(self._sess)
        self.handle = self.DS_t.init_handles(self._sess)
        self.save_param_state("initial")

    def set_options(self, settings):
        default = dict(
            lr=0.001,
            adam={},
            scope_re="target_map",
            has_segment_labels=True,
            max_epochs=-1
        )
        default.update(settings)
        return default

    def cb_epoch(self):
        self.step_epoch()
        self.run_summary_tr(do_print=False)
        self.cb_printer(do_print=False)
        self.summary_flush_loss()
        self.update_besties()
        self.init_its_tr()
        self.init_its_val()
        if self.opts.max_epochs > 0 and self.info.epoch > self.opts.max_epochs:
            raise StopTraining(f"Targetmap reached maximum number of epochs: {self.opts.max_epochs}")
        return True

    def cb_init(self):
        color_print("Training target map", style="notice")
        self.init_its_tr()
        # print("Resetting target map optimizer")
        # self._sess.run(tf.variables_initializer(self.optimizer.variables()))
    def check_early_stopping(self, val_perfocmance_):
        super().check_early_stopping()
        if val_perfocmance_ > self.opts.stop_val_perf and self.info.step > self.opts.min_steps and self.info.local_step > self.opts.min_local_steps:
            self.return_val = val_perfocmance_
            raise EarlyStopping("Reached satisfactory performance measure {}".format(val_perfocmance_), self)


    def cb_printer(self, do_print=True, check_stopping=True, write_out=True):
        info=self.info
        # print_notifications()
        # self.init_its_val()
        # self.run_summary_tr(do_print=do_print)
        val_acc_,val_performance_,val_loss_,summary_str=self.run_summary_val(self.disc_acc,self.performance_measure,self.loss, do_print=do_print, return_str=True, write_out=write_out)
        # if val_loss_ < self.best_value:
        # if self.info.step>100:
            # self.best_loss.try_update(val_loss_)
            # self.best_wdiv_loss.try_update()
        # wloss=val_loss_/self.info.step
        # if self.best_weighted_loss and self.info.step>100:
            # self.save_param_state("best_weighted_loss")
            # self.best_weighted_loss=wloss
        # self.best_value.try_update(val_performance_)
        if hasattr(self, "f1"):
            self._current_f1=self.get_epoch_value_val(self.f1, max_summary_steps=self.opts.max_summary_steps)
        if hasattr(self, "_current_perf"):
            self._current_perf=self.get_epoch_value_val(self.performance_measure, max_summary_steps=self.opts.max_summary_steps)
        # best_perf=self.update_best_value(val_performance_,val_performance_,do_print=do_print)
        out_str=f"tm, epoch: {self.info.epoch} ({self.info.epoch}), step: {self.info.step}\n"
        out_str+=self.update_besties()
        if do_print:
            print(out_str)
        if check_stopping:
            self.check_early_stopping(val_performance_)
        if hasattr(self, "best_value"):
            out_str+=str(self.best_value)+"\n"
        out_str+=summary_str+"\n"
        return out_str
