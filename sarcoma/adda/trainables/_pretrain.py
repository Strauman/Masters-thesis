import tensorflow as tf
from . import Trainable, TF_VAR, metrics, StopTraining
from abc import ABC, abstractmethod
from .. import color_print
from .. import util
from ..helpers.ansi_print import write_top_right
from util import Dottify
import numpy as np
from pprint import pprint as pp
from ..helpers.trainhelp import get_graph_params, restore_graph_params
class _Pretrain(Trainable):
    #pylint: disable=W0201
    Ys = TF_VAR()
    Yhat = TF_VAR()
    global_step = TF_VAR()
    Zs = TF_VAR()
    inputs = TF_VAR("Xs")
    best_value=float("inf")
    best_value_op = property(lambda self: self.loss)
    best_value_comp="<"


    def sess_enter(self, sess):
        super().sess_enter(sess)
        self.DS_s.init_handles(self._sess)
        self.graph_params = get_graph_params()
        # self.DS_t.init_handles(self._sess)

    def set_options(self, settings):
        default = dict(
            scope_re="^(classifier|source_map)"
        )
        default.update(settings)
        return default

    def cb_epoch(self):
        self.step_epoch()
        self.run_summary_tr(do_print=False)
        self.cb_printer(do_print=False)
        self.summary_flush_loss()
        self.init_its_tr()
        return True

    # def cb_iter(self):
    #     _, tr_summary_ = self._sess.run([self.train_op, self.summary_op_tr], self.feeds_tr())
    #     self.summary_writer_tr.add_summary(tr_summary_, info.step)

    def cb_init(self):
        color_print("Training pretrainer", style="notice")
        self.DS_s.it_init_tr(self._sess)
        self.DS_s.it_init_val(self._sess)

    def check_early_stopping(self,val_perf_): # pylint: disable=W0221
        super().check_early_stopping()
        if val_perf_ > self.opts.stop_val_perf and self.info.step > self.opts.min_steps and self.info.local_step > self.opts.min_local_steps:
            self.return_val = val_perf_
            self.validate_val()
            raise StopTraining("Reached satisfactory performance measure {}".format(val_perf_), self)



    def cb_printer(self, do_print=True,check_stopping=True):
        val_perf_,summary_str=self.run_summary_val(self.performance_measure,do_print=do_print, return_str=True)
        # best_perf=self.update_best_value(do_print=do_print)
        self.update_besties()
        if self.print_ops and do_print:
            for op in self.print_ops:
                print(self._sess.run(op))
        out_str=f"pre, epoch: {self.info.epoch} ({self.info.epoch}), step: {self.info.step}\n"
        out_str += self.update_besties()
        if do_print:
            print(out_str)
        if check_stopping:
            self.check_early_stopping(val_perf_)
        return out_str+summary_str
