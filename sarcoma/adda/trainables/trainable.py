from ..helpers import trainhelp as hlp
from .. import color_print
from ..helpers import util
import tensorflow as tf
from abc import ABC, abstractmethod
from functools import singledispatch
import inspect
from sys import exit as xit
import sys
import os
from pprint import pprint as pp
from .. import trainvalfunc
from trainvalfunc import TrainValFuncs as TVC, trvalfunc as tvf, trvalprop as tvp, splits, split_names
import numpy as np
from ..helpers.trainhelp import FinishedEpoch, StopTraining, EarlyStopping
from ..helpers.trainhelp import ensure_list
from ..helpers.trainhelp import BatchPersistent, MeanAccumulator, splitlist_indices
import time
from .. import metrics
from ..helpers.trainhelp import get_graph_params, restore_graph_params, optimal_auc_cutoff, _DummyClass, Bestie, TF_Bestie
import pout

class TF_VAR(object):
    """Just for signifying that a class variable actualy should be a tensorflow variable"""

    def __init__(self, name=None):
        self.name = name

# class Trainable_Base(object):
#     """ Base class for trainable things """
#
#     def __init__(self, arg):
#         super(Trainable_Base, self).__init__()
#         self.arg = arg


# class SummaryHandler(object):
#     """docstring for SummaryHandler."""
class ImplementationError(Exception):
    pass


# class DictAttatch(dict):
#

@TVC
class Trainable(object):
    """docstring for Trainable."""
    class GraphKeys(object):
        INIT_ONCE = "INIT_ONCE"
        INTRINSIC = "TRAINABLE_INTRINSIC"

    @staticmethod
    def variables_initializer(*collections):
        vars = []
        for c in collections:
            vars += tf.get_collection(c)

        return tf.variables_initializer(vars)

    @staticmethod
    def initialize_intrinsic_variables():
        return tf.variables_initializer(Trainable.GraphKeys.INTRINSIC)
    is_training = TF_VAR()
    random_test = TF_VAR()
    graph_params = {}


    def best_f1(self, concise=False, cheat=False):
        """
        returns (VALUES_FOR_TRAINING), (VALUES_FOR_VAL)
        Where VALUES_FOR_ is when the thresholds for both splits have been tried on TRAINING or VAL.
        VALUES_FOR_TRAINING: validation_f1, training_f1, threshold
        VALUES_FOR_VAL: validation_f1, threshold
        """
        if not hasattr(self, "f1"):
            return -1
        labels = self.labels
        probs = self.probabilities
        # Split using training threshold
        best_tr_f1, best_tr_thr = metrics.best_f1_for_thresholds(labels=labels, predictions=probs, num_thresholds=200, feed=self.new_feeds_tr())
        tf_f1 = metrics.f1_score(labels=self.labels, soft_predictions=self.probabilities, threshold=best_tr_thr)
        best_f1_val = self.get_epoch_value_val(MeanAccumulator(tf_f1), max_summary_steps=self.opts.max_summary_steps)
        # best_f1_val=self._sess.run(tf_f1, self.feeds_val())
        # Validation
        best_at_val_f1, best_val_th = metrics.best_f1_for_thresholds(labels=labels, predictions=probs, num_thresholds=200, feed=self.new_feeds_val())
        if concise:
            if cheat:
                return best_at_val_f1, best_val_th
            else:
                return best_f1_val, best_tr_thr
        return (best_f1_val, best_tr_f1, best_tr_thr), (best_at_val_f1, best_val_th)
    # @graph_params.setter
    # def graph_params(self, value):
    def get_f1_for_thresh(self, threshold=0.5):
        if not hasattr(self, "f1"):
            return -1
        f1 = metrics.f1_score(labels=self.labels, soft_predictions=self.probabilities, threshold=threshold)
        return self.get_epoch_value_val(MeanAccumulator(f1), max_summary_steps=self.opts.max_summary_steps)

    # def get_optimal_f1(self, labels, predictions):
    #     self.init_it_val()
    #     yshape=tf.shape(labels)
    #     raveled_labels=tf.reshape(labels,[-1,yshape[1]*yshape[2]])
    #     raveled_predictions=tf.reshape(predictions,[-1,yshape[1]*yshape[2]])
    #     optimal_cutoff=optimal_auc_cutoff(labels=raveled_labels, predictions=raveled_predictions, num_thresholds=200, feed=self.feeds_val())
    #     print("OPTIMAL thresh:", optimal_cutoff)
    #     self.init_it_val()
    #     f1=metrics.f1_score(labels=labels, soft_predictions=predictions, threshold=optimal_cutoff)
    #     optimal_f1=self._sess.run(f1, self.feeds_val())
    #     return optimal_f1

    def __init__(self, DS_s=None, DS_t=None, settings=None, name=None, tensorboard_path=None, saver=None, writeable=True, *setup_args, **setup_kwargs):
        self.tensorboard_path = tensorboard_path
        name = name or self.__class__.__name__
        self.metric_names = util.Dottify(
            avg_loss="3_average_loss"
        )
        self.writeable = writeable
        self._optimizer = None
        self.saver = saver
        self.name = name
        self._tf_step = tf.Variable(0, name=f"{self.name}_step", trainable=False, collections=[Trainable.GraphKeys.INTRINSIC])
        self._tf_epoch = tf.Variable(0, name=f"{self.name}_epoch", trainable=False, collections=[Trainable.GraphKeys.INTRINSIC])
        # tf.add_to_collection(Trainable.GraphKeys.INIT_ONCE, self._tf_step)
        # tf.add_to_collection(Trainable.GraphKeys.INIT_ONCE, self._tf_epoch)
        self.print_summaries_tr = []
        self.print_summaries_val = []
        self.print_ops = []
        self.loss_list = []
        self.prev_print = 0
        self.stopping_loss_list = []
        self.DS_s = DS_s
        self.DS_t = DS_t
        self.loss = None
        self._batch_persistent_ops = {}
        # Update operations
        self._update_ops = None
        self._model_losses = None
        self._train_vars = None
        self._did_access = util.Dottify(model_losses=False, update_ops=False, train_vars=False)
        self._auto_init_feed = False
        self._aux_feeds={}
        self.value_summaries={}
        if settings is None:
            settings = {}
        # if tf_varnames is None:
            # tf_varnames = []

        if isinstance(settings, util.Dottify):
            settings = settings.__dict__
        elif not isinstance(settings, dict):
            raise TypeError("Argument `settings` has to be of type dict or Dottify")
        user_opts = settings
        early_stopping = dict(loss_difference_threshold=None, loss_difference_memory=-1, percentage_total_decrease=None)
        if "early_stopping" in settings.keys():
            es = settings["early_stopping"]
            early_stopping.update(es)
        dotted_es = util.Dottify(early_stopping)
        settings["early_stopping"] = dotted_es
        color_print("Early stopping settings:")
        pp(settings["early_stopping"])
        meta_default_opts = dict(
            max_summary_steps=None,
            printerval=100,
            stop_val_dice=np.inf,
            stop_val_acc=np.inf,
            max_epochs=100,
            scope_re="*",
            min_local_steps=0,
            min_steps=0,
            reset_optimizers=False
        )
        default_opts = self.set_options(user_opts)
        meta_default_opts.update(default_opts)
        self.opts = util.Dottify(meta_default_opts)
        self.return_val = True
        self.graph = tf.get_default_graph()
        self.varnames = []
        self.info = util.Dottify(step=1, epoch=1, local_epoch=1, local_step=0)
        self.graph_built = False
        self.__sess = None
        self._train_inited = False
        self.has_summary = False
        self.local_init_loss = None
        self.setup(*setup_args, **setup_kwargs)
        self.besties=[] if not hasattr(self, "besties") else self.besties

    def update_besties(self):
        out_str=""
        if not self.besties:
            return ""
        for b in self.besties:
            out_str+=b.update()+"\n"
        return out_str

    def get_besties_strings(self, verbose=None):
        out_str="\n"
        if not self.besties:
            return ""
        for b in self.besties:
            out_str+=b.summary(verbose)+"\n"
        return out_str

    def get_optimizer(self):
        if self._optimizer is not None:
            return self._optimizer
        try:
            self._optimizer = self.opts.optimizer(self)
            print(f"Optimizer for {self.name}:{self._optimizer.get_name}")
            return self._optimizer
        except AttributeError:
            raise AttributeError("Pretrain needs optimizer in settings!")

    def set_options(self, settings):
        return settings

    def ignore_use(self, elem):
        if elem is self._update_ops and self._update_ops:
            color_print("Ignoring unused update ops", style="danger")
        elif elem is self._model_losses and self._model_losses:
            color_print("Ignoring unused losses", style="danger")
        elif elem is self._train_vars and self._train_vars:
            color_print("Not scoping trainable variables!", style="danger")

    def initialize_tf_variables(self):
        # First find the declared variables
        self.declared_tf_vars = []
        # pp(inspect.getmembers(self.__class__))
        # xit()
        for attr, attr_inst in inspect.getmembers(self.__class__):
            if isinstance(attr_inst, TF_VAR):
                class_name = attr
                tensor_name = attr
                if attr_inst.name is not None:
                    tensor_name = attr_inst.name
                self.declared_tf_vars.append((class_name, tensor_name))

        self.varnames = self.declared_tf_vars
        for class_name, tensor_name in self.varnames:
            # if hasattr(self, class_name) and class_name not in self.declared_tf_vars:
                # raise AttributeError(f"Class {self.__class__.__name__} (name {self.name}) already have a {varname} attribute")
            tf_var = self.graph.get_tensor_by_name(tensor_name + ":0")
            setattr(self, class_name, tf_var)

    def initialize_training_bools(self):
        if self.DS_s is not None:
            self.DS_s.set_training_bool(self.is_training)
        if self.DS_t is not None:
            self.DS_t.set_training_bool(self.is_training)

    def init_collections(self):
        self._update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, self.opts.scope_re)
        print("UpdateOps")
        pp(self._update_ops)
        self._model_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.opts.scope_re)
        self._train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.opts.scope_re)
        self._did_access.model_losses = False
        self._did_access.update_ops = False
        self._did_access.train_vars = False

    def assert_collections_used(self):
        if self._model_losses and not self._did_access.model_losses:
            raise ImplementationError("There were unused model losses. Use self.model_losses in your build_graph!")
        if self._update_ops and not self._did_access.update_ops:
            raise ImplementationError("There were unused update ops. Use self.update_ops in your build_graph!")

    @property
    def model_losses(self):
        self._did_access.model_losses = True
        return self._model_losses

    @property
    def update_ops(self):
        self._did_access.update_ops = True
        return self._update_ops

    @property
    def train_vars(self):
        self._did_access.train_vars = True
        return self._train_vars

    def setup_saving(self):
        if self.saver is None:
            return
        if self.saver._inited:
            color_print(f"{self.name} Saver already inited. Cannot add step and epoch values :/", style="warning")
            # return
        self.saver.add_vars(self._tf_step, self._tf_epoch)
    # def variables_initializer(self):
        # return pretrainer._tf_step, pretrainer._tf_epoch

    def setup(self, *args, **buildkwargs):
        self.initialize_tf_variables()
        self.initialize_training_bools()
        self.setup_summary()
        if not self.graph_built:
            self.init_collections()
            self.build_graph(**buildkwargs)
            self.setup_saving()
            self.assert_collections_used()
            if self.loss is None:
                color_print(f"{self.name}: build_graph did not define self.loss, and wil not be able to get it for automatic validation", style="warning")

            self.merge_summaries()
            self.graph_built = True

    # def add_update_ops(self,*ops):
        # for op in ops:
        # self._update_ops.append(op[])
        # return self._update_ops

    def post_restore(self):
        self.info.step, self.info.epoch = tf.get_default_session().run([self._tf_step, self._tf_epoch])
        print(f"RESTORED AT step: {self.info.step} epoch: {self.info.epoch}")

    def sess_enter(self, sess):
        self.__sess = sess
        self.summary_writer_tr.add_graph(sess.graph)
        self.info.step, self.info.epoch = sess.run([self._tf_step, self._tf_epoch])
        color_print(f"{self.name} is entering session: Starting at step: {self.info.step},epoch: {self.info.epoch}", style="success")
        # self.summary_flush_loss()

    @tvf
    def _init_data_(self, split):
        if self.DS_s:
            self.DS_s.send_(split, batch=1000)
            _ = self.DS_s.iterator_(split)
            self.DS_s.init_iterator_handle_(split, self._sess)
        if self.DS_t:
            self.DS_t.send_(split, batch=1000)
            _ = self.DS_t.iterator_(split)
            self.DS_t.init_iterator_handle_(split, self._sess)

    def init_testdata(self):
        self._init_data_(splits.tst)

    # def __getattr__(self, attr):
    #     # DataSet().dataset_tr -> DataSet().dataset_(str2split['tr'])
    #     x = re.match(r'(.*?_)(tr|val|tst)$', attr)
    #     if x:
    #         split_name = x.group(2)
    #         split_num = str2split[split_name]
    #         if not hasattr(self, x.group(1)):
    #             super().__getattribute__(attr)
    #         # self.is_routed=True
    #         return getattr(self, x.group(1))(split_num)
    #     super().__getattribute__(attr)
    def reset_optimizer(self):
        self._sess.run([tf.variables_initializer(self._optimizer.variables())])

    def _train(self):
        did_print_err = False
        # with tf.get_default_session() as sess:
        self.info.local_step = 1
        self.info.local_epoch = 1
        self.local_init_loss = None
        if self.opts.reset_optimizers:
            color_print(f"Resetting optimizer variables for {self.name}", style="warning")
            self._sess.run([tf.variables_initializer(self._optimizer.variables())])
            # pp(self._optimizer.get_name())
            # print(self._optimizer.name)
            # print("Variables")
            # pp(self._optimizer.variables())
            # print("Collection")
            # pp(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,self._optimizer.get_name()))
            # print(self._optimizer.get_slot())
            # print(self._optimizer.get_slot_names())
            # xit()
            # optimizer_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"scope/prefix/for/optimizer")
            # sess.run(tf.initialize_variables(optimizer_scope))
            # print("resetting optimiser vars:")
        if callable(self.cb_init):
            self.cb_init()
            print("Init done for {}".format(self.__class__.__name__))
        # sess.run(tf.local_variables_initializer())
        while True:
            if self.cb_epoch() is False:
                break
            if self.info.local_epoch >= self.opts.max_epochs:
                raise EarlyStopping(f"Reached max epochs {self.info.local_epoch} >= {self.opts.max_epochs}")
            while True:
                try:
                    # print("X")
                    # pout.p("Full iteration")
                    self.cb_iter()
                    # pout.p("Check printerval")
                    deltat = time.time() - self.prev_print
                    if deltat > self.opts.printerval:
                        # pout.p("Do printerval")
                        print(f"dt={deltat} > {self.opts.printerval}")
                        try:
                            self.cb_printer()
                        except NotImplementedError as e:
                            if not did_print_err:
                                did_print_err = True
                                color_print("Printer callback is not implemented", style="warning")
                                print(str(e))
                        finally:
                            self.prev_print = time.time()
                        # pout.p()
                    # pout.p()
                    # pout.p()
                    # self.info.step += 1
                    # self.info.local_step += 1
                except FinishedEpoch:
                    break
            self.info.step += 1
            self.info.epoch += 1
            self.info.local_epoch += 1

    def _train_init(self, sess, **buildkwargs):
        if self._train_inited:
            return
        self._train_inited = True
        if not self._sess:
            self._sess = sess

    @property
    def _sess(self):
        if self.__sess is None:
            raise AttributeError(f"Hummm... Seems like you are trying to access the session before {self.name} has been informed of it's existence. \n Did you call sess_enter(sess)?")
        return self.__sess

    def update_state(self):
        self._sess.run([self._tf_step.assign(self.info.step), self._tf_epoch.assign(self.info.epoch)])

    def train(self):
        # self._train_init(sess)
        try:
            self._train()
        except KeyboardInterrupt as e:
            print("KeyboardInterrupt")
            self.update_state()
            time.sleep(0.7)
            return self.return_val
        except StopTraining as e:
            print("Requested to stop training")
            print(str(e))
            self.update_state()
            return self.return_val

    # User implemented
    def build_graph(self):
        """ Build tf graph """
        raise NotImplementedError("build_graph not implemented")

    @tvf
    def feeds_(self, split):
        return self.get_reuseable_feed_(split, f"main:{self.name}", reset=False)
        # if self._auto_init_feed:
        #     self.init_it_(split)
        # if self.DS_s is None:
        #     # if we don't have DS_s we only return DS_t
        #     return self.DS_t.feed_(split)
        # elif self.DS_t is None:
        #     # and vica versa
        #     return self.DS_s.feed_(split)
        # # return both if they are good
        # return hlp.merge_dicts(self.DS_s.feed_(split), self.DS_t.feed_(split))
    @tvf
    def get_reuseable_feed_(self, split, feed_name, reset=True):
        feed_name+=f":{split}"
        if feed_name not in self._aux_feeds:
            if self.DS_s is None:
                # if we don't have DS_s we only return DS_t
                feed,it=self.DS_t.new_feed_(split, reuseable=True)
                init_op=it.initializer
                # it=[it]
            elif self.DS_t is None:
                # and vica versa
                feed,it=self.DS_s.new_feed_(split, reuseable=True)
                init_op=it.initializer
                # it=[it]
            else:
            # return both if they are good
                feed_s,it_s=self.DS_s.new_feed_(split, reuseable=True)
                feed_t,it_t=self.DS_t.new_feed_(split, reuseable=True)
                feed=hlp.merge_dicts(feed_s,feed_t)
                # it=[it_s,it_t]
                init_op=tf.group([it_s.initializer, it_t.initializer])
            self._aux_feeds[feed_name]=(feed, init_op)
        else:
            feed, init_op=self._aux_feeds[feed_name]
        if reset:
            tf.get_default_session().run(init_op)
        return feed


    @tvf
    def new_feeds_(self, split, feed_name=None):
        # self.init_it_(split)
        # return self.feeds_(split)
        if self.DS_s is None:
            # if we don't have DS_s we only return DS_t
            return self.DS_t.new_feed_(split)
        elif self.DS_t is None:
            # and vica versa
            return self.DS_s.new_feed_(split)
        # return both if they are good
        return hlp.merge_dicts(self.DS_s.new_feed_(split), self.DS_t.new_feed_(split))

    @tvp
    def _npz_savefile_(self, split):
        return f"/root/data/results/predictions-{self.name}-{split_names[split]}"

    def _save_predictions_(self, split, **data_dict):
        savefile = self._npz_savefile_(split)
        color_print(f"Saving to {savefile}.npz")
        return np.savez(savefile, **data_dict)

    @tvf
    def validate_(self, split, do_print=True, full_epoch=None, write_out=True, color=True, return_str=False):
        if split == splits.tst:
            if util.ask_user(f"Are you sure you want to forward pass test data for {self.name}?"):
                self._init_data_tst()
            else:
                print("NOT RUNNING TEST")
                return None, None
        if hasattr(self, "loss") and hasattr(self, "performance_measure"):
            # loss_, perf_ = self._sess.run([self.loss, self.performance_measure], self.feeds_(split))
            loss_, perf_ = self.run_summary_(split, self.loss, self.performance_measure, do_print=False, full_epoch=full_epoch, write_out=write_out)
            # self.init_it_tr()
            out_string = f"{self.name}/{split_names[split]}: loss {loss_}, perf {perf_}"
            if do_print:
                color_print(out_string, style="success")
            rets = [loss_, perf_]
            if return_str:
                rets.append(out_string)
            return rets
        else:
            print("You have to define loss and performance measure")
            return None, None

    @tvf
    def predict_(self, split, do_print=True, do_save=False):
        loss_, performance_ = self.validate_(split, do_print=do_print)
        if not hasattr(self, "inputs") or not hasattr(self, "predictions"):
            raise ImplementationError(f"No `inputs` and/or `predictions` tensor have been implemented for {self.name}")
        X, Y = self._sess.run([self.inputs, self.predictions], self.new_feeds_(split))
        if do_save:
            color_print("Predicting: saving to.", style="notice")
            self._save_predictions_(split, X=X, Y=Y)
        else:
            color_print("Predicting: not saving.", style="notice")
        return X, Y, loss_, performance_

    @tvf
    def save_predictions_(self, split, do_print=True):
        return self.predict_(split, do_print=do_print, do_save=True)

    # @tvf
    # def predict_(self,split):
    #     self.init_its_(split)
    #     if not hasattr(self, "input") or not hasattr(self, "predictions"):
    #         ImplementationError(f"No `input` or `predictions` tensor have been implemented for {self.name}")
    #     X,Y=self._sess.run([self.inputs,self.predictions], self.feeds_(split))
    #     np.savez(f"/root/data/results/predictions-{self.name}-{split_names[split]}", X=X, Y=Y)

    @tvf
    def init_its_(self, split):
        feed_name=f"main:{self.name}:{split}"
        if feed_name not in self._aux_feeds.keys():
            self.get_reuseable_feed_(split, f"main:{self.name}", reset=True)
            return
        _, init_op=self._aux_feeds[feed_name]
        tf.get_default_session().run(init_op)
        # if self.DS_s is not None:
        #     self.DS_s.init_it_(split, self._sess)
        # if self.DS_t is not None:
        #     self.DS_t.init_it_(split, self._sess)

    # Aliases
    @tvf
    def init_it_(self, split):
        return self.init_its_(split)

    def save_param_state(self, graph_params_name):
        self.update_state()
        # color_print(f"Saving parameter state {graph_params_name}", style="danger")
        # self.graph_params[graph_params_name] = get_graph_params(scope=self.opts.scope_re)
        self.graph_params[graph_params_name] = get_graph_params()
        intrinsic = get_graph_params(collection=Trainable.GraphKeys.INTRINSIC)
        besties_vars = get_graph_params(collection=Bestie.BESTIE_COLLECTION)
        # custom_gvars=[self._tf_step,self._tf_epoch]
        # custom_gvar_vals={gvar.op.name: value for gvar, value in zip(custom_gvars, tf.get_default_session().run(custom_gvars))}
        self.graph_params[graph_params_name].update(intrinsic)
        self.graph_params[graph_params_name].update(besties_vars)

    def update_best_value(self, new_value=None, performance_value=None, do_print=False, graph_params_name="best_perf"):
        raise DeprecationWarning("update_best_value is deprecated")
        run_measures = {
            'new': tf.zeros_like(self.best_value_op),
            'perf': tf.zeros_like(self.performance_measure)
        }
        if new_value is None:
            run_measures['new'] = self.best_value_op
        if performance_value is None:
            run_measures['perf'] = self.performance_measure
        # run_ops = [run_measures['perf'], run_measures['new']]
        # print(type(run_measures['perf']), type(run_measures['new']), run_measures['perf'], run_measures['new'])
        # xit()
        perf_val_out, new_value_out = self.run_summary_val(run_measures['perf'], run_measures['new'], do_print=False, write_out=False)
        new_value_ = new_value_out if new_value is None else new_value
        perf_val_ = perf_val_out if performance_value is None else performance_value
        if self._upd_conds[self.best_value_comp](new_value_):
            if callable(perf_val_):
                perf_val_ = perf_val_(self)
            self.best_value = new_value_
            self.perf_at_best = perf_val_
            self.update_state()
            self.save_param_state(graph_params_name)
            # print(self.graph_params)
        out_str = f"Perf@best: {self.perf_at_best}, value@best={self.best_value}"
        if do_print:
            print(out_str)
        return out_str

    def check_early_stopping(self):
        return True
        es = self.opts.early_stopping
        if self.info.local_step < 100:
            return True
        if es.loss_difference_threshold is not None:
            if self.stopping_loss_list:
                if es.loss_difference_memory is not None:
                    self.stopping_loss_list = self.stopping_loss_list[-es.loss_difference_memory:]
                    mem = es.loss_difference_memory
                else:
                    mem = len(self.stopping_loss_list)
                avg_losses = self.stopping_loss_list
                avg_delta = np.mean(np.diff(avg_losses))
                # print(f"avg_delta_loss: {avg_delta}")
                if np.abs(avg_delta) <= es.loss_difference_threshold:  # and avg_delta >= 0:
                    self.run_summary_val()
                    self.run_summary_tr()
                    raise EarlyStopping(f"Loss stopped changing. Avg change over {mem} iters is |{avg_delta}| <= {es.loss_difference_threshold}.")
        if es.percentage_total_decrease is not None and self.local_init_loss is not None:
            if self.stopping_loss_list[-1] / self.local_init_loss < es.percentage_total_decrease:
                raise EarlyStopping(f"Reached {es.percentage_total_decrease} fraction of initial loss. {self.stopping_loss_list[-1]}/{self.local_init_loss} < {es.percentage_total_decrease}.")

    def cb_init(self):
        raise NotImplementedError()

    def cb_epoch(self):
        raise NotImplementedError()

    @property
    def iter_op(self):
        return [self.train_op, self.loss, self.summary_op_tr]

    def cb_iter(self, *ops, reset_epoch=False):
        self.info.step += 1
        self.info.local_step += 1
        # train_op = tf.group([self.train_op, self._update_ops])
        # ops=list(ops)
        # if not ops:
        # ops=[tf.no_op()]
        run_ops = [self.train_op, self.loss]
        try:
            # pout.p("running ops")
            _, loss_ = self._sess.run(run_ops, self.feeds_tr())
            # pout.p()
            # pout.p("registering loss")
            self.register_cb_iter(loss_)
            # pout.p()
        except tf.errors.OutOfRangeError:
            if reset_epoch:
                self.cb_epoch()
            raise FinishedEpoch("No one dealt with finished epoch, so just quitting", trainable=self)

    def register_cb_iter(self, loss_):
        self.loss_list.append(loss_)
        self.stopping_loss_list.append(loss_)
        if self.local_init_loss is None:
            self.local_init_loss = loss_
        # self.summary_writer_tr.add_summary(tr_summary_, info.step)
        # raise NotImplementedError()

    def step_epoch(self):
        self.info.step += 1
        self.info.epoch += 1
        self.info.local_epoch += 1

    def cb_printer(self):
        raise NotImplementedError()

    # Summary stuff
    def summary_flush_loss(self):
        # Add it to the Tensorboard summary writer
        if len(self.loss_list) > 1:
            loss_mean = np.mean(self.loss_list)
            self.loss_summary.value[0].simple_value = loss_mean
            self.loss_list = []
            self.summary_writer_tr.add_summary(self.loss_summary, self.info.step)
        # else:
            # self.loss_summary.value[0].simple_value = float("NaN")
        # Make sure to specify a step parameter to get nice graphs over time
    def add_value_summary(self, name, cb):
        summary=tf.Summary()
        summary.value.add(tag=f"{self.name}/{name}", simple_value=0)
        self.value_summaries[name]=dict(s=summary, cb=cb)

    def setup_summary(self):
        #pylint: disable=W0201
        # Set up custom loss averaging
        self.loss_summary = tf.Summary()
        self.loss_summary.value.add(tag=f"{self.name}/{self.metric_names.avg_loss}", simple_value=0)

        self.summary_op_tr = tf.no_op()
        self.summary_op_val = tf.no_op()
        self.summary_collections = util.Dottify(tr=f"{self.name}_tr", val=f"{self.name}_val")
        self.summary_update_collections = util.Dottify(tr=f"upd_{self.name}_tr", val=f"upd_{self.name}_val")

        if self.tensorboard_path is None:
            color_print("No tensorboard path provided; no summaries...", style="warning")
            return
        # print(self._sess)
        # xit()
        self.has_summary = True
        run_suffix = ""
        if "--run-name" in sys.argv:
            run_suffix = sys.argv[sys.argv.index("--run-name") + 1]
        if "--no-tb" in sys.argv or not self.writeable:
            self.summary_writer_tr = _DummyClass()
            self.summary_writer_val = _DummyClass()
        elif "ON_SERVER" in os.environ:
            self.summary_writer_tr = tf.summary.FileWriter(os.path.join(self.tensorboard_path, "train" + run_suffix))
            self.summary_writer_val = tf.summary.FileWriter(os.path.join(self.tensorboard_path, "validation" + run_suffix))
    def stop_writing(self):
        self.summary_writer_tr = _DummyClass()
        self.summary_writer_val = _DummyClass()

    @tvf
    def get_epoch_value_(self, split, operations,max_summary_steps=None, limit_max_steps=None):
        # all ops must be of class BatchPersistent
        _ops = ensure_list(operations)
        ops = []
        for op in _ops:
            if not isinstance(op, (BatchPersistent, MeanAccumulator)):
                # raise TypeError("All ops must be of BatchPersistent type! ({op} wasn't)")
                if not op in self._batch_persistent_ops:
                    self._batch_persistent_ops[op] = MeanAccumulator(op)
                op = self._batch_persistent_ops[op]
            ops.append(op)

        # self.init_it_(split)
        # print(self._auc.shape)
        # return self._sess.run(self._auc, self.feeds_val())
        self._sess.run([op.flush_op for op in ops])
        feed = self.get_reuseable_feed_(split, feed_name="epoch_value_feeder")
        # feed=self.feeds_(split)
        if limit_max_steps is True:
            max_summary_steps=self.opts.max_summary_steps
        elif limit_max_steps is False:
            max_summary_steps=None
        self.run_epoch_update_ops([op.update_op for op in ops], feed, max_summary_steps)
        out_ops = self._sess.run([op.output_tensor for op in ops])
        if len(out_ops) == 1:
            return out_ops[0]
        return out_ops

    def run_epoch_update_ops(self, ops, feed, max_summary_steps=None):
        num_ops = len(ops)
        ops = ensure_list(ops)
        _results = [[] for op in ops]
        i=0
        while True:
            i+=1
            try:
                _results = self._sess.run(ops, feed)
                if max_summary_steps is not None and i >= max_summary_steps:
                    # print(f"Trained {i} summary batches before stop.")
                    return _results
            except tf.errors.OutOfRangeError:
                # print(f"Did {i} summary run steps")
                return _results
        # splitlist_indices

    @tvf
    def run_summary_(self, split, *other_ops, do_print=True, full_epoch=None, write_out=True, reset_iter=True, return_str=False, limit_steps=True):
        self.update_state()
        other_ops = list(other_ops)
        if not self.has_summary:
            color_print("No summary to run?", syle="warning")
            return
        # Decide between splits
        get_sumr = {
            splits.tr: (self.summary_op_tr, tf.get_collection(self.summary_update_collections.tr), self.summary_writer_tr, self.print_summaries_tr),
            splits.val: (self.summary_op_val, tf.get_collection(self.summary_update_collections.val), self.summary_writer_val, self.print_summaries_val),
            splits.tst: (self.summary_op_val, tf.get_collection(self.summary_update_collections.val), None, self.print_summaries_val)
        }
        if full_epoch is None:
            full_epoch = (split == splits.val)
        _summary_op, accumulators, writer, to_print = get_sumr[split]
        # List for iterability and such
        summary_op = [_summary_op]
        # Get the update operations
        update_ops = [accum.update_op for accum in accumulators]
        # Merge the standard ops with the given other_ops
        # Make sure other_ops also have accumulator functionality
        for i, oop in enumerate(other_ops):
            if not isinstance(oop, BatchPersistent):
                # raise RuntimeError("Trying to run a non-MeanAccumulator operation :/")
                if not oop in self._batch_persistent_ops:
                    self._batch_persistent_ops[oop] = MeanAccumulator(oop)
                other_ops[i] = self._batch_persistent_ops[oop]

        other_update_ops = [oop.update_op for oop in other_ops]
        all_update_ops = update_ops + other_update_ops
        all_flush_ops = [accum.flush_op for accum in accumulators] + [oop.flush_op for oop in other_ops]
        self._sess.run(all_flush_ops)
        # Reset iterator
        # if reset_iter or full_epoch:
            # self.init_it_(split)
        feed_name="run_summary_feeder"
        feed = self.get_reuseable_feed_(split,feed_name=feed_name)
        max_summary_steps = self.opts.max_summary_steps if limit_steps is True else None
        if full_epoch:
            self.run_epoch_update_ops(all_update_ops, feed, max_summary_steps)
        else:
            self._sess.run(all_update_ops, feed)
        # Now run and get the output and summary ops. Shouldn't be anything to feed.
        summary_ = self._sess.run(summary_op)
        if isinstance(summary_, list) and len(summary_) == 1:
            summary_ = summary_[0]
        try:
            if write_out and writer is not None:
                writer.add_summary(summary_, global_step=self.info.step)
        except BaseException:
            print(summary_)
            raise
        vsummary_values={}
        for name,summary_d in self.value_summaries.items():
            val=summary_d['cb'](self)
            if val is None:
                continue
            vsummary_values[name]=val
            summary_handle=summary_d['s']
            summary_handle.value[0].simple_value=val
            writer.add_summary(summary_handle, self.info.step)

        out_str = ""
        # Format printing operations. to_print is formatted as a ziplist: [(name,op),...]
        print_names, print_ops = list(zip(*to_print))
        print_vals = self._sess.run(print_ops)
        for i, n in enumerate(print_names):
            out_str += f"{n}: {print_vals[i]}\n"
        if do_print:
            print(out_str)
        output = self._sess.run([oop.output_tensor for oop in other_ops], self.get_reuseable_feed_(split, feed_name=feed_name))
        if return_str:
            return output + [out_str]
        if isinstance(output, list) and len(output) == 1:
            output = output[0]
        return output

        # return self._sess.run(other_ops, self.feeds_(split))

    # @tvf
    # def run_summary_(self, split, *other_ops, do_print=True):
    #     if not self.has_summary:
    #         color_print("No summary to run?",syle="warning")
    #         return
    #
    #     get_sumr = {
    #         splits.tr: (self.summary_op_tr, self.summary_writer_tr, self.print_summaries_tr),
    #         splits.val: (self.summary_op_val, self.summary_writer_val, self.print_summaries_val)
    #     }
    #     other_ops = list(other_ops)
    #     # print(other_ops)
    #     op, writer, to_print = get_sumr[split]
    #     print_names, print_ops = list(zip(*to_print))
    #     # print_names,print_ops=list(print_names),list(print_ops)
    #     ops = [op]
    #     out_ops_ = []
    #     if other_ops:
    #         ops = [op] + other_ops
    #     if do_print:
    #         ops += print_ops
    #     feed=self.feeds_(split)
    #     # Setup iterator
    #     self.init_it_(split)
    #     sess_results=[]
    #     # if not isinstance(ops, (list,tuple)):
    #         # ops=[]
    #     # pp(ops)
    #     # xit()
    #     for myop in ops:
    #         if isinstance(myop, (tuple,list)):
    #             raise TypeError(f"Wrong type of operation: expected tf op, not {type(myop)}")
    #     # if split==splits.val:
    #         # sess_results=self.summary_full_epoch(ops, feed)
    #     # else:
    #     sess_results=self._sess.run(ops,feed)
    #     # for myop in ops:
    #     #     try:
    #     #         sess_results.append(self._sess.run(myop, feed))
    #     #     except Exception:
    #     #         print(myop, feed)
    #     #         raise
    #     #         xit()
    #     summary_ = sess_results[0]
    #     print_split_idx = len(other_ops) + 1
    #     if len(other_ops) == 1:
    #         out_ops_ = sess_results[1]
    #     elif len(other_ops) > 1:
    #         out_ops_ = sess_results[1:print_split_idx]
    #
    #     for i, n in enumerate(print_names):
    #         print(f"{n}: {sess_results[print_split_idx+i]}")
    #     # color_print("Running summary", style="notice")
    #     # print("session_runs:")
    #     # pp(ops)
    #     # print("Session results:")
    #     # pp(sess_results)
    #     # xit()
    #     writer.add_summary(summary_, global_step=self.info.step)
    #     if isinstance(out_ops_, list) and len(out_ops_) == 1:
    #         out_ops_ = out_ops_[0]
    #     # print(summary_.decode("utf-8"))
    #     return out_ops_

    def merge_summaries(self):
        if not self.has_summary:
            color_print("No summaries to merge", style="notice")
            return
        self.summary_op_tr = tf.summary.merge_all(self.summary_collections.tr)
        self.summary_op_val = tf.summary.merge_all(self.summary_collections.val)

    def add_summary(self, tensor, name, incl_val=True, also_print=True):
        """ incl_val: whether or not to also add summary for validation. Else test. """
        if not self.has_summary:
            return
        nme = f"{self.name}/{name}"
        update_ops_collections = [self.summary_update_collections.tr]
        collections = [self.summary_collections.tr]
        if incl_val:
            collections.append(self.summary_collections.val)
            update_ops_collections.append(self.summary_update_collections.val)
        cs = MeanAccumulator(tensor=tensor, name=name, collections=collections)
        for c in update_ops_collections:
            tf.add_to_collection(c, cs)
        tf.summary.scalar(name=nme, tensor=cs.output_tensor, collections=cs.collections)
        if also_print:
            self.print_summaries_tr.append((f"train:{nme}", cs.output_tensor))
            if incl_val:
                self.print_summaries_val.append((f"val:{nme}", cs.output_tensor))
        # tf.summary.scalar(name + "_tr", tensor=tensor, collections=[self.summary_collections.tr])
        # tf.summary.scalar(name + "_val", tensor=tensor, collections=[self.summary_collections.val])

    def add_mean_summary(self, tensor, name, incl_val=True, also_print=True):
        #
        pass
