import tensorflow as tf
from .. import color_print
import os
from sys import exit as xit
from pprint import pprint as pp
from termios import tcflush, TCIFLUSH
import sys
import itertools
import tempfile
import time
import click
from .. import metrics
from .util import ensure_list, merge_dicts, call_if_lambda
from . import util
# from ..trainables.trainable import Trainable
from collections import OrderedDict
# https://www.tensorflow.org/api_docs/python/tf/Graph#get_operation_by_name


def get_variables_in(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def get_saver_for(scope):
    return tf.train.Saver(var_list=get_variables_in(scope))


def get_variable_by_name(name):
    # https://stackoverflow.com/questions/35678883/tensorflow-getting-variable-by-name
    graph = tf.get_default_graph()
    # return graph.get_tensor_by_name("bar:0")
    return [var for var in tf.global_variables() if var.op.name == name][0]
    # with tf.variable_scope("pretraining", reuse=tf.AUTO_REUSE):


class _SessionSaver(tf.Session):
    """docstring for SessionSaver."""

    def __init__(self, genfunc, *args, **kwargs):
        self.genfunc = genfunc
        self.gen = None
        self.string = "HELLO"
        super(_SessionSaver, self).__init__(*args, **kwargs)

    def __enter__(self, *args, **kwargs):
        super().__enter__(*args, **kwargs)
        self.gen = self.genfunc(self)
        return next(self.gen)

    def __exit__(self, exc_type, exc_value, exc_tb):
        print("SESSION SAVER EXIT:")
        # errstr=f"exc_type:{exc_type}\n exc_value:{exc_value}\n exc_tb:{exc_tb}"
        # print(errstr)
        if "--errespect" in sys.argv and exc_type is not None:  # isinstance(exc_type, BaseException):
            print("RESPECTING ERRORS!")
            color_print(f"RESPECTING ERROR!", style="danger")
            # print(errstr)
            return False
        if "--abortable" in sys.argv:
            return False
        try:
            next(self.gen)
        except StopIteration:
            pass
        except SystemExit:
            os._exit(0)
        # color_print(f"exc_type:{exc_type}, exc_value:{exc_value}, exc_tb:{exc_tb}", style="danger")
        super().__exit__(exc_type, exc_value, exc_tb)


class SessionSaver(object):
    """
    USAGE:
    @SesssionSaver
    def session_saver(sess):
        ActionRightAfterEnter
        yield sess
        ActionRightBeforeExitEvenIfException
    with session_saver() as sess:
        sess.run(init_op)
        #...
    """

    def __init__(self, function):
        self.gen = function

    def __call__(self, *args, **kwargs):
        return _SessionSaver(self.gen, *args, **kwargs)
# StopTraining=trainable.StopTraining


def ask_restore(restore_list, savers, sess, do_ask=None, failsafe=True, index_combos=None, sysarg_key="--restore", restore_suffix=None, restore_state=None):
    if "--no-restore" in sys.argv:
        color_print("--no-restore given -- not restoring any models", style="danger")
        return
    sysarg_val=util.get_valv(sysarg_key, "ask")
    _do_ask=True
    if do_ask is not None:
        _do_ask=do_ask
        names_to_restore = restore_list
    elif sysarg_val != "ask":
        color_print(f"Restoring key {sysarg_val}")
        _do_ask=False
        if index_combos is None:
            index_combos = []
        index_combos = [] + index_combos + [("All", list(range(len(restore_list)))), ("None", [])]
        combo_firsts = [nme[0].lower() for nme, _ in index_combos]
        sysarg_value_idx = sys.argv.index(sysarg_key) + 1
        sysarg_value = sys.argv[sysarg_value_idx]
        if sysarg_value in combo_firsts:
            restore_combo = index_combos[combo_firsts.index(sysarg_value)]
            color_print(f"Restoring combo '{restore_combo[0]}'", style="notice")
            names_to_restore = [restore_list[i] for i in restore_combo[1]]
        else:
            errstr=f"Cannot find combo starting with {sysarg_value} ({sysarg_key})"
            if not failsafe:
                raise click.BadParameter(errstr)
            else:
                color_print("TRIGGERED FAILSAFE", style="error")
                color_print(errstr, style="error")
                _do_ask=True

    if _do_ask:
        color_print(f"{sysarg_val} given", style="warning")
        _, names_to_restore = util.ask_list(restore_list, message="Choose models to restore", names=None, allow_all=True, allow_none=True, index_combos=index_combos, failsafe=False)

    for saver_name in names_to_restore:
        print(f"Restoring {saver_name}")
        restore_dir = None
        # if saver_name == "pretrain":
        # restore_dir=None
        # if restore_suffix is not None:
        #     print(savers[saver_name].restore_directory)
        #     restore_dir=savers[saver_name].restore_directory+restore_suffix
        # if not os.path.isfile(restore_dir+".index"):
        # color_print(f"Could not restore version {restore_suffix} from {saver_name}. Attempting to restore default.", style="danger")
        # restore_dir=None
        savers[saver_name].restore(sess, restore_dir=None, restore_suffix=restore_suffix, restore_state=restore_state)
    if not names_to_restore:
        print("Not restoring any models")
    # not_restored=list(set(restore_list)-set(names_to_restore))
    # if not_restored:
    #     not_restored_str=",".join(not_restored)
    #     if not "--keep-tb" in sys.argv:
    #         color_print("{FATALBG}{WHITE}--keep-tb not given: Deleting tensorboard for not_restored: {not_restored_str}{ENDC}", not_restored_str=not_restored_str)
    #     else:
    #         return
    #     for sname in not_restored:
    #         pass


from shutil import copytree


def ask_save(save_list, savers, sess, do_ask=True, index_combos=None, failsafe=True, sysarg_key="--save", sleep=5, save_copy_suffix=None, state=None, label=None):
    if "--discard" in sys.argv:
        color_print("--discard given -- not saving any models", style="danger")
        return
    do_dialog = True
    if index_combos is None:
        index_combos = []
    sysarg_val=util.get_valv(sysarg_key, "ask")
    if sysarg_val != "ask":
        do_dialog = False
        tmp_index_combos = [("All", list(range(len(save_list)))), ("None", [])] + index_combos
        combo_firsts = [nme[0].lower() for nme, _ in tmp_index_combos]
        sysarg_value_idx = sys.argv.index(sysarg_key) + 1
        sysarg_value = sys.argv[sysarg_value_idx]
        if sysarg_value in combo_firsts:
            restore_combo = tmp_index_combos[combo_firsts.index(sysarg_value)]
            tcflush(sys.stdin, TCIFLUSH)
            saving_msg = f"SAVING combo '{restore_combo[0]}'."
            if "--rush" not in sys.argv:
                saving_msg += f"Enter any key (within {sleep} seconds) to prevent, and get dialog"
            color_print(saving_msg, style="warning")
            start = time.time()
            nb_char = ""
            while True and "--rush" not in sys.argv:
                nb_char = util.read_stdin()
                # if start+sleep <= time.time() or nb_char:
                if start + sleep <= time.time() or nb_char:
                    if nb_char:
                        print(nb_char)
                    break
                time.sleep(0.1)
            if nb_char:
                do_dialog = True
                color_print(f"Not saving. Getting dialog", style="notice")
            else:
                names_to_save = [save_list[i] for i in restore_combo[1]]
        else:
            try:
                raise click.BadParameter(f"Cannot find combo starting with {sysarg_value} ({sysarg_key})")
            except click.BadParameter as e:
                if not failsafe:
                    raise
                else:
                    print(e)
                    do_dialog=True
    if do_dialog:
        tcflush(sys.stdin, TCIFLUSH)
        _, names_to_save = util.ask_list(save_list, message="Choose models to save", names=None, allow_all=True, allow_none=True, index_combos=index_combos, failsafe=failsafe)
    for saver_name in names_to_save:
        print(f"Saving {saver_name}")
        savers[saver_name].do_save(sess, save_dir=None, label=label, state=state, suffix=save_copy_suffix)
        # color_print("ONLY PRETENDING TO SAVE (NO SAVING ACTUALLY HAPPENED!)", style="danger")
        # if suffix:
        #
        # else:
        #     savers[saver_name].do_save(sess)
        #
            # copytree(src=savers[saver_name].save_directory,dst=savers[saver_name].save_directory)
    if not names_to_save:
        print("Not saving...")


class StopTraining(Exception):
    """Invoke to stop trainable from training"""
#

    def __init__(self, msg, trainable=None, *args, **kwargs):
        #         if trainable is not None:
        # #             #pylint: disable=W0212
        #             if trainable.__sess is not None and trainable.saveable:
        #                 trainable.__sess.run(trainable._saver_save())
        #                 print(f"Saving state of {self.name}...")
        #             else:
        #                 color_print(f"Not saving state of {self.name}...",style="warning")
        #
        super(StopTraining, self).__init__(msg, *args, **kwargs)


class FinishedEpoch(Exception):
    def __init__(self, msg, trainable=None):
        self.trainable = trainable
        super(FinishedEpoch, self).__init__(msg)


class EarlyStopping(StopTraining):
    def __init__(self, msg, *args, **kwargs):
        color_print(f"EARLY STOPPING:{{ENDC}}\n{{S_WARNING}}{msg}", style="warning")
        print(msg)
        super(EarlyStopping, self).__init__(msg, *args, **kwargs)


class NOT_PROVIDED():
    pass


class _INTERNAL_MARKER():
    def __init__(self,mark):
        self.mark=mark
    def __eq__(self,other):
        return (isinstance(other, self.__class__) and other.mark == self.mark)

class _MARKER():
    def __init__(self,mark, value=None):
        self.mark=mark
        self.value=value

class TRYLIST():
    ALL=_MARKER(0)
    RESTORE_DIR=_MARKER(2)
    NONE=_MARKER(1)
    FULL_ONLY=_MARKER(3)
    PRETRAINS=_MARKER(4)
    DEFAULT=_MARKER(5)


class _NOSAVER():
    def __init__(self,reason=""):
          self.reason=reason

class SaverObject(util.Dottify):
    """docstring for SaverObject."""

    def __init__(self, *args, **kwargs):
        super(SaverObject, self).__init__(*args, **kwargs)
        self.saver = None

    def add_vars(self, *vars):
        if self._inited:
            raise AttributeError("Cannot add variables after saver is initialized")
        for var in vars:
            self.var_list.append(var)

    @property
    def _inited(self):
        return (self.saver is not None)

    def init(self):
        try:
            self.saver = tf.train.Saver(var_list=self.var_list)
        except ValueError as e:
            if str(e) == "No variables to save":
                errstr=str(e)
                print(f"Couldn't make saver for {self.scope}, because `{errstr}`")
                self.saver = _NOSAVER(errstr)

    def do_save(self, sess, save_dir=None, label=None, state=None, suffix=None):
        save_basedir = save_dir or self.save_directory
        label=label or self.label
        final_suffix=""
        final_suffix+=suffix or ""
        final_suffix+=f"-{state}" if state else ""
        final_suffix+=f"-{label}" if label else ""
        save_dir=f"{save_basedir}{final_suffix}"
        # if self.saver is False:
            # color_print(f"{self.scope} doesn't have a saver, so skipping save...", style="warning")
        if not self._inited:
            color_print("Initializing saver automatically (do_save)", style="warning")
            self.init()
        try:
            print(f"Saving {self.scope} to {save_dir}")
            if self.saver is False:
                color_print(f"{self.scope} doesn't have a saver...", style="warning")
                return False
            orig=self.saver.save(sess, save_dir)
            if label:
                label_save_dir=f"{save_basedir}-{label}"
                color_print(f"Also {self.scope} to {label_save_dir}")
                self.saver.save(sess, label_save_dir)
            return orig
        except ValueError as e:
            if str(e) == "No variables to save":
                color_print("{self.scope} has no variables to save, so skipping...", style="warning")
            else:
                raise
        except Exception as e:
            color_print(f"COULD NOT SAVE {self.scope}", style="danger")
            print(e)
    def restore(self, sess, restore_dir=None, restore_state=None, restore_suffix=None, try_list=None, verbose=3):
        if not self._inited:
            if verbose > 0:
                color_print("Initializing saver automatically (restore)", style="warning")
            if verbose > 1:
                print(f"savedir {restore_dir}")
            self.init()

        if isinstance(self.saver, _NOSAVER):
            color_print(f"Could not restore {self.scope} because it does not have a saver ({self.saver.reason})", style="warning")
            return
        try_list = try_list or []
        restore_dir = restore_dir or self.restore_directory
        restore_state = restore_state or ""
        restore_label=self.label or None
        restore_suffix = restore_suffix or ""
        # print(restore_suffix)
        # print(restore_label)
        # xit()
        restorevar_dict = dict(
            restore_dir=restore_dir,
            restore_suffix=restore_suffix,
            restore_state=restore_state,
            restore_label=restore_label
            )
        if self.try_list:
            try_list=self.try_list
        if try_list==TRYLIST.DEFAULT:
            try_restore = []
            restore_base=f"{restore_dir}"
            if restore_suffix:
                restore_base+=f"{restore_suffix}"
            if restore_state and restore_dir:
                try_restore.append(f"{restore_base}-{restore_dir}-{restore_state}")
            if restore_state:
                try_restore.append(f"{restore_base}-{restore_state}")
            if restore_label and restore_suffix:
                try_restore.append(f"{restore_base}-{restore_label}")
            # try_restore.append(f"{restore_dir})

        elif try_list==TRYLIST.ALL:
            try_restore = [
                f"{restore_dir}{restore_suffix}-{restore_state}-{restore_label}",
                f"{restore_dir}{restore_suffix}-{restore_state}",
                f"{restore_dir}{restore_suffix}-{restore_label}",
                f"{restore_dir}-{restore_state}",
                f"{restore_dir}{restore_suffix}",
                f"{restore_dir}"
            ]
        elif try_list==TRYLIST.PRETRAINS:
            try_restore=[]
            color_print(f"RESTORING TRYLIST.PRETRAINS. lbl: {self.label}", style="notice")
            if restore_label and restore_suffix:
                try_restore.append(f"{restore_dir}{restore_suffix}-{restore_label}")
            if restore_label:
                try_restore.append(f"{restore_dir}-{restore_label}")
            # try_restore.append(f"{restore_dir}")
        elif try_list==TRYLIST.RESTORE_DIR:
            try_restore=f"{restore_dir}"
        elif try_list==TRYLIST.FULL_ONLY:
            try_restore=f"{restore_dir}{restore_suffix}-{restore_state}",
        else:
            if not isinstance(try_list, (list,str)):
                raise TypeError(f"try_list must be list,str or TYPELIST. Found {try_list}={type(try_list)}")
            try_restore = [d.format(**restorevar_dict) for d in ensure_list(try_list)]
        if not try_restore:
            raise ValueError("Nothing to restore. Maybe not enough params available (label, suffix, state)?")
        if verbose > 0:
            print(f"Restoring {self.scope} from {restore_dir}")

        # return
        do_restore_dir = None
        for r_dir in try_restore:
            if os.path.isfile(r_dir + ".index"):
                do_restore_dir = r_dir
                if verbose > 0:
                    color_print(f"ATTEMPTING RESTORE FROM: {do_restore_dir}", style="notice")
                break
            else:
                # color_print(f"Could not restore from {r_dir}", style="notice")
                if verbose > 1:
                    print(f"Could not restore from {r_dir}")
        # if os.path.isfile(restore_dir_state+".index"):
            # restore_dir=restore_dir_state
            # color_print(f"Restoring state {restore_state}", style="notice")
        # else:
            # color_print(f"Could not find state restoring: for {restore_dir_state}. Trying with {restore_dir}", style="warning")
        # if not os.path.isfile(do_restore_dir + ".index"):
        if not do_restore_dir:
            print(f"Tried: {try_restore}")
            if not util.ask_user(f"Could not find checkpoint for {self.scope}. Not restoring. Continue?", default=True):
                raise FileNotFoundError(f"Could not find any checkpoint for {self.scope}.")
            return False
        else:
            try:
                if not self.saver:
                    if verbose > 0:
                        color_print(f"No saver found for {self.scope}. Continuing.")
                    return None
                res = self.saver.restore(sess, do_restore_dir)
                return res
            except BaseException as e:
                if not util.ask_user("Couldn't restore {}. Err: {} Continue?".format(self.scope, str(e)), default=True):
                    raise
                    # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
                    # latest_ckp = tf.train.latest_checkpoint(self.restore_directory)
                    # print(latest_ckp)
                    # xit()
                    # pp(print_tensors_in_checkpoint_file(self.restore_directory, all_tensors=True, tensor_name=''))
                    # raise


def update_saver(saver, save_root, restore_root=None, saver_name=None, save_dir_name=None, restore_dir_name=None, var_list=None, restore_scope_name=None, save_scope_name=None):
    scope_name=saver.scope
    if restore_scope_name is None:
        restore_scope_name = scope_name
    if save_scope_name is None:
        save_scope_name = scope_name
    if save_dir_name is None:
        save_dir_name = scope_name
    if restore_dir_name is None:
        restore_dir_name = save_dir_name

    if saver_name is None:
        saver_name = scope_name
    if restore_root is None:
        restore_root = save_root
    restore_dir = os.path.join(restore_root, restore_dir_name, restore_scope_name)
    save_dir = os.path.join(save_root, save_dir_name, save_scope_name)
    new_values=dict(
        var_list=var_list,
        restore_directory=restore_dir,
        save_directory=save_dir
    )
    saver.update(new_values)
    return saver

def new_saver(scope_name, save_root, restore_root=None, saver_name=None, save_dir_name=None, restore_dir_name=None, var_list=None, restore_scope_name=None, save_scope_name=None, label=None, try_list=None):
    if restore_scope_name is None:
        restore_scope_name = scope_name
    if save_scope_name is None:
        save_scope_name = scope_name
    if save_dir_name is None:
        save_dir_name = scope_name
    if restore_dir_name is None:
        restore_dir_name = save_dir_name

    if saver_name is None:
        saver_name = scope_name
    if var_list is None:
        var_list = get_variables_in(scope_name)
    if restore_root is None:
        restore_root = save_root
    restore_dir = os.path.join(restore_root, restore_dir_name, restore_scope_name)
    save_dir = os.path.join(save_root, save_dir_name, save_scope_name)
    saver_obj = SaverObject(
        var_list=var_list,
        scope=scope_name,
        restore_directory=restore_dir,
        save_directory=save_dir,
        label=label,
        try_list=try_list or []
    )
    return saver_obj

    # setattr(saver_obj,"do_save", _do_save)
    # saver_obj.restore = types.MethodType(_restore, saver_obj)
    # saver_obj.do_save = types.MethodType(_do_save, saver_obj)
    # setattr(savers, saver_name, saver_obj)

def savers_from_cfg(cfg):
    return util.Dottify(
        classifier=new_saver("classifier", cfg.models_dir, save_dir_name=cfg.source_dataset, restore_dir_name=cfg.source_dataset, label=cfg.trainers.pretrain.label, try_list=TRYLIST.PRETRAINS),
        source_map=new_saver("source_map", cfg.models_dir, save_dir_name=cfg.source_dataset, restore_dir_name=cfg.source_dataset, label=cfg.trainers.pretrain.label, try_list=TRYLIST.PRETRAINS),
        target_map=new_saver("target_map", cfg.model_save_root, save_dir_name=cfg.target_dataset, restore_dir_name=cfg.target_dataset, try_list=TRYLIST.DEFAULT),
        discriminator=new_saver("discriminator", cfg.model_save_root, try_list=TRYLIST.DEFAULT),
        cross_src_target=new_cross_saver(name="cross_src_target", save_root=cfg.models_dir, source_scope="source_map", dest_scope="target_map", save_dir_name=cfg.target_dataset, restore_dir_name=cfg.source_dataset, label=cfg.trainers.pretrain.label,try_list=TRYLIST.PRETRAINS)
    )
# def new_savers(*scope_names):
#     saver_objs=[]
#     for s in scope_names:
#         saver_objs.append(new_saver(s))

def cross_map(src_scope, dst_scope):
    src_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, src_scope)
    dst_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, dst_scope)
    src_names = [v.op.name for v in src_variables]
    dst_names = [v.op.name for v in dst_variables]
    # src_names=[v.name for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, src_scope)]
    # dst_names=[v.op.name for v in hlp.get_variables_in(dst_scope)]
    # pp(src_names)

    # src_variables=[var for var in tf.global_variables() if var.op.name in src_names]
    # dst_variables=[var for var in tf.global_variables() if var.op.name in dst_names]

    assert len(src_names) == len(dst_names) == len(dst_variables) == len(src_variables), "Number of variables in cross does not match"
    cmap_list = {}
    for src, dst in zip(src_names, dst_variables):
        cmap_list[src] = dst
    return cmap_list


def new_cross_saver(name, save_root, source_scope, dest_scope, save_dir_name, restore_dir_name, var_list=None, **kwargs):
    var_list=var_list or cross_map(source_scope, dest_scope)
    return new_saver(scope_name=name, save_root=save_root, var_list=var_list, save_dir_name=save_dir_name, restore_dir_name=restore_dir_name, save_scope_name=dest_scope, restore_scope_name=source_scope,  **kwargs)


def sanitize_scope_name(name):
    n = name
    n = n.replace(" ", "_")
    return n


# def make_accumulator(tensor, ret_all=False):
#     accumulator=tf.Variable(initial_value=tf.zeros_like(tensor), dtype=tensor.dtype)
#     batch_nums=tf.Variable(initial_value=tf.zeros_like(tensor),dtype=tensor.dtype)
#     accumulate_op=tf.assign_add(accumulator, tensor)
#     step_batch=tf.assign_add(batch_nums,1)
#     update_op=tf.group([step_batch, accumulate_op])
#     eps=1e-5
#     output_tensor=accumulator/(tf.nn.relu(batch_nums-eps)+eps)
#     flush_op=tf.group([tf.assign(accumulator, 0), tf.assign(batch_nums, 0)])
#     return output_tensor,update_op,flush_op
from abc import abstractmethod as absmeth


class BatchPersistent(object):
    pass
#     """docstring for BatchPersistent."""
#     def __init
#     @property
#     @absmeth
#     def doesnthave(self):
#         pass

# from abc import ABC, abstractmethod

# class MeanAccumulator(BatchPersistent):
#     """docstring for MeanAccumulator."""
#
#     def __init__(self, tensor, name=None, collections=None):
#         self.name = name
#         self.tensor = tensor
#         self.collections = collections
#         self.accumulator = accumulator = tf.Variable(initial_value=0, dtype=tensor.dtype)
#         self.batch_nums = batch_nums = tf.Variable(initial_value=0, dtype=tf.float32)
#         self.accumulate_op = accumulate_op = tf.assign_add(accumulator, tf.reduce_sum(tensor))
#         self.step_batch = step_batch = tf.assign_add(batch_nums, 1)
#         self.update_op = update_op = tf.group([step_batch, accumulate_op])
#         eps = 1e-5
#         self.output_tensor = accumulator/(tf.ones_like(tensor)*batch_nums)
#         self.flush_op = flush_op = tf.group([tf.assign(accumulator, 0), tf.assign(batch_nums, 0)])
#         super(MeanAccumulator, self).__init__()
#
#     @staticmethod
#     def make_accumulator(tensor):
#         a = MeanAccumulator(tensor)
#         return a.output_tensor, a.update_op, a.flush_op


class MeanAccumulator(BatchPersistent):
    """docstring for MeanAccumulator."""

    def __init__(self, tensor, name=None, collections=None):
        self.name = name
        self.tensor = tf.identity(tensor)
        self.collections = collections
        self.accumulator = accumulator = tf.Variable(initial_value=tf.zeros_like(tensor), dtype=tensor.dtype, trainable=False)
        self.batch_nums = batch_nums = tf.Variable(initial_value=tf.zeros_like(tensor), dtype=tensor.dtype, trainable=False)
        self.accumulate_op = accumulate_op = tf.assign_add(accumulator, tensor)
        self.step_batch = step_batch = tf.assign_add(batch_nums, 1)
        self.update_op = update_op = tf.group([step_batch, accumulate_op])
        eps = 1e-5
        self.output_tensor = output_tensor = accumulator / (tf.nn.relu(batch_nums - eps) + eps)
        self.flush_op = flush_op = tf.group([tf.assign(accumulator, 0), tf.assign(batch_nums, 0)])
        super(MeanAccumulator, self).__init__()

    @staticmethod
    def make_accumulator(tensor):
        a = MeanAccumulator(tensor)
        return a.output_tensor, a.update_op, a.flush_op


# class MeanAccumulator:
#     def __init__(self,name,tensor,collections):
#          self.name=name
#          self.input_tensor=tensor
#          self.collections=collections
#          self.output_tensor,self.opdate_op,self.flush_op=make_accumulator(tensor)

    # def __init__(self, name, tensor, collections):
    #     ### Definitions
    #     self.name=name
    #     self.input_tensor=tensor
    #     self.collections=collections
    #     self.accumulator=tf.Variable(initial_value=tf.zeros_like(self.input_tensor),name=sanitize_scope_name(f"{self.name}_accumulator"), dtype=self.input_tensor.dtype)
    #     self.batch_nums=tf.Variable(initial_value=tf.zeros_like(self.input_tensor),dtype=self.input_tensor.dtype)
    #     ### Define update options
    #     # Accumulate
    #     accumulate_op=tf.assign_add(self.accumulator, self.input_tensor)
    #     # Increase batch_size by one
    #     step_batch=tf.assign_add(self.batch_nums,1)
    #     # Make an update operation and an output_tensor
    #     self.update_op=tf.group([step_batch, accumulate_op])
    #     self.output_tensor=self.accumulator/self.batch_nums
    #     self.flush_op=tf.group([tf.assign(self.accumulator, 0), tf.assign(self.batch_nums, 0)])
    # def accumulate(self):
def splitlist_indices(*lists_or_list_lengths):
    lens = []
    for l in lists_or_list_lengths:
        if isinstance(l, (list, tuple)):
            lens.append(len(l))
        elif isinstance(l, int):
            lens.append(l)
    start_idx = 0
    ret_indices = []
    for llen in lens:
        ret_indices.append(util.Slicer[start_idx:start_idx + llen])
        start_idx += llen
    return ret_indices


def uniquify_filename(path, sep='', return_number=False):
    # https://stackoverflow.com/questions/13852700/python-create-file-but-if-name-exists-add-number
    def name_sequence():
        count = itertools.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s=sep, n=next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
        fd, filename = tempfile.mkstemp(dir=dirname, prefix=filename, suffix=ext)
        tempfile._name_sequence = orig
    return filename
# def savable_graph():
#     #------ Memory checkpoint operations ------#
#     # For saving model checkpoints to memory
#     graph = tf.get_default_graph()
#     gvars = graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#     assign_ops = [graph.get_operation_by_name(v.op.name + "/Assign") for v in gvars]
#     init_values = [assign_op.inputs[1] for assign_op in assign_ops]
#     gvars_state = sess.run(gvars)
#
# # Global variables by bad design
# saved_state = None  # Holds a saved graph state for saving and/or restoring from memory


def get_graph_params(collection=None):
    if collection is None:
        collection = tf.GraphKeys.GLOBAL_VARIABLES
    gvars = tf.get_collection(collection)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}


def restore_graph_params(graph_params):
    gvar_names = list(graph_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: graph_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


set_graph_params = restore_graph_params


def sum_epoch(ops, feed):
    ops = ensure_list(ops)
    sess = tf.get_default_session()
    totals = None  # type: list
    while True:
        try:
            tmp = sess.run(ops, feed)
            if totals is None:
                totals = tmp
            else:
                for i, t in enumerate(tmp):
                    totals[i] += t  # pylint: disable=E1137
        except tf.errors.OutOfRangeError:
            break
    return totals


import numpy as np


def optimal_auc_cutoff(labels, predictions, num_thresholds, feed):
    thresholds = np.linspace(0, 1, num_thresholds, dtype=np.float32)
    pre_local = set(tf.local_variables())
    _, b_true_positives = tf.metrics.true_positives_at_thresholds(labels, predictions, thresholds)
    _, b_true_negatives = tf.metrics.true_negatives_at_thresholds(labels, predictions, thresholds)
    _, b_false_positives = tf.metrics.false_positives_at_thresholds(labels, predictions, thresholds)
    _, b_false_negatives = tf.metrics.false_negatives_at_thresholds(labels, predictions, thresholds)
    cmtx_list = [b_true_positives, b_true_negatives, b_false_positives, b_false_negatives]
    post_local = set(tf.local_variables()) - pre_local
    tf.get_default_session().run(tf.variables_initializer(post_local))
    # tf.get_default_session().run(tf.local_variables_initializer())
    true_positives, true_negatives, false_positives, false_negatives = sum_epoch(cmtx_list, feed)
    positives = true_positives + false_negatives
    negatives = true_negatives + false_positives
    tp_rate = true_positives / positives
    fp_rate = false_positives / negatives
    corner_distance = np.hypot(fp_rate, tp_rate - 1)
    best_threshold_idx = np.argmin(corner_distance)
    optimal_cutoff = thresholds[best_threshold_idx]
    return optimal_cutoff


    # totals,batch_ops=list(zip(*total_batch_pairs))
# class GobbleOut():
#     def write(*args, **kwargs):
#         pass
#     def __enter__(*args, **kwargs):
#         pass
#     def __exit__(*args, **kwargs):
#         pass
import types


class _DummyClass(object):
    """
    A class that doesn't do anything when methods are called, items are set and get etc.
    I suspect this does not cover _all_ cases, but many.
    """

    def _returnself(self, *args, **kwargs):
        return self
    __getattr__ = __enter__ = __exit__ = __call__ = __getitem__ = _returnself

    def __str__(self):
        return ""
    __repr__ = __str__

    def __setitem__(*args, **kwargs):
        pass

    def __setattr__(*args, **kwargs):
        pass


class c_with(object):
    """
    Wrap another context manager and enter it only if condition is true.
    Parameters
    ----------
    condition:  bool
        Condition to enter contextmanager or possibly else_contextmanager
    contextmanager: contextmanager, lambda or None
        Contextmanager for entering if condition is true. A lambda function
        can be given, which will not be called unless entering the contextmanager.
    else_contextmanager: contextmanager, lambda or None
        Contextmanager for entering if condition is true. A lambda function
        can be given, which will not be called unless entering the contextmanager.
        If None is given, then a dummy contextmanager is returned.
    """

    def __init__(self, condition, contextmanager, else_contextmanager=None):
        self.condition = condition
        self.contextmanager = contextmanager
        self.else_contextmanager = _DummyClass() if else_contextmanager is None else else_contextmanager

    def __enter__(self):
        if self.condition:
            self.contextmanager = call_if_lambda(self.contextmanager)
            return self.contextmanager.__enter__()
        elif self.else_contextmanager is not None:
            self.else_contextmanager = call_if_lambda(self.else_contextmanager)
            return self.else_contextmanager.__enter__()

    def __exit__(self, *args):
        if self.condition:
            return self.contextmanager.__exit__(*args)
        elif self.else_contextmanager is not None:
            self.else_contextmanager.__exit__(*args)
# def islambda(v):
#   LAMBDA = lambda:0
#   return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__
# def call_if_lambda(f):
#     return f if not islambda(f) else f()
# # from contextlib import nullcontext
#
# class Gobble(object):
#     def __getattr__(self, item):
#         return self
#
#     def __call__(self, *args, **kwargs):
#         return self
#
#
# class c_with(object):
#     """
#     Wrap another context manager and enter it only if condition is true.
#     """
#     def __init__(self, condition, contextmanager, else_contextmanager=None):
#         """
#         @param condition: Condition to enter contextmanager or possibly else_contextmanager
#         @param contextmanager: Contextmanager, or lambda function returning a contextmanager, for entering if condition is true
#         @param else_contextmanager: Contextmanager, lambda function returning a contextmanager or None. Entering if condition is true, and returning a dummy context manager if None given.
#         """
#         self.condition = condition
#         self.contextmanager = contextmanager
#         self.else_contextmanager = Gobble() if else_contextmanager is None else else_contextmanager
#     def __enter__(self):
#         if self.condition:
#             self.contextmanager=call_if_lambda(self.contextmanager)
#             return self.contextmanager.__enter__()
#         elif self.else_contextmanager is not None:
#             self.else_contextmanager=call_if_lambda(self.else_contextmanager)
#             return self.else_contextmanager.__enter__()
#     def __exit__(self, *args):
#         if self.condition:
#             return self.contextmanager.__exit__(*args)
#         elif self.else_contextmanager is not None:
#             self.else_contextmanager.__exit__(*args)


class _NOT_PROVIDED():
    pass


class Bestie():
    BESTIE_COLLECTION = "BESTIE_COLLECTION"
    _value = None
    tf_value = None
    comparer = "<"
    _upd_conds = property(
        lambda self: {
            "<": lambda new_value_, self=self: new_value_ < self.value,
            "<=": lambda new_value_, self=self: new_value_ <= self.value,
            ">": lambda new_value_, self=self: new_value_ > self.value,
            ">=": lambda new_value_, self=self: new_value_ >= self.value
        }
    )

    def __init__(self, trainable, comparer="<", condition=None, init_val=None, vals=None, param_state=None, name=None, collection=_NOT_PROVIDED, verbose=False, print_keys=None):
        """
        @param vals: callable or tf_op. tf_op will be run in group.
        callable must take two args: (tr,sess)
        """
        if isinstance(collection, _NOT_PROVIDED):
            collection = Bestie.BESTIE_COLLECTION
        if param_state is not None and name is None:
            print(f"Setting name to param_state ({param_state})")
            name = param_state
        if name is None:
            raise TypeError("A Bestie needs a name (or param_state)!")
        comparer = comparer if condition is None else None
        init_vals = {
            ">": 0,
            ">=": 0,
            "<": float("inf"),
            "<=": float("inf")
        }
        self.tr = trainable  # type: Trainable
        self.verbose = verbose
        #pylint: disable=W0212
        default_vals = OrderedDict(
            step=self.tr._tf_step,
            epoch=self.tr._tf_epoch
        )
        #pylint: enable=W0212

        vals = vals or {}
        if not isinstance(vals, OrderedDict):
            vals = OrderedDict(vals)

        self._value = init_vals[comparer] if comparer is not None else init_val
        if self._value is None:
            if init_val is None:
                raise TypeError("If custom condition is given, init_val must also be given")
            else:
                raise TypeError("Bestie._value ended up being None?")

        self._state_val_funcs = merge_dicts(default_vals, vals)
        self.state_vals = {k: "N/A" for k in self._state_val_funcs}
        self.comparer = comparer
        self.condition = condition or self._upd_conds[comparer]
        self.param_state = param_state
        self.name = name
        self.print_keys = print_keys or []
        if collection is not None:
            self.tf_value = tf.Variable(initial_value=self.value, name=name, trainable=False)
            tf.add_to_collection(collection, self.tf_value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = val
        self._set_tf_value()

    def _get_tf_value(self):
        return tf.get_default_session().run(self.tf_value)

    def _set_tf_value(self, val=None):
        if self.tf_value is not None and tf.get_default_session() is not None:
            if val is None:
                val = self._value
            tf.get_default_session().run(self.tf_value.assign(val))

    def try_update(self, new_value):
        if self.condition(new_value):
            sess = tf.get_default_session()
            old_value = self.value
            self.value = new_value
            # print(f"UPDATING VAL {self.name}: {old_value} -> {new_value}")
            run_ops = OrderedDict()
            for name, value_op in self._state_val_funcs.items():
                if callable(value_op):
                    self.state_vals[name] = value_op(self.tr, sess)
                else:
                    run_ops[name] = value_op
            names = list(run_ops.keys())
            run_ops = list(run_ops.values())
            # self.tr.init_it_val()
            run_result = sess.run(run_ops, self.tr.get_reuseable_feed_val(feed_name=self.name + ":feed"))
            # run_result=self.tr.get_epoch_value_val(run_ops)
            # result_dict=OrderedDict()
            for n, r in zip(names, run_result):
                # result_dict[n]=r
                self.state_vals[n] = r
            self.state_vals[self.value_name] = new_value
            if self.tf_value is not None:
                self.tf_value.assign(new_value)
            if self.param_state:
                color_print(f"SAVING PARAM STATE {self.param_state}", style="warning")
                self.tr.save_param_state(self.param_state)

    @property
    def summary_head(self):
        return color_print(f"{self.name}", style="success", as_str=True)+"\n"

    @property
    def value_name(self):
        return self.name or "value"

    def summary(self, verbose=None):
        if verbose is None:
            verbose = self.verbose
        out_str = self.summary_head + "\n"
        out_str += self.value_name
        out_str += ":" + str(self._value) + "\n"
        for k, v in self.state_vals.items():
            if k==self.value_name:
                continue
            if self.verbose or k in self.print_keys:
                out_str += f"{str(k)}: {str(v)}\n"

        # return "\n".join(": ".join(str(_) for _ in kv) for kv in self.state_vals.items())
        return out_str

    def __str__(self):
        return self.summary()

        # for k,v in self.state_vals.items():
        #     out_str
# class Bestie(_Bestie):
#     def __init__(self, *args, **kwargs):
#           super(Bestie, self).__init__(*args,**kwargs)


class TF_Bestie(Bestie):
    _try_update = lambda self, *args, **kwargs: self.try_update(*args, **kwargs)
    # self.tr.confusion
    # def try_update(self, *args,**kwargs):
    # raise AttributeError("try_update is removed from TF_BESTIE")
    # comparer="<", condition=None, init_val=None, vals=None, param_state=None, name=None, collection=_NOT_PROVIDED, verbose=False, print_keys=None
    def __init__(self, trainable, tf_op, comparer="<", condition=None, init_val=None, vals=None, param_state=None, name=None, collection=_NOT_PROVIDED, verbose=False, print_keys=None,**kwargs):
        super(TF_Bestie, self).__init__(trainable=trainable, comparer=comparer, condition=condition, init_val=init_val, vals=vals, param_state=param_state, name=name, collection=collection, verbose=verbose, print_keys=print_keys, **kwargs)
        self.tf_op = tf_op

    def run_op(self):
        return self.tr.get_epoch_value_val(self.tf_op, max_summary_steps=self.tr.opts.max_summary_steps)

    def update(self, new_val=None):
        if new_val is None:
            new_val = self.run_op()
        self._try_update(new_val)
        return self.summary()


class CB_Bestie(Bestie):
    _try_update = lambda self, *args, **kwargs: self.try_update(*args, **kwargs)
    # self.tr.confusion
    # def try_update(self, *args,**kwargs):
    # raise AttributeError("try_update is removed from TF_BESTIE")

    def __init__(self, trainable, callback, *args, **kwargs):
        super(CB_Bestie, self).__init__(trainable, *args, **kwargs)
        self.callback = callback

    def run_op(self):
        return self.callback(self)

    def update(self, new_val=None):
        if new_val is None:
            new_val = self.run_op()
        self._try_update(new_val)
        return self.summary()
from tensorflow.python.client import device_lib
def available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
