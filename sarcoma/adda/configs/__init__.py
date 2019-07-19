from abc import ABC, abstractmethod
from .. import util
from copy import deepcopy
import sys
import os
from sys import exit as xit
import tensorflow as tf

import util
from typing import Type, Callable
import typing

import inspect
from pprint import pprint as pp

class RequiredError(AttributeError):
    pass


class NotReadyError(AttributeError):
    pass


class ReadOnlyError(AttributeError):
    pass


class ReservedAttributeError(AttributeError):
    pass

class Optional(object):
    def __init__(self,value):
        self.value=value
class Required(object):
    """docstring for Required."""

    def __init__(self):
        # self.instance=instance
        # self.required_instance = instance
        self.__call__ = self._fail
        self.__repr__ = self._fail
        self.__str__ = self._fail
        self.__get__ = self._fail

    def _fail(self, *args, **kwargs):
        err_str=f"Could not find required attribute {super(Required, self).__str__()} in config."
        raise RequiredError(err_str)
        pass


class CFG_TEMPLATE:
    pass


def make_template(cls):
    return type(cls.__name__,(cls,CFG_TEMPLATE),cls.__dict__.copy())
    # cls = self.__class__  # pylint: disable=E0203
    # self.__class__ = cls.__class__(cls.__name__, (cls, CFG_TEMPLATE), {}))

class _CFG(util.Dottify):
    def __init__(self, from_dict=None, is_template=False, **kwargs):
        # super(_CFG,self).__init__(from_dict=from_dict, **kwargs)
        if is_template:
            cls = self.__class__  # pylint: disable=E0203
            self.__class__ = cls.__class__(cls.__name__, (cls, CFG_TEMPLATE), {})
        # print("ARCH:", [isinstance(self,c) for c in [CFG_PRINTER,ARCHITECTURE_CFG,_CFG]])
        # help(self)

        # print(f"{self.__class__.__name__}", default_attrs)
        self.__dict__ = {}
        # print(f"Is template?: {is_template}")
        self.initialize(from_dict=from_dict, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.__dict__, name):
            return getattr(self.__dict__, name)
        # super(Dottify, self).__getattr__(name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def _set_attr_from_dict(self):
        for k, v in self.__dict__.items():
            setattr(self, k, v)
        # if "_meta_dict" in self.__dict__:
            # del self.__dict__["_meta_dict"]
    # def update()

    def __call__(self, from_dict=None, **kwargs):
        if from_dict is None:
            from_dict = {}
        # if not self.is_template:
        #     raise TypeError(f"Config class {self.__class__.__name__} is not callable unless it's a template.")
        self.__dict__.update(from_dict)
        self.__dict__.update(kwargs)
        self._set_attr_from_dict()
        self._check_required()
        return self
    def _set_missing_defaults(self):
        all_varnames=set(dir(self))
        cfg_varnames=set(dir(_CFG))
        unique_attributes=all_varnames-cfg_varnames
        default_attrs={}
        for attr_name in unique_attributes:
            if attr_name[0]=="_":
                continue
            if attr_name in self.__dict__:
                continue
            if not hasattr(self, attr_name):
                continue
            attr_inst=getattr(self, attr_name)
            if isinstance(attr_inst, Required) or callable(attr_inst):
                continue
            else:
                print(f"Not given {attr_name}. Setting it to default {attr_name}={attr_inst}")
                default_attrs[attr_name]=attr_inst

    def initialize(self, from_dict=None, **kwargs):
        if from_dict is None:
            from_dict = {}
        self.__dict__.update(from_dict)
        self.__dict__.update(kwargs)
        self._set_attr_from_dict()
        # if not self.is_template:
        # print(f"CHECKING REQUIRED {self.__class__.__name__}")
        if not isinstance(self, CFG_TEMPLATE):
            self._check_required()

    def _iterator(self):
        for k, v in self.__dict__.items():
            yield k, v

    def __iter__(self):
        return self._iterator()

    def _check_required(self):
        for attr, attr_inst in inspect.getmembers(self):
            if isinstance(attr_inst, _CFG):
                continue
            # if isinstance(attr_inst, Optional):
            #     self.__dict__
            if isinstance(attr_inst, Required):
                print(attr, attr_inst)
                err_str=f"Could not find required attribute {attr} in config {self.__class__.__name__}."
                raise RequiredError(err_str)

        # self._checked_required = True
    # def __getattribute__(self,attr):
    #     # if _getattr is not set, behave as normal
    #     if attr in ["_lock_getattr", "_checked_requirements", "_check_required", "__class__"]:
    #         return super(_CFG, self).__getattribute__(attr)
    #     if not hasattr(self, "_lock_getattr"):
    #         return super(_CFG, self).__getattribute__(attr)
    #     if self._checked_requirements:
    #         return super(_CFG, self).__getattribute__(attr)
    #     try:
    #         return self._check_required()
    #     except Exception as e:
    #         print(attr)
    #         raise
        # return super().__getattribute__(attr)
    # def _super__getattribute(self,attr):
    #     return super().__getattribute__(attr)
    #
    # def _fail__getattribute__(self, attr):
    #     print("attr", attr)
    #     self._check_required()
    #     if attr in ["_checked_required", "_check_required"]:
    #         return super().__getattribute__(attr)


class TR_CFG(_CFG):
    """docstring for Training."""
    source_batch = Required()
    target_batch = Required()
    print_summaries = False
    prefetch=False
    ds_s_val_send=[]
    ds_t_val_send=[]
    # ds_tr_send=[]


class Saving_CFG(_CFG):
    restore_list = []
    save_list = []

class COMBO_CFG(_CFG):
    initial_disc_train_steps=0
    initial_tm_train_steps=0
    disc_train_first=True
    simultaneous_iteration=False
    disc_tm_train_epoch_count=(1, 1)
    disc_tm_train_step_count=(1, 1)
    max_steps=None
    printerval=10
    write_out_interval=10
    printerval_seconds=True
    reset_tm_optimizer_steps=0
    reset_disc_optimizer_steps=0
    disc_steps_after_reset=0
    tm_steps_after_reset=0
    cluster_reset_after=True
    cluster_reset_both_after=False

class ADVERSARIAL_CFG(_CFG):
    max_epochs=float("inf")
    scope_re=Required()
    reset_optimizers=False
    printerval=10
    min_steps=0
    early_stopping=util.Dottify(
        loss_difference_threshold=None,
        loss_difference_memory=None,
        percentage_total_decrease=None
    )
    trainable=Required()
    optimizer=Required()
    saving=Required()

class Trainer_CFG(_CFG):
    pretrain = Required()  # type: ADVERSARIAL_CFG
    target = Required()  # type: ADVERSARIAL_CFG
    disc = Required()  # type: ADVERSARIAL_CFG
    adversarial_saving = Required()  # type: Saving_CFG


# def getset(func):
#     internal_name=f"_{func.__name__}"
#     print("internalname", internal_name)
#     def _getter(self,*args,**kwargs):
#         func(self,*args,**kwargs)
#         return getattr(self, internal_name)
#     def _setter(self,val):
#         return setattr(self, internal_name, val)
#     return {fget=_getter, fset=_setter, fdel=None, doc=None}


# class COMBO_SETTINGS(_CFG):
#     disc_tm_train_epoch_count = Required()
#     disc_tm_train_step_count = Required()
#     initial_disc_train_steps = Required()
#     initial_tm_train_steps = Required()
#     disc_train_first = Required()
#     simultaneous_iteration = Required()
#     write_out_interval = Required()
#     # discriminato


class CFG_Models(_CFG):
    source_map = Required()
    target_map = Required()
    discriminator = Required()
    classifier = Required()


class ARCHITECTURE_CFG(object):
    def __init__(self, *args, **kwargs):
        super(ARCHITECTURE_CFG, self).__init__(*args, **kwargs)
        self._savers = None

    @property
    def savers(self):
        if self._savers is None:
            raise NotReadyError("Tried to access savers before they were set")
        return self._savers

    @savers.setter
    def savers(self, savers):
        if self._savers is not None:
            raise ReadOnlyError("Tried to set savers more than once")
        else:
            self._savers = savers


import json
import shutil
sh_columns, sh_rows = shutil.get_terminal_size(fallback=(80, 24))
import pprint
# pp=ppf.pprint
from collections.abc import Iterable
import pout


def adam_summary(optim):
    return dict(
        instance=optim,
        learning_rate=optim._lr,
        beta1=optim._beta1,
        beta2=optim._beta2,
        epsilon=optim._epsilon
    )
def pout_summary(optim):
    return pout.s(optim)
    pass


class CFG_PRINTER(object):
    # def __init__(self, *args, **kwargs):
    #     print("ARCH_PR:", [isinstance(self,c) for c in [CFG_PRINTER,ARCHITECTURE_CFG,_CFG]])
    #     super(CFG_PRINTER,self).__init__(*args,**kwargs)
    # def pretty(d, indent=0):
    #    for key, value in d.items():
    #       print('\t' * indent + str(key))
    #       if isinstance(value, dict):
    #          pretty(value, indent+1)
    #       else:
    #          print('\t' * (indent+1) + str(value))
    _handlers = {
        tf.train.AdamOptimizer: adam_summary,
        tf.train.GradientDescentOptimizer: pout_summary,
        tf.TensorShape: lambda x: f"{str(x)} (calculated)"
    }

    def _check_handler_formats(self, attr_inst):
        found = False
        if hasattr(attr_inst, "summary_call"):
            # print(f"{attr_inst} had summary_call")
            attr_inst=attr_inst()
            # print(f"is now:{attr_inst}")
        for attr_type, attr_handler in self._handlers.items():
            if isinstance(attr_inst, attr_type):
                info = attr_handler(attr_inst)
                return info
            else:
                # print(f"{attr_inst} is not {attr_type}")
                pass
        if hasattr(attr_inst,"_summary"):
            if callable(attr_inst._summary):
                return attr_inst._summary()
            else:
                return attr_inst._summary
        return None
    # def summary_linear(self):
    #     # Iterate all members
    #     summary_dict={}
    #     # pout.v(self.__dir__())
    #     # xit()
    #     for attr, attr_inst in inspect.getmembers(self):
    #         # if isinstance(attr_inst, util.Dottify):
    #         if hasattr(attr_inst, "__dict__"):
    #             summary_dict[attr]=attr_inst.__dict__
    #         else:
    #             key,info=self._check_handler_formats(attr, attr_inst)
    #             if (key,info) != (None, None):
    #                 if key is None: key = attr
    #                 summary_dict[key]=info
    #             elif not callable(attr_inst):
    #                 print (f"{attr}:{hasattr(attr_inst, 'im_self')}")
    #                 summary_dict[attr]=attr_inst
    #             # elif hasattr(attr_inst, "__str__"):
    #
    #
    #     # xit()
    #     pout.v(summary_dict)

    def summary_recursive(self, root=None, do_print=True):
        summary_dict = self._summary_recursive(self)
        ppf=pprint.PrettyPrinter(indent=0, width=sh_columns,depth=10, compact=False)
        # pout.v(summary_dict)
        # out=pprint.pformat(summary_dict,indent=0, width=sh_columns,depth=None)
        # print(out)
        # print(summary_dict)
        # ppf.pprint(summary_dict)
        summary=pout.s(summary_dict)
        if do_print:
            print(summary)
        return summary



    def _summary_recursive(self, root, depth=0, max_depth=4):
        summary_dict = {}
        # if not hasattr(root, "items"):
        formatted = self._check_handler_formats(root)
        if formatted is not None:
            # root=formatted
            return formatted
        if not isinstance(root, Iterable) or not hasattr(root, "items"):
            return root
        # root=dictify(root)
        for k, v in root.items():
            is_expandable = (hasattr(v, "__dict__"))
            if is_expandable and depth < max_depth:
                summary_dict[k] = self._summary_recursive(v, depth + 1)
            else:
                summary_dict[k] = v
        return summary_dict
        # return pprint.pformat(summary_dict)

    # @staticmethod
    # def _summary_recursive(root):
    #     is_expandable=(has_attr(root, "__dict__"))
    #     if is_expandable:
    #         for key,val in root.items():
    #             if hasattr(val, "__dict__"):
    #                 pass
    # _summary_recursive=staticmethod(_summary_recursive)

  #   def myprint(d):
  # for k, v in d.items():
  #   if isinstance(v, dict):
  #     myprint(v)
  #   else:
  #     print("{0} : {1}".format(k, v))
        # return _summary_recursive(root)
        # if hasattr(attr_inst, "__dict__"):
        # return
    summary = summary_recursive
    # json.dumps(summary_dict)
    # print("Dottify:")
    # if hasattr(attr_inst, "summary"):
    #     attr_inst.summary()
    # else:

    # if isinstance(attr_inst, _CFG):
    # continue
    # print(attr, attr_inst)
    # print(attr, attr_inst)
    # raise RequiredError(f"Could not find required attribute {attr} in config.")


def __auto_super(cls):
    def __init__(self, *args, **kwargs):
        super(cls, self).__init__(*args, **kwargs)
    return __init__


def auto_super(cls):
    def __init__(self, *args, **kwargs):
        super(cls, self).__init__(*args, **kwargs)
    setattr(cls, "__init__", __init__)
    return cls
# @auto_super


class ADDA_CFG(CFG_PRINTER, ARCHITECTURE_CFG, _CFG):
    def __init__(self, *args, **kwargs):
        super(ADDA_CFG, self).__init__(*args, **kwargs)
    models = Required()  # type: CFG_Models
    source_map = Required()
    target_map = Required()
    discriminator = Required()
    classifier = Required()
    source_dataset = Required()
    target_dataset = Required()
    trcfg = Required()  # type: TR_CFG
    trainers = Required()  # type: Trainer_CFG
    combo_settings = Required()  # type: COMBO_SETTINGS
    input_shape = None
    hidden_shape = None
    model_save_root = None
    model_save_subdir = ""


class ADDA_CFG_TEMPLATE(ADDA_CFG, CFG_TEMPLATE):
    pass

class TST_CFG(CFG_PRINTER, _CFG):
    def __init__(self, *args, **kwargs):
        super(TST_CFG, self).__init__(*args, **kwargs)

class TST_CFG_TMPL(CFG_PRINTER, _CFG, CFG_TEMPLATE):
    def __init__(self, *args, **kwargs):
        super(TST_CFG_TMPL, self).__init__(*args, **kwargs)

# T_ADDA_CFG=typing.NewType('ADDA_CFG', ADDA_CFG)
# T_ADDA_CFG=typing.TypeVar('ADDA_CFG', ADDA_CFG)

def summary_call(fn):
    fn.summary_call = True
    return fn

import collections
def recursive_update(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    # if isinstance(_merge_dct, _CFG):
    # merge_dct = dict(_merge_dct)
    # else:
    # merge_dct = _merge_dct
    for k, _ in merge_dct.items():
        key_in_dict = k in dct
        if isinstance(dct, util.Dottify):
            key_in_dict = k in dct.__dict__
        if key_in_dict and isinstance(dct[k], (dict, util.Dottify)) and isinstance(merge_dct[k], (collections.Mapping, _CFG, util.Dottify)):
            recursive_update(dct[k], merge_dct[k])
        else:
            # if key_in_dict:
            # print(f"Endval: ({type(v)}) {k}:{merge_dct[k]}: {isinstance(merge_dct[k],_CFG)}", "")
            # print((k in dct),isinstance(dct[k], (dict, _CFG)),isinstance(merge_dct[k], (collections.Mapping, _CFG, util.Dottify)))
            dct[k] = merge_dct[k]

def combine(template, *configs, trigger=True):
    template=deepcopy(template)
    name = template["model_save_subdir"] if "model_save_subdir" in template.keys() else ""
    for c in configs:
        if not isinstance(c, _CFG):
            if callable(c):
                c = c()
        if "model_save_subdir" in c.keys():
            name += c["model_save_subdir"]
        # template.update(c)
        # print(template)
        recursive_update(template, c)
    # print(template)
    # print(f"Making {name}")
    template=deepcopy(template)
    if trigger:
        return template(model_save_subdir=name)
    else:
        template.model_save_subdir=name
        return template
    # return template

class Triggerable(util.Dottify):
    def __init__(self, *args,**kwargs):
        super(Triggerable, self).__init__(*args,**kwargs)
    def __call__(self):
        cfg=self.callback
        cfg=util.call_if_lambda(cfg)
        if not isinstance(cfg, _CFG):
            raise ValueError(f"Invalid template: {cfg}")
        cfg._check_required()
        cfg._set_missing_defaults()

        # print("is_template:", cfg.is_template)
        if hasattr(cfg, "name"):
            raise ValueError(f"Config already has a name: `{cfg.name}`, so can't give it the name: `{self.name}`")
        cfg.name=self.name
        return cfg
    @property
    def template(self):
        return self.callback

def new_template(name, callback):
    if not (isinstance(name, str) and callable(callback)):
        raise TypeError("new_template: did you mix up positional arguments? \n Name has to be string and callback has to be callable!")
    untriggered_obj=Triggerable(
        untriggered=True,
        name=name,
        callback=callback
    )
    return untriggered_obj

# def trigger_config(config_object):
#     util.call_if_lambda(config_object)
#     if hasattr(config_object, "untriggered"):
#         config_object=config_object()
#     if hasattr(config_object, "is_template"):
#         config_object=config_object()
#     return config_object
def cfg_label(label, **kwargs):
    return ADDA_CFG_TEMPLATE(trainers=Trainer_CFG(is_template=True, pretrain=util.Dottify(label=label)), **kwargs)
