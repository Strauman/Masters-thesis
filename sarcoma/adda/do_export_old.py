import os
import numpy as np
from.helpers.cprints import color_print
import tensorflow as tf
import sys
# clearscreen()
from .seeding import set_seed_state
import matplotlib
matplotlib.use("PDF")
# {cfg.name}/
from . import export_help
from matplotlib import pyplot as plt
# plt.style.use("ggplot")
# plt.rcParams["axes.grid"] = False

import matplotlib.gridspec as gridspec

from .helpers.trainhelp import ensure_list
from . import trainhelp as hlp
from .graphbuilders.discriminator import Discriminator
# import do_tables as stuffer

# from .trainables.targetmap import Targetmap
# from .trainables.train import Pretrain
from .graphbuilders.targetmap import Targetmap as Targetmap
from .trainables.trainable import Trainable
from .graphbuilders.pretrain import Pretrain as Pretrain
# from .build_graph import *
# with open(config_save_file,"a") as f:
# f.write(f"SEED: {seed_state}")
from .trainables import EarlyStopping, StopTraining
from .helpers import util
from .export_help import take_random, run_epoch, ensure_lstcp, take_idc
from sys import exit as xit
from contextlib import contextmanager
from . import exporthead
export_seed_state = (10, 2432, 39, 4123)
top_val = 0.65
from collections import OrderedDict

N_OUT_WARN = 20
DS_s = DS_t = cfg = tensorboard_path = CONFIG_NAME = None
pretrain_settings = target_settings = disc_settings = None

import matplotlib2tikz
from shutil import rmtree

relpath = lambda path: os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), path))
tikz_basepath = lambda path: os.path.normpath(os.path.join(relpath(f"../result-figs"), path))
tikz_dirpath = lambda path: os.path.normpath(os.path.join(tikz_basepath(cfg.name), path))
tikz_filepath = lambda path: os.path.normpath(os.path.join(tikz_dirpath(path), f"figure.tex"))


def del_dir_contents(path):
    for the_file in os.listdir(path):
        file_path = os.path.join(path, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def ask_makedirs(path, do_empty=False):
    if not os.path.isdir(path):
        if util.ask_user(f"Recursively create {path}?", default=True):
            os.makedirs(path)
        else:
            raise FileNotFoundError(f"Can't find directory {path}")
    elif do_empty:
        if util.ask_user(f"Clear/rm {path}?", default=False, y_args=["--overwrite"]):
            print("CLEARING")
            del_dir_contents(path)
            # os.makedirs(path)
        else:
            print("NOT CLEARING")


def _save_tikz_export(filename, gs=None, custom_subdir=None):
    if custom_subdir:
        save_tikz_dir = os.path.join(custom_subdir, filename)
        save_tikz_path = os.path.join(save_tikz_dir, f"figure.tex")
    else:
        save_tikz_dir = tikz_dirpath(filename)
        save_tikz_path = tikz_filepath(filename)
    ask_makedirs(save_tikz_dir, do_empty=True)
    print(f"Saving to {save_tikz_path}")
    dpi = 72
    img_size = 128
    if gs is not None:
        nr, nc = gs.get_geometry()
        fig_wid = dpi * int(nc * 128) / 72
    textsize = dpi
    matplotlib2tikz.save(f"{save_tikz_path}")


def saveas(filename, gs=None, custom_subdir=None):
    if custom_subdir:
        preview_save_path = os.path.join(custom_subdir, filename)
    else:
        preview_save_path = tikz_dirpath(f"{filename}.pdf")
    ask_makedirs(os.path.dirname(preview_save_path))
    # if "--tikz" in sys.argv:
    # preview_save_path = tikz_dirpath(f"preview.pdf")
    # else:
    # preview_save_path = relpath(f"testfig.pdf")
    print(f"Preview saved to {preview_save_path}")
    plt.savefig(preview_save_path, bbox_inches='tight', dpi=300)
    if "--tikz" in sys.argv:
        _save_tikz_export(filename, gs=gs, custom_subdir=custom_subdir)


def callclass(cls):
    return cls()


@callclass
class aix(util.Dottify):
    X = None
    Y = None
    F1 = None
    Yhat = None
    indices = OrderedDict(
        X=lambda tr: tr.inputs,
        Y=lambda tr: tr.labels,
        F1=lambda tr: tr.indiv_f1,
        Yhat=lambda tr: tr.outputs
    )

    def setup(self):
        #pylint: disable=W0201
        self.functions = []
        for i, (arr, fn) in enumerate(self.indices.items()):
            setattr(self, arr, i)
            self.functions.append(fn)

    def tf_vars(self, tr):
        return [f(tr) for f in self.functions]


    # X=0
    # Y=1
    # F1=2
    # idcs=[0,1,2]
    # tf_arrs=lambda tr: [tr.inputs, tr.labels, tr.indiv_f1]
# aix = _ARRIDC()  # type: _ARRIDC
arr_idc = aix  # type: aix
delta_top_val = 0.05
cfg = cfg  # type: ADDA_CFG

# tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999)
# pretrain_settings = cfg.trainers.pretrain
# target_settings = cfg.trainers.target
# disc_settings = cfg.trainers.disc
best_pretrain_val = 0
sess = None
disc_trainer = None  # type: Discriminator
tm_trainer = None  # type: Targetmap
pretrainer = None  # type: Pretrain
trainers = []


def reset_seed(state=None):
    state = state or export_seed_state
    set_seed_state(export_seed_state)


def tf_dataset_from_data(X, Y):
    def gen():
        for x, y in zip(X, Y):
            yield (x, y)
    return tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32), output_shapes=(X[0].shape, Y[0].shape))
    # source_iter
    # target_iter

# FILTERS


def ensure_arrs_dec(f):
    def _wrapper(arrs, *args, **kwargs):
        arrs = ensure_lstcp(arrs)
        return f(arrs, *args, **kwargs)
    _wrapper.__name__ = f.__name__
    return _wrapper


def filter_best(arrs, idx, n_out=-1):
    arrs = ensure_lstcp(arrs)
    best_idx = np.argsort(arrs[idx], 0)[::-1]
    arrs = take_idc(arrs, best_idx, n_out=n_out)
    return arrs


def filter_worst(arrs, idx, n_out=-1):
    arrs = ensure_lstcp(arrs)
    best_idx = np.argsort(arrs[idx], 0)
    arrs = take_idc(arrs, best_idx, n_out=n_out)
    return arrs


def filter_area(arrs, idx, n_out=-1):
    arrs = ensure_lstcp(arrs)
    areas = np.argsort(np.sum(arrs[idx], axis=(1, 2)), 0)[::-1]
    output = take_idc(arrs, areas, n_out=n_out)
    return output


def filter_area_size(arrs, idx, n_out=-1, min_val=None, max_val=None):
    arrs = ensure_lstcp(arrs)
    areas = np.sum(arrs[idx], axis=(1, 2))
    if min_val is not None and max_val is not None:
        idc = np.where(areas >= min_val & areas <= min_val)
    elif min_val is not None:
        idc = np.where(areas >= min_val)
    elif max_val is not None:
        idc = np.where(areas <= max_val)
    else:
        raise ValueError("Either min_val or max_val must be given to filter_area_size")
    output = take_idc(arrs, idc, n_out=n_out)
    return output


def filter_quantile_range(arrs, idx, q_range=[0.25, 0.75]):
    arrs = ensure_lstcp(arrs)
    q_thresholds = np.quantile(arrs[idx], q_range)
    lower = q_thresholds[0]
    upper = q_thresholds[1]
    idc = (arrs[idx] >= lower) & (arrs[idx] <= upper)
    return take_idc(arrs, idc)


def filter_quantile(arrs, idx, n_out=-1, quantile=0.0, lower_tail=True):
    arrs = ensure_lstcp(arrs)
    q_threshold = np.quantile(arrs[idx], quantile)
    if lower_tail:
        idc = arrs[idx] <= q_threshold
    else:
        idc = arrs[idx] >= q_threshold
    return take_idc(arrs, idc, n_out=n_out)


def filter_area_quantile(arrs, idx=0, n_out=-1, quantile=0.3):
    arrs = ensure_lstcp(arrs)
    areas = np.argsort(np.sum(arrs[idx], axis=(1, 2)), 0)[::-1]
    # plt.hist(areas, bins='auto')
    # saveas()
    q_thresh = np.quantile(areas, quantile)
    print(f"Quantile val: {q_thresh}. a_max={np.max(areas)}, a_min={np.min(areas)}")
    area_idx = areas[areas <= q_thresh]
    arrs = take_idc(arrs, area_idx, n_out=n_out)
    return arrs


def filter_random(arrs, n_out=None):
    arrs = ensure_lstcp(arrs)
    arrs = take_random(arrs, n_out=n_out)
    return arrs


def filter_evenly(arrs, idx, n_out, min_index=0):
    sorted_idc = np.argsort(arrs[idx])
    spread_idc = np.linspace(min_index, arrs[idx].shape[0] - 1, n_out).astype(np.int)
    idc = sorted_idc[spread_idc]
    return take_idc(arrs, idc)


def filter_truncate(arrs, n_out):
    return [a[:n_out] for a in arrs]


@ensure_arrs_dec
def filter_minimum(arrs, idx, n_out=-1, min_val=0.0):
    filtered_idc = arrs[idx] >= min_val
    return take_idc(arrs, filtered_idc, n_out=n_out)


@ensure_arrs_dec
def filter_maximum(arrs, idx, n_out=-1, max_val=0.0):
    filtered_idc = arrs[idx] <= max_val
    return take_idc(arrs, filtered_idc, n_out=n_out)


def filter_sort(arrs, idx, n_out=None, reverse=False):
    arrs = ensure_lstcp(arrs)
    if reverse:
        new_order = np.argsort(arrs[idx], 0)
    else:
        new_order = np.argsort(arrs[idx], 0)[::-1]

    return take_idc(arrs, new_order, n_out=n_out)

# /FILTERS
# FILTER COLLECTIONS


def filter_f1_clean(arrs, n_out=7):
    arrs = filter_random(arrs)
    arrs = filter_truncate(arrs, n_out)
    arrs = filter_sort(arrs, idx=aix.F1)
    return arrs
# /FILTER COLLECTIONS


# CALLBACKS
def label_seg_f1(F1):
    np.set_printoptions(precision=3)

    def _wrap(r, c, gs, **kwargs):
        if c == 2:
            export_help.get_ax(r, c, gs).set_xlabel(f"F$_1$:{np.array2string(F1[r])}")
    return _wrap


def label_da_f1(F1_before, F1_after):
    np.set_printoptions(precision=3)

    def _wrap(r, c, gs, **kwargs):
        f1 = None
        if c == 2:
            f1 = F1_before
        elif c == 3:
            f1 = F1_after
        if f1 is not None:
            export_help.get_ax(r, c, gs).set_xlabel(f"F$_1$:{np.array2string(f1[r])}")
    return _wrap
# /CALLBACKS
# FIGURE GENERATORS


def get_tf_arrs_from_feed(tr, feed):
    print(f"Restore for getting ars for {tr.name}")
    # tr.init_it_val()
    arrs = run_epoch(feed, *aix.tf_vars(tr=tr))
    return arrs


def get_xy_arrs(tr):
    print(f"Getting X,Y from {tr.name}")
    # tr.init_it_val()
    arrs = run_epoch(tr.get_reuseable_feed_val("export"), aix.indices['X'](tr), aix.indices['Y'](tr))
    return arrs


def get_arrs(tr):
    print(f"Restore for getting ars for {tr.name}")
    # tr.init_it_val()
    arrs = run_epoch(tr.get_reuseable_feed_val("export"), *aix.tf_vars(tr=tr))
    return arrs


def get_f1_scores(tr=None):
    if tr is None:
        tr = tm_trainer
    # tr.init_its_val()
    f1_scores = tf.get_default_session().run(tr.indiv_f1, tr.feeds_val())
    return f1_scores


def f1_boxplot(filename, tr=None, f1_scores=None, **bplot_kwargs):
    if tr is None:
        tr = tm_trainer
    if f1_scores is None:
        f1_scores = get_f1_scores(tr)
    plt.figure()
    plt.boxplot(f1_scores, **bplot_kwargs)
    saveas(filename)


def filter_arrs(arrs, filters=None, n_out=None):
    filters = filters or []
    # Select data with good F1_scores after ADDA
    for cb in filters:
        if callable(cb):
            arrs = cb(arrs=arrs)
    if n_out is not None:
        arrs = [a[:n_out] for a in arrs]
    return arrs


def _data_for_grids(tr, filters=None, n_out=None, arr_func=get_arrs, get_arrs_kwargs=None):
    filters = filters or []
    if get_arrs_kwargs is None:
        arrs = arr_func(tr)
    else:
        arrs = arr_func(**get_arrs_kwargs)
    return filter_arrs(arrs, filters, n_out)


def _xy_for_grids(tr, filters=None, n_out=None):
    filters = filters or []
    return _data_for_grids(tr, filters, n_out, arr_func=get_xy_arrs)


def export_da_grid(filename, filters=None, n_out=None, **kwargs):
    filters = filters or []
    # Get data after adda:
    tr = tm_trainer
    handle = tr.DS_t.tf_it_handle
    arrs = _data_for_grids(tr=tr, filters=filters, n_out=n_out)
    X = arrs[arr_idc.X]
    Y = arrs[arr_idc.Y]
    F1 = arrs[arr_idc.F1]
    post_handle, DS_best = export_help.handle_from_data(X, Y, return_dataset=True)
    Yhat_post = run_epoch({handle: post_handle}, tr.outputs)
    # Get data before adda:
    # Reset session
    tf.get_default_session().run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    tr.init_it_val()
    Yhat_pre, F1_before = run_epoch({handle: export_help.handle_for_dataset(DS_best)}, tr.outputs, tr.indiv_f1)
    # Plot grid
    gs = export_help.export_grids([X, Y, Yhat_pre, Yhat_post], titles=["Input", "Ground\n truth", "Before DA", "After DA"], cb=label_da_f1(F1_before, F1), **kwargs)
    plt.gcf().suptitle("TITLE")
    saveas(filename, gs)


def export_segmentation_grid(filename="testfig", filters=None, n_out=None, **kwargs):
    filters = filters or []
    tr = pretrainer
    handle = tr.DS_s.tf_it_handle
    arrs = _data_for_grids(tr=tr, filters=filters, n_out=n_out)
    X = arrs[arr_idc.X]
    Y = arrs[arr_idc.Y]
    F1 = arrs[arr_idc.F1]
    Yhat = arrs[arr_idc.Yhat]
    # post_handle, DS_best = export_help.handle_from_data(X, Y, return_dataset=True)
    # Yhat = run_epoch({handle: post_handle}, tr.outputs)
    # Plot grid
    gs = export_help.export_grids([X, Y, Yhat], titles=["Input", "Ground\n truth", "Prediction"], cb=label_seg_f1(F1), **kwargs)
    saveas(filename, gs)


def export_segmentation_contours(filename, filters=None, n_out=10, **kwargs):
    filters = filters or []
    tr = pretrainer
    handle = tr.DS_s.tf_it_handle
    arrs = _data_for_grids(tr=tr, filters=filters, n_out=n_out)
    X = arrs[arr_idc.X]
    Y = arrs[arr_idc.Y]
    F1 = arrs[arr_idc.F1]
    Yhat = arrs[arr_idc.Yhat]
    contour_collection = [
        [Y, (X, dict(colors="green"))],  # Column 1
        [Y, (Yhat, dict(colors="red"))]  # Column 2
    ]
    # contour_collection = [X,[Y,(Yhat_pre, dict(colors="red"))]]
    # gs = export_help.export_contours(contour_collection)
    gs = export_help.export_contours(contour_collection, titles=["Input\n+ground truth", "ground truth+prediction"], **kwargs)  # , cb=label_da_f1(F1_before, F1), **kwargs)
    # plt.gcf().suptitle("TITLE")
    saveas(filename, gs)


def export_da_contours(filename, filters=None, **kwargs):
    filters = filters or []
    # Get data after adda:
    tr = tm_trainer
    handle = tr.DS_t.tf_it_handle
    arrs = _data_for_grids(tr=tr, filters=filters)
    X = arrs[arr_idc.X]
    Y = arrs[arr_idc.Y]
    F1_post = arrs[arr_idc.F1]
    Yhat_post = arrs[arr_idc.Yhat]
    post_handle, post_ds = export_help.handle_from_data(X, Y, return_dataset=True)
    # Yhat_post = run_epoch({handle: post_handle}, tr.outputs)
    # Get data before adda:
    # Reset session
    tf.get_default_session().run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    Yhat_pre, F1_pre = run_epoch({handle: post_handle}, tr.outputs, tr.indiv_f1)
    # Plot grid
    contour_collection = [
        [X, (Y, dict(colors="green"))],  # Column 1
        [Yhat_pre, (Y, dict(colors="red"))],  # Column 2
        [Y, (Yhat_post, dict(colors="magenta"))]  # Column 3
    ]
    # contour_collection = [X,[Y,(Yhat_pre, dict(colors="red"))]]
    # gs = export_help.export_contours(contour_collection)
    gs, figaxes = export_help.export_contours(contour_collection, titles=["Input", "Before\n ADDA", "After\n ADDA"], **kwargs)  # , cb=label_da_f1(F1_before, F1), **kwargs)
    # plt.gcf().suptitle("TITLE")
    saveas(filename, gs)

# a=np.arange(10).reshape(10)
# b=np.arange(10).reshape(10)
# c=np.arange(10)[::-1].reshape(10)
# arrs=[a,b,c]
# # print(np.concatenate(arrs,1))
# print(np.concatenate([np.expand_dims(a,-1) for a in arrs],1))
# # arrs=filter_best(arrs, idx=0, n_out=5)
# arrs=filter_best(arrs, idx=2, n_out=None)
# # arrs=filter_best(arrs, n_out=5)
# # arrs=filter_minimum(arrs, n_out=5, min_val=5)
# print([a.shape for a in arrs])
#
# # print(np.concatenate(arrs,1))
# print(np.concatenate([np.expand_dims(a,-1) for a in arrs],1))
# xit()


# /FIGURE GENERATORS


# TEMPS

# /TEMPS
# def save_preview()


def export_figs():
    reset_seed()
    n_out = 7
    export_segmentation_grid(filename=f"segmentation", filters=[
        # lambda arrs: filter_minimum(arrs,idx=2,n_out=n_out,min_val=0.7),
        # lambda arrs: filter_quantiles(arrs, idx=aix.F1, quantile=0.5, lower_tail=False),
        # lambda arrs: filter_area_size(arrs,idx=aix.Yhat, min_val=150),
        # lambda arrs: filter_random(arrs),
        ### NO RANDOMNESS BELOW ###
        # lambda arrs: filter_evenly(arrs, aix.F1, n_out),
        # lambda arrs: filter_truncate(arrs, n_out),
        # lambda arrs: filter_sort(arrs, idx=aix.F1),
        lambda arrs: filter_f1_clean(arrs, n_out)
    ],
        gs_args=dict(wspace=0.5)
    )
    # plt.gcf().suptitle("TITLE")


def da_default():
    # f1_scores=get_f1_scores(tm_trainer)
    # f1_scores=filter_quantiles(f1_scores, quantiles=[0.25,0.75],lower_tail=True)
    #
    # best_f1, threshold=tm_trainer.best_f1()
    # tm_trainer.output_threshold=threshold
    # f1_boxplot(filename="boxplot")
    # export_da_grid(filename=f"DA", filters=[
    #     lambda arrs: filter_quantile_range(arrs, aix.F1, q_range=[0.25, 0.75]),
    #     # lambda arrs: filter_random(arrs),
    #     ### NO RANDOMNESS BELOW ###
    #     lambda arrs: filter_f1_clean(arrs)
    #     # lambda arrs: filter_evenly(arrs, aix.F1, n_out),
    #     # lambda arrs: filter_truncate(arrs, 10),
    #     # lambda arrs: filter_sort(arrs, idx=aix.F1),
    # ],
    #     gs_args=dict(hspace=0.5))
    export_da_contours("DAContourGrid", filters=[
        ### NO RANDOMNESS BELOW ###
        lambda arrs: filter_f1_clean(arrs)
    ],
        gs_args=dict(hspace=0.5))
    # print(tm_trainer.best_f1())
    # export_da_grid(cfg_name)


config_maps = {
    'T2_PT_ADDA': da_default,
    'T2_PT_RESET': da_default
}

# LOADED_FAVS={}


def load_config_fav(fav):
    global cfg, tm_trainer, disc_trainer, pretrainer, trainers, LOADED_FAVS
    # if fav in LOADED_FAVS.keys():
    #     cfg, tm_trainer, disc_trainer, pretrainer=LOADED_FAVS[fav]
    if not cfg or cfg.name != fav.name:
        cfg, tm_trainer, disc_trainer, pretrainer = exporthead.load_config(fav.name)
        # LOADED_FAVS[fav]=(cfg, tm_trainer, disc_trainer, pretrainer)

    trainers = [tm_trainer, disc_trainer, pretrainer]
    do_load_state = False
    if hasattr(cfg, "state") and hasattr(cfg, "run_num"):
        do_load_state = cfg.state != fav.state
        do_load_state = do_load_state or cfg.run_num != fav.run_num
    else:
        cfg.state = fav.state
        cfg.run_num = fav.run_num
        do_load_state = True
    exporthead.get_state(fav.state, fav.run_num, cfg, *trainers)


def run_exports(cfg_prefix, numbers):
    global cfg, tm_trainer, disc_trainer, pretrainer
    number_adda = numbers[0]
    number_reset = numbers[1]
    for suff, num in zip(["_ADDA", "_RESET"], list(numbers)):
        cfg, tm_trainer, disc_trainer, pretrainer = exporthead.load_config(cfg_prefix + suff)
        trainers = [tm_trainer, disc_trainer, pretrainer]
        exporthead.get_state("abs_confusion", num,cfg, *trainers)
        if cfg.name in config_maps.keys():
            config_maps[cfg.name]()
        else:
            da_default()


def optlist(lst):
    if len(lst) == 1:
        return ensure_list(lst[0])
    else:
        return lst


class Result(util.Dottify):
    source = None
    target = None
    suff = None
    num = None
    state = None


class States():
    conf = "abs_confusion"
    best = "best_perf"
    loss = "best_loss"
    initial = init = "initial"
    latest = "latest"


def select_favs(favdict):
    # favout = OrderedDict()
    favout = []
    for outname, fname in favdict.items():
        updates = None
        if isinstance(fname, (list, tuple)):
            fname, updates = fname
        fav = favs[fname]  # type: Result
        if updates is not None:
            # print("Updated")
            # print(fav)
            fav = fav.merge_cp(updates)
            # print("to")
            # print(fav)
        # favout[outname] = fav
        favout.append(fav)
    return favout


def compare_adda_reset(filename, filters=None, **kwargs):
    filters = filters or []
    # Get data after adda:
    tr = tm_trainer
    handle = tr.DS_t.tf_it_handle
    arrs = _data_for_grids(tr=tr, filters=filters)
    X = arrs[arr_idc.X]
    Y = arrs[arr_idc.Y]
    F1_post = arrs[arr_idc.F1]
    Yhat_post = arrs[arr_idc.Yhat]
    post_handle, post_ds = export_help.handle_from_data(X, Y, return_dataset=True)
    # Yhat_post = run_epoch({handle: post_handle}, tr.outputs)
    # Get data before adda:
    # Reset session
    # tf.get_default_session().run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    # Yhat_pre, F1_pre = run_epoch({handle: post_handle}, tr.outputs, tr.indiv_f1)
    # Plot grid
    contour_collection = [
        [X, (Y, dict(colors="green"))],  # Column 1
        [Yhat_pre, (Y, dict(colors="red"))],  # Column 2
        [Y, (Yhat_post, dict(colors="magenta"))]  # Column 3
    ]
    # contour_collection = [X,[Y,(Yhat_pre, dict(colors="red"))]]
    # gs = export_help.export_contours(contour_collection)
    gs, figaxes = export_help.export_contours(contour_collection, titles=["Input", "Before\n ADDA", "After\n ADDA"], **kwargs)  # , cb=label_da_f1(F1_before, F1), **kwargs)
    # plt.gcf().suptitle("TITLE")
    saveas(filename, gs)


class Comparer(object):
    def __init__(self, favs, filename, out_dirname=None, filters=None):
        self.filename = filename
        self.outpath = tikz_basepath(out_dirname) if out_dirname else None
        self.favs = favs
        self.titles = []
        self.filters = filters or [
            ### NO RANDOMNESS BELOW ###
            # lambda arrs: filter_f1_clean(arrs, n_out=7)
        ]

    def process_self(self):
        for fav in self.favs:
            reset_seed()
            load_config_fav(fav)
            self.process_fav(fav)
        reset_seed()
        self.finalize()


@callclass
class UNSET():
    pass


class ResultValues(util.Dottify):
    X = UNSET
    Y = UNSET
    F1 = UNSET
    Yhat = UNSET

    def __init__(self, tf_value_array, **custom_vals):
        value_dict = OrderedDict()
        for kname in aix.indices.keys():
            value_dict[kname] = tf_value_array[getattr(aix, kname)]
            # setattr(self, kname, tf_value_array[getattr(aix,kname)])
        super().__init__(from_dict=value_dict, **custom_vals)


class ContourStandardResultValues(ResultValues):
    Yhat_pre = None
    Yhat_post = None
    F1_pre = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ContourCompare(Comparer):
    def __init__(self, favs, filename, out_dirname=None, contour_cb=None, same_data=True, out_rows=7, XY_filters=None, tr=None, ds_cb=None, filters=None):
        super().__init__(favs=favs, filename=filename, filters=filters, out_dirname=out_dirname)
        self.contour_collection = []
        self._data = None
        self._same_data = same_data
        self.F1=[]
        self._XY = {} if self._same_data else None
        self.DS = None
        self.current_target_name = None
        default_tr = lambda: tm_trainer
        self.tr = tr or default_tr
        self._tr = self.tr
        default_ds_cb = lambda tr: tr.DS_t
        self.ds_cb = ds_cb or default_ds_cb
        self._ds_cb = self.ds_cb
        self.XY_filters = XY_filters or [
            lambda arrs: filter_random(arrs)
        ]
        self.out_rows = out_rows

        def contour_cb_default(comp, data: ContourStandardResultValues):
            retlist = []
            # if not hasattr(comp, "did_truth"):
            #     comp.did_truth = True
            #     comp.titles += ["Input + \n ground truth"]
            #     retlist += [
            #         [data.X, (data.Y, dict(colors="green"))],  # Column 1
            #     ]

            retlist += [
                [data.Yhat, (data.Y, dict(colors="red"))],
                # [data.Yhat_pre, (data.Y, dict(colors="red"))],  # Column 2
                # [data.Yhat_post, (data.Y, dict(colors="magenta"))]
            ]  # Column 3
            return retlist

        # self.contour_cb=contour_cb or contour_cb_default
        self.contour_cb = contour_cb_default

    def finalize(self):
        def f1labels(F1):
            np.set_printoptions(precision=3)
            def _wrap(r, c, gs, **kwargs):
                # if c == 2:
                export_help.get_ax(r, c, gs).set_xlabel(f"F$_1$:{np.array2string(F1[r][c])}")
            return _wrap
        # print(self.contour_collection)
        gs, figax = export_help.export_contours(self.contour_collection, titles=self.titles)#, cb=label_seg_f1(np.array(self.F1).reshape(self.out_rows,-1)))
        saveas(self.filename, gs, custom_subdir=self.outpath)

    @property
    def XY(self):
        if self._same_data:
            if self.current_target_name not in self._XY:
                return None
            return self._XY[self.current_target_name]
        else:
            return self._XY

    def _set_data(self):
        tr = self.tr()
        # tr=tm_trainer
        handle = self.ds_cb(tr).tf_it_handle
        del self.DS
        self.DS = export_help.tf_dataset_from_data(*self.XY)
        self.DS = self.DS.batch(min(self.out_rows, 10))
        ds_feed = lambda: {handle: export_help.handle_for_dataset(self.DS)}
        reset_seed()
        arrs = filter_arrs(get_tf_arrs_from_feed(tr, ds_feed()), filters=self.filters+[lambda arrs: filter_truncate(arrs, n_out=self.out_rows)])
        self.F1+=list(arrs[aix.F1])
        self._data = ContourStandardResultValues(arrs, Yhat_post=arrs[aix.Yhat])

    def _setXY(self, target_name=None):
        reset_seed()
        XY = _xy_for_grids(tr=tm_trainer, filters=self.XY_filters)
        if self._same_data:
            target_name = target_name or self.current_target_name
            self._XY[target_name] = XY
        else:
            self._XY = XY

    @property
    def data(self):
        if not self._data:
            self._set_data()
        return self._data

    def process_fav(self, fav):
        if fav not in self.favs:
            return
        self.tr = self._tr
        self.ds_cb = self._ds_cb
        if hasattr(fav, "tr"):
            self.tr = fav.tr
        if hasattr(fav, "ds_cb"):
            self.ds_db = fav.ds_cb
        # Check if we want the same data at all?
        if not self._same_data:
            # We don't care about the same data.
            self._setXY()
            # Set variables in case someone wants to use them
            self.current_target_name = fav.target
        elif not self.XY or not self.current_target_name:
            # Init -- first run we have to set target!
            self.current_target_name = fav.target
            self._setXY(fav.target)
        # Check if we changed the target
        elif self.current_target_name != fav.target:
            # Changed target - we have to change data!
            self.current_target_name = fav.target
            self._setXY(fav.target)

        self._set_data()
        if hasattr(fav, "contour_cb"):
            self.contour_collection += ensure_list(fav.contour_cb(comp=self, data=self.data))
        else:
            self.contour_collection += ensure_list(self.contour_cb(comp=self, data=self.data))
        self.titles += fav.titles


def process_multiple(comparers, favs):
    # for fname, fav in favs.items():
    for fav in favs:
        for c in comparers:
            load_config_fav(fav)
            c.process_fav(fav)
    for c in comparers:
        c.finalize()


if __name__ == '__main__':
    # load_config("cfg_name")
    # load_config("T2_PT_RESET")
    # stuffer.load_config()
    MOD_FAVS = [  # ACTUAL FAVS
        ("PT_T2", (78, 50)),
        ("PT_CT", (1, 5)),
        ("CT_PT", (110, 21)),
        ("CT_T2", (1, 4)),
        ("T2_CT", ("0", "0")),
        ("T2_PT", (41, 72)),

    ]
    favs = {}
    for prefix, restore_numbers in MOD_FAVS:
        for suff, run_num in zip(["ADDA", "RESET"], list(restore_numbers)):
            source, target = prefix.split("_")
            favs[prefix + f"_{suff}"] = Result(
                source=source,
                target=target,
                suff=suff,
                run_num=run_num,
                state=States.conf,
                name=prefix + f"_{suff}"
            )

    # MOD_FAVS = [## ACTUAL FAVS
    #     ("PT_T2", (78, 50)),
    #     # ("T2_PT", (39, 71)),
    #     ("T2_PT", (41, 72)),
    #     # ("CT_T2", (1,4)),
    #     # ("CT_PT", (110, 21)),
    #     # ("PT_CT", (1,5)),
    # ]
    # for pref, runs in MOD_FAVS:
    #     run_exports(pref, nums)

    input_cb = lambda comp, data: [[data.X, (data.Y, dict(colors="red"))]]
    pred_cb = lambda comp, data: [[data.Yhat, (data.Y, dict(colors="red"))]]
    input_pred = lambda comp, data: [[
        data.X,
        (data.Y, dict(colors="green")),
        (data.Yhat, dict(colors="red"))
    ]]

    def combine_ccb(*ccbs):

        def _wrap(comp, data):
            out_lst = []
            for ccb in ccbs:
                out_lst += ccb(comp, data)
        return _wrap
    rvsadda = OrderedDict(
        input=("PT_T2_RESET", Result(state=States.initial, titles=["Input + ground truth"], contour_cb=input_cb)),
        pre=("PT_T2_RESET", Result(state=States.initial, titles=["Before \n DA"], contour_cb=pred_cb)),
        reset=("PT_T2_RESET", Result(state=States.conf, titles=["RESET \n AFTER"], contour_cb=pred_cb)),
        adda=("PT_T2_ADDA", Result(state=States.conf, titles=["ADDA \n AFTER"], contour_cb=pred_cb))
    )
    RESET_VS_ADDA = select_favs(rvsadda)
    INPUT_MODALITIES = select_favs(OrderedDict(
        pt=("CT_PT_RESET", Result(tr=lambda: tm_trainer, state=States.initial, titles=["PET image"], contour_cb=input_cb)),  # <- DOING INPUT OF TARGET DATASET!
        ct=("T2_CT_RESET", Result(tr=lambda: tm_trainer, state=States.initial, titles=["CT image"], contour_cb=input_cb)),  # <- DOING INPUT OF TARGET DATASET!
        t2=("PT_T2_RESET", Result(tr=lambda: tm_trainer, state=States.initial, titles=["MRI T2 iage"], contour_cb=input_cb))  # <- DOING INPUT OF TARGET DATASET!
    )
    )
    SEGMENTATION = select_favs(OrderedDict(
        t2=("PT_T2_RESET", Result(state=States.conf, titles=["T2 image seg"], contour_cb=input_pred)),  # <- DOING INPUT OF _TARGET_ DATASET!
        pt=("CT_PT_RESET", Result(state=States.conf, titles=["PT image seg"], contour_cb=input_pred)),  # <- DOING INPUT OF _TARGET_ DATASET!
        ct=("T2_CT_RESET", Result(state=States.conf, titles=["CT image seg"], contour_cb=input_pred)),  # <- DOING INPUT OF _TARGET_ DATASET!
    ))
    all_favs = [RESET_VS_ADDA, INPUT_MODALITIES]
    # ContourCompare(favs=RESET_VS_ADDA, filename="comparefirst", out_dirname="PT_T2_ADDA_RESET").process_self()
    # ContourCompare(favs=INPUT_MODALITIES, filename="inputmodalities", out_dirname="misc", same_data=False).process_self()
    # F1best Filter
    seg_def_filter=[
        lambda arrs: filter_best(arrs, aix.F1)
    ]
    # load_config_fav(select_favs({"A":("T2_CT_RESET", Result(state=States.conf, titles=["T2 image seg"], contour_cb=input_pred))})[0])
    # print(pretrainer.best_f1())
    # ContourCompare(favs=SEGMENTATION, filename="segmentation", tr=lambda: pretrainer, ds_cb=lambda tr: tr.DS_s, out_dirname="misc", same_data=False, filters=seg_def_filter).process_self()

    # domain_grid_fig(pretrainer, DS_s)
    # domain_grid_fig(tm_trainer, DS_t)
    # saveas()
    # f1_large_areas()
    # f1_for_a_quantiles()
    # export_da_f1best_grid()
    # reset_seed()
    # plt.figure()
    # plt.title("Domain adaptation results")
    #
    # reset_seed()
    # export_da_grid(filename="testfig", callbacks=[
    #     # lambda arrs:filter_area(arrs, aix.Yhat, n_out=n_out),
    #     lambda arrs:filter_min_area(arrs, aix.Yhat, min_val=100),
    #     # lambda arrs: filter_best(arrs,aix.F1,n_out=n_out),
    #     lambda arrs: filter_worst(arrs,aix.F1,n_out=n_out),
    #     # lambda arrs: filter_minimum(arrs,idx=aix.F1,n_out=None,min_val=0.5),
    #     # lambda arrs: filter_quantiles(arrs, idx=aix.F1, quantile=0.25, lower_tail=True),
    #     # lambda arrs: filter_random(arrs,n_out=n_out),
    #     ### NO RANDOMNESS BELOW ###
    #     lambda arrs: filter_truncate(arrs, n_out),
    #     # lambda arrs: filter_evenly(arrs, aix.F1, n_out),
    #     lambda arrs: filter_sort(arrs, idx=aix.F1),
    # ],
    #     gs_args=dict(wspace=0.5, hspace=0.25)
    # )
    # plt.title("")
