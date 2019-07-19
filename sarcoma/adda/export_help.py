import tensorflow as tf
import matplotlib
matplotlib.use("PDF")
from matplotlib import pyplot as plt
# plt.style.use("ggplot")
# plt.rcParams["axes.grid"] = False


import matplotlib.gridspec as gridspec
# from . import util
from .helpers.util import ensure_list, merge_dicts
import os
import sys
from sys import exit as xit
relpath = lambda path: os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), path))
# class Subplotter(object):
#     """docstring for Subplotter."""
#
#     def __init__(self, ncol, nrow):
#         super(Subplotter, self).__init__()
#         self.ncol = ncol
#         self.nrow = nrow
#     def subplot_args(self, idx):


def ensure_lstcp(arrs):
    if isinstance(arrs, list):
        a=arrs
    elif isinstance(list, tuple):
        a=list(arrs)
    else:
        a=[arrs]
    return a+[]

def run_epoch(feed, *ops):
    results = []
    ops = ensure_list(ops)
    sess=tf.get_default_session()
    while True:
        try:
            r = ensure_list(sess.run(ops, feed))
            results.append(r)
        except tf.errors.OutOfRangeError:
            break

    out_results = [*results[0]]
    for i in range(len(results) - 1):
        for j, mtx in enumerate(results[i + 1]):
            out_results[j] = np.concatenate([out_results[j], mtx])
    # for o_r in out_results:
    #     print(o_r.shape)
    if len(out_results) == 1:
        return out_results[0]
    return out_results

class _Slicer(object):
    def __getitem__(self, val):
        return val
Slicer = _Slicer()

def take_random(arrs, idc=None, n_out=-1):
    """
        get `n_out` number of elements of arrs chosen randomly from the top `pool_size` of idcs
    """
    arrs=ensure_lstcp(arrs)
    pick_idc=idc or Slicer[:]
    if idc:
        num_choices=idc.shape[0]
    else:
        num_choices=arrs[0].shape[0]
    if n_out is not None and n_out > 0:
        random_picks = np.random.choice(num_choices, n_out)
    else:
        random_picks = np.random.choice(num_choices, num_choices)

    for i, arr in enumerate(arrs):
        arrs[i] = arr[pick_idc][random_picks]
    return arrs

def take_idc(arrs, idc, n_out=None):
    """
        get `n_out` number of elements of arrs chosen randomly from the top `pool_size` of idcs
    """
    arrs=ensure_lstcp(arrs)
    idc=np.squeeze(idc)
    if n_out is not None and n_out > 0:
        s_out = Slicer[:n_out]
    else:
        s_out = Slicer[:]

    for i, arr in enumerate(arrs):
        arrs[i]=arrs[i][idc][s_out]
    return arrs

def tf_dataset_from_data(X, Y):
    def gen():
        for x, y in zip(X, Y):
            yield (x, y)
    return tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.int32), output_shapes=(X[0].shape, Y[0].shape))


def handle_from_data(X, Y, batch=10, return_dataset=False):
    new_dataset = tf_dataset_from_data(X, Y)
    new_dataset = new_dataset.batch(batch)
    ret = handle_for_dataset(new_dataset)
    if return_dataset:
        ret = [ret, new_dataset]
    return ret


def handle_for_dataset(ds):
    new_iterator = ds.make_one_shot_iterator()
    s_handle = tf.get_default_session().run(new_iterator.string_handle())
    return s_handle

AXES=None
def make_figaxes(nrows,ncols):
    global AXES
    fig,ax=plt.subplots(nrows,ncols,figsize=(ncols,nrows))
    AXES=ax
    return fig,ax

def set_figaxes(fig, ax):
    global AXES
    plt.figure(fig.number)
    AXES=ax

def get_ax(row, col, gs):
    nrows, ncols = gs.get_geometry()
    plt_idx=row * ncols + col
        # axes=axes.ravel()
    # axes=plt.gca()
    plt.sca(AXES[row,col])
    return AXES[row,col]

    # return plt.subplot(gs[plt_idx])
    # return plt.subplot(nrows,ncols,plt_idx+1)


def set_ax(row, col, gs):
    ax = get_ax(row, col, gs)
    ax.autoscale(enable=True)
    # plt.axis('off')
    ax.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        left=False,
        labelbottom=False,
        labelleft=False
    )  # labels along the bottom edge are off
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    ax.set_aspect('auto')
    return ax


import numpy as np
N_OUT_LIMIT=20
from .helpers.util import ask_choose_one, flush_stdin
def do_limit(matrices):
    tmp_out_shape=matrices[0].shape[0]
    print("Out shape: {tmp_out_shape}")
    if tmp_out_shape > N_OUT_LIMIT:
        choices=[
            "Abort",# 0
            f"Continue with {N_OUT_LIMIT}",# 1
            f"Continue with {tmp_out_shape}",# 2
            "Continue with custom"# 3
        ]
        res_idx,_=ask_choose_one(message=f"Output bigger than N_OUT_LIMIT: ({tmp_out_shape}>{N_OUT_LIMIT}). What to do?", lst=choices)
        if res_idx==0:
            raise InterruptedError("Asked to abort")
        elif res_idx==2:
            matrices=[m[:N_OUT_LIMIT] for m in matrices]
        elif res_idx==3:
            flush_stdin()
            limit=input("Enter new out limit: ")
            if not limit.isdigit():
                raise TypeError("Not an int!")
            matrices=[m[:int(limit)] for m in matrices]
    return matrices

def set_column_titles(titles, gs):
    if titles is not None:
        titles = ensure_list(titles)
        for i, t in enumerate(titles):
            get_ax(0, i, gs).set_title(t)

def export_grids(matrices, titles=None, gs=None, cb=None, gs_args=None):
    matrices=do_limit(matrices)
    if gs_args is None:
        gs_args={}
    default_plt_kwargs = dict(cmap=plt.cm.bone)
    # Ensure that matrices is on form [(matrix, ArgsOrNone), (matrix, ArgsOrNone)]
    if isinstance(matrices, tuple) and len(matrices) == 2:
        matrices = [matrices]
    elif isinstance(matrices, list):
        for i, mtx_tupl in enumerate(matrices):
            if not isinstance(mtx_tupl, tuple) or not len(mtx_tupl) == 2:
                matrices[i] = (mtx_tupl, None)
    if gs is None:
        nr, nc = matrices[0][0].shape[0], len(matrices)
        gs = gridspec.GridSpec(nr, nc, **gs_args)
        print(nr, nc)
    else:
        nr,nc=gs.get_geometry()
    # plt.figure(figsize=(nc,nr))
    figax=make_figaxes(nr, nc)
    plt.subplots_adjust(**gs_args)
    if titles is not None:
        titles = ensure_list(titles)
        for i, t in enumerate(titles):
            get_ax(0, i, gs).set_title(t)
    for col, (images, plt_args) in enumerate(matrices):
        for row, im in enumerate(images):
            set_ax(row, col, gs)
            plt_kwargs = plt_args if plt_args is not None else {}
            plt_kwargs = merge_dicts(default_plt_kwargs, plt_kwargs)
            plt.imshow(im, **plt_kwargs)
            if callable(cb):
                cb(r=row, c=col, gs=gs)
    return gs,figax

def export_contours(contour_collections, titles=None, gs=None, cb=None, gs_args=None, main_plt_kwargs=None, figax=None, offset_col=0):
    """
    contour_collections=[[(image_mtx, PLT_ARGS),(contour_mtx, CONTOUR_ARGS),contour_mtx,...], [image_mtx,...],...]
    """
    if gs_args is None:
        gs_args={}
    default_plt_kwargs=dict(cmap=plt.cm.bone)
    if main_plt_kwargs is None:
        main_plt_kwargs={}
    main_plt_kwargs = merge_dicts(default_plt_kwargs, main_plt_kwargs)
    first_im_shape=ensure_list(contour_collections[0])[0].shape
    if first_im_shape[0] > N_OUT_LIMIT:
        raise ValueError(f"Output bigger than N_OUT_LIMIT: ({first_im_shape[0]}>{N_OUT_LIMIT})")
    if gs is None:
        nr, nc = first_im_shape[0], len(contour_collections)
        gs = gridspec.GridSpec(nr, nc, **gs_args)
        print(nr, nc)
    else:
        nr,nc=gs.get_geometry()
    if not figax:
        figax=make_figaxes(nr, nc)
    else:
        set_figaxes(*figax)
    set_column_titles(titles, gs)
    plt.subplots_adjust(**gs_args)
    for _col,_c_lst in enumerate(contour_collections):
        col=offset_col+_col
        current_collection=ensure_list(_c_lst)
        ims=current_collection[0]
        local_plt_kwargs=main_plt_kwargs
        if isinstance(ims,tuple):
            local_plt_kwargs=ims[1]
            ims=ims[0]
        for row,im in enumerate(ims):
            set_ax(row, col, gs)
            # Plot image
            plt.imshow(im, **local_plt_kwargs)
            if len(current_collection) > 1:
                for cnt in current_collection[1:]:
                    cnt_kwargs={}
                    if isinstance(cnt, tuple):
                        cnt_kwargs=cnt[1]
                        cnt=cnt[0]
                        # print(cnt_kwargs)
                    current_cont=cnt[row]
                    if "--tikz" in sys.argv:
                        current_cont=np.flipud(current_cont)
                    # Plot contours
                    try:
                        plt.contour(current_cont, levels=0, linewidths=4, **cnt_kwargs)
                    except:
                        print(current_cont.shape)
                        raise
                if callable(cb):
                    cb(r=row, c=col, gs=gs)
    return gs,export_grids


def domain_comparison_contour(X, Y, Yhat, gs, column_offset=0, colname="COLNAME"):
    nr, nc = gs.get_geometry()
    get_ax(0, 0, gs).set_title(colname)
    for i, (x, y, yhat) in enumerate(zip(X, Y, Yhat)):
        if i >= nr:
            break
        gs.update(wspace=0.01, hspace=0.01)
        set_ax(i, 0, gs)
        plt.imshow(x, cmap=plt.cm.bone)
        plt.contour(y, levels=0, colors="green")
        plt.contour(yhat, levels=0, colors="red")
        plt.show()
