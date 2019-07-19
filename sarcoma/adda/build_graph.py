import sys
import os
import tensorflow as tf
from . import datasets, util, metrics
from sys import exit as xit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from .train import train
from pprint import pprint as pp
from . import color_print, warning_print
import time
from .helpers import trainhelp as hlp
StopTraining = hlp.StopTraining
import types
from typing import Iterable
from .helpers.trainhelp import new_saver, new_cross_saver, uniquify_filename

from .configs import ADDA_CFG
import numpy as np
# Config setup
# from .configs.PT_MRT2 import MAIN_CFG as cfg
from .configs.configs import ask_config
cfg, CONFIG_NAME = ask_config()
# from.configs.MNIST_SVHN_LBL import MAIN_CFG as cfg
# from .configs.MNIST_USPS_SEG import MAIN_CFG as cfg
# from .configs.MNIST_SVHN import MAIN_CFG as cfg
cfg = cfg  # type: ADDA_CFG
# cfg.summary()
# xit()

DO_SUMMARY = "--summary" in sys.argv
# Model setup
if "--discard" in sys.argv:
    color_print("--discard given -- not saving any models", style="danger")
if "--no-restore" in sys.argv:
    color_print("--no-restore -- not restoring any models", style="danger")
if "--nofile" in sys.argv:
    color_print("NOT SAVING ANY CONFIG FILES (--nofile given)", style="danger")
random_test=tf.random.uniform(
    (),
    minval=0,
    maxval=10,
    dtype=tf.dtypes.int32,
    seed=None,
    name="random_test"
)
source_map = cfg.models.source_map
target_map = cfg.models.target_map
classifier = cfg.models.classifier
discriminator = cfg.models.discriminator

source_map_config = cfg.source_map
target_map_config = cfg.target_map
# target_map_config = util.Dottify(depth=5, filters=8)
classifier_config = cfg.classifier
discriminator_config = cfg.discriminator
is_training = tf.placeholder(tf.bool, name="is_training")
runned_file = sys.argv[0]
os.path.relpath(runned_file, __file__)
deltat = time.time() - os.path.getmtime(os.path.realpath(runned_file))
color_print("{BLUEBG}{WHITE}{filename} modified delta: {time}{ENDC}", time=deltat, filename=runned_file)
color_print("{BLUEBG}{WHITE}Model: {modelname}{ENDC}", modelname=cfg.model_save_subdir)
current_file_path = os.path.realpath('__file__')
# color_print(f"realpath('__file__'):{current_file_path}")
# color_print(f"f_code: {sys._getframe().f_code.co_filename}")
# color_print(f"argv file:{}")

if "ON_SERVER" in os.environ:
    color_print("ON SERVER", style="notice")
else:
    color_print("IS LOCAL (no ON_SERVER in environ)", style="notice")
tensorboard_path = "/root/data/current_graph/"
# DS_s = datasets.get_dataset("PT_ONLY_128")
# DS_t = datasets.get_dataset("T2_ONLY_128")
DS_s = datasets.get_dataset(cfg.source_dataset)
DS_t = datasets.get_dataset(cfg.target_dataset)
SUMMARY=DO_SUMMARY
if "--delete-tb" in sys.argv and "ON_SERVER" in os.environ:
    from shutil import rmtree
    if "--run-name" in sys.argv:
        run_name=sys.argv[sys.argv.index("--run-name")+1]
        color_print("{FATALBG}{WHITE}--delete-tb and --run-name given: Deleting specific tensorboard files:{ENDC}")
        dir2del=[os.path.join(tensorboard_path,"validation"+run_name), os.path.join(tensorboard_path,"train"+run_name)]
        print("\n".join(dir2del))
        for d in dir2del:
            if os.path.isdir(d):
                rmtree(d)
    else:
        color_print("{FATALBG}{WHITE}--delete-tb given: Deleting tensorboard files{ENDC}")
        rmtree(tensorboard_path)
        os.mkdir(tensorboard_path)
    time.sleep(0.5)
s_bsize = cfg.trcfg.source_batch
t_bsize = cfg.trcfg.target_batch
# val_s_batch_size=1000
val_s_batch_size=s_bsize
val_t_batch_size=val_s_batch_size
# DS_s = datasets.get_dataset("MNIST_SEG")
DS_s.ds_send_tr(batch=s_bsize)
DS_s.ds_send_tr(shuffle=500)
DS_s.ds_send_val(batch=val_s_batch_size)
# DS_s.ds_send_tr(("prefetch", dict(buffer_size=100)))
# DS_s.ds_send_tst("repeat")
# DS_s.ds_send_tst(batch=1000)
source_iter, source_handle = DS_s.make_iterators()
#
DS_t.ds_send_tr(batch=t_bsize)
DS_t.ds_send_tr(shuffle=500)
DS_t.ds_send_val(batch=val_t_batch_size)
# DS_t.ds_send_tr(("prefetch", dict(buffer_size=100)))
target_iter, target_handle = DS_t.make_iterators()
#
def name_tensors(tensors: Iterable, names: Iterable[str]) -> list:
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    if not isinstance(names, (list, tuple)):
        names = [names]
    return_tensors = []
    for t, n in zip(tensors, names):
        return_tensors.append(tf.identity(t, name=n))
    return return_tensors


Xt, Yt = target_iter.get_next()
Xs, Ys = source_iter.get_next()
#pylint: disable=E0632
Xs, Ys = name_tensors([Xs, Ys], ["Xs", "Ys"])
Xt, Yt = name_tensors([Xt, Yt], ["Xt", "Yt"])
#pylint: enable=E0632
# Xs=tf.placeholder(tf.float32, (None,128,128))
# Ys=tf.placeholder(tf.int8, (None,128,128))


#------ Persistent savable variables ------#
global_step = tf.Variable(0, name='global_step', trainable=False)
epoch_step = tf.Variable(0, name='epoch_step', trainable=False)


# from . import tfmodels
# if "ON_SERVER" in os.environ:
#     deltat = time.time() - os.path.getmtime(os.path.realpath(tfmodels.unet_2nd.__file__))
#     color_print("{BLUEBG}{WHITE}unet_2nd modified delta: {time}{ENDC}", time=deltat)

models_dir="/root/data/models/"
model_save_root = os.path.join(models_dir, cfg.model_save_subdir)

dX_shape = cfg.input_shape

with tf.variable_scope("source_map"):
    enc_s = source_map(Xs, cfg, role_cfg=cfg.source_map, is_training=is_training)
    Zs = enc_s(Xs)
    Zs_in = Input(tensor=Zs, name="Z_source_input")
    # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, enc_s.get_updates_for(Xs))
Zs_named=tf.identity(Zs, "Zs")
with tf.variable_scope("target_map"):
    enc_t = target_map(Xt, cfg, role_cfg=cfg.target_map, is_training=is_training)
    Zt = enc_t(Xt)
    # Zt=tf.identity(Zt, "Zt")
    # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, enc_t.get_updates_for(Xt))
    Zt_in = Input(tensor=Zt, name="Z_target_input")
    # enc_t=enc_s
    # Zt = enc_t([Xt])
with tf.variable_scope("hidden"):
    DummyX = Input(shape=dX_shape, name="dummyX_input")
    hs = enc_s(DummyX)

# print(hs)

cfg.hidden_shape=hs.shape[1:]
dZ_shape = cfg.hidden_shape

DummyZ = Input(shape=dZ_shape, name="dummyZ_input")
with tf.variable_scope("classifier"):
    clf = classifier(DummyZ, cfg, is_training=is_training)

with tf.variable_scope("discriminator"):
    disc = discriminator(DummyZ, cfg, is_training=is_training)

# ----- PreTraining step

Xs_in = Input(tensor=Xs, name="X_source_input")
Yhat_mod = Model(inputs=[Xs_in], outputs=[clf(enc_s(Xs_in))], name="m_Yhat")
# print(Yhat_mod.get_updates())
# xit()
if cfg.trcfg.print_summaries:
    print("Summary of Yhat model:")
    Yhat_mod.summary()
Yhat = tf.identity(Yhat_mod.output, "Yhat")

# pp(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
# ----- Discriminator step
# D_Zt_in=Input(tensor=tf.keras.backend.stop_gradient(Zt))
# D_Zs_in= Input(tensor=tf.keras.backend.stop_gradient(Zs))

D_Zt_in = Zt_in
D_Zs_in = Zs_in
# D_Zt_in=Input(tensor=Zt, name="Z_target_input_D")
# D_Zs_in=Input(tensor=Zs, name="Z_source_input_D")
# with tf.variable_scope("discriminator"):

# Choose random Z:
# rand=tf.random.uniform(())
# Z_to_train,disc_labels=tf.cond(tf.greater(rand,.5), lambda: (D_Zt_in,tf.ones(tf.shape(Zt)[0])), lambda: (D_Zs_in,tf.zeros(tf.shape(Zs)[0])))
# Z_to_train=Input(tensor=Z_to_train)
# disc_labels=name_tensors(disc_labels,"disc_labels")
# Dhat_model=Model(inputs=[Z_to_train], outputs=[disc(Z_to_train)])

# Concatenate Z:
disc_labels = tf.concat([tf.zeros(tf.shape(Zt)[0]), tf.ones(tf.shape(Zs)[0])], axis=0, name="disc_labels")
# disc_inputs = tf.concat([D_Zt_in, D_Zs_in], axis=0)
d_mod = tf.keras.layers.concatenate([D_Zt_in, D_Zs_in], axis=0, name="disc_concat")
# XY_id=tf.random.uniform((),minval=0,maxval=tf.shape(Zt)[0],dtype=tf.int32)
# XY_id=tf.random.shuffle(XY_id)
# disc_labels=disc_labels[XY_id]
# d_mod=d_mod[XY_id]

disc_labels = name_tensors(disc_labels, "disc_labels")
Dhat_model = Model(inputs=[D_Zt_in, D_Zs_in], outputs=[disc(d_mod)], name="m_Dhat")
if cfg.trcfg.print_summaries:
    print("Summary of Dhat model:")
    Dhat_model.summary()

Dhat = Dhat_model.output
# Dhat=Dhat_model([Zt,Zs])
Dhat = name_tensors(Dhat, "Dhat")
# ----- Target map step
# Zt_in = Input(tensor=Zt, name="Z_target_input_tensor")

tm_disc_labels = tf.ones(tf.shape(Zt)[0], name="tm_disc_labels")
Dhat_t_model = Model(inputs=[Zt_in], outputs=[disc(Zt_in)], name="m_Dhat_t")
if cfg.trcfg.print_summaries:
    print("Summary of Dhat_targset model:")
    Dhat_t_model.summary()

Dhat_t = Dhat_t_model.output
Dhat_t = name_tensors(Dhat_t, "Dhat_t")

## --- Classifiers
Xt_in = Input(tensor=Xt, name="X_target_input_tensor")
# YhatS_mod=Model(inputs=[Xs_in], outputs=[clf(enc_s(Xs_in))])
YhatT_mod = Model(inputs=[Xt_in], outputs=[clf(enc_t(Xt_in))], name="m_Yhat_t")
if cfg.trcfg.print_summaries:
    print("Summary of Yhat_target model:")
    YhatT_mod.summary()

YhatT = YhatT_mod.output
YhatT = name_tensors(YhatT, "YhatT")
# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


# Make savers
_savers = util.Dottify(
    classifier=new_saver("classifier", models_dir, save_dir_name=cfg.source_dataset, restore_dir_name=cfg.source_dataset, label=cfg.trainers.pretrain.state),
    source_map=new_saver("source_map", models_dir, save_dir_name=cfg.source_dataset, restore_dir_name=cfg.source_dataset, label=cfg.trainers.pretrain.state),
    target_map=new_saver("target_map", model_save_root, save_dir_name=cfg.target_dataset, restore_dir_name=cfg.target_dataset),
    discriminator=new_saver("discriminator", model_save_root),
    cross_src_target=new_cross_saver(name="cross_src_target", save_root=models_dir, source_scope="source_map", dest_scope="target_map", save_dir_name=cfg.target_dataset, restore_dir_name=cfg.source_dataset, label=cfg.trainers.pretrain.state)
)
# savers=_savers
# cfg.savers=savers
cfg.savers = _savers
savers = cfg.savers
import shutil
sh_columns, sh_rows = shutil.get_terminal_size(fallback=(80, 24))
default_positions = [.33, .55, .67, 1., 1.]



def print_mod_summary(mod: Model):
    color_print(f"{mod.name}", style="notice")
    mod.summary(line_length=int(sh_columns * 0.8), positions=None)

def get_summaries(summary_func=print_mod_summary, include_sanity=True, header_print_func=print):
    print(sh_columns, sh_rows)

        # mod.summary(line_length=int(sh_columns * 0.8), positions=default_positions)
    print(__name__)
    header_print_func("Source map")
    summary_func(enc_s)
    if not (source_map is target_map):
        header_print_func("Target map ≠ source map")
        header_print_func("Target map:")
        summary_func(enc_t)
    else:
        header_print_func("Target map = source map")
    header_print_func("Classifier")
    summary_func(clf)
    header_print_func("Discriminator")
    summary_func(disc)
    if include_sanity:
        header_print_func("Sanity model checks:")
        header_print_func("Classification outputs:")
        summary_func(Yhat_mod)
        summary_func(YhatT_mod)
        header_print_func("Discriminator outputs:")
        summary_func(Dhat_model)
        summary_func(Dhat_t_model)


summary_out_str=""
def accumulate_summary_out(mod):
    global summary_out_str
    summary_out_str+=mod.to_json()

def _pretty_acc_func(*args,**kwargs):
    global summary_out_str
    summary_out_str+="\t".join(args)+"\n"

def pr_accumulate_summary_out(mod):
    mod.summary(line_length=int(sh_columns * 0.8), positions=default_positions, print_fn=_pretty_acc_func)

def accumulate_summary_header(head):
    global summary_out_str
    decor_char="-"
    n_decor=4
    filldecor=decor_char*len(head)
    decor_leftright=decor_char*n_decor
    decor_line=f"\n{decor_leftright}{filldecor}{decor_leftright}\n"
    decor_str=f"{decor_leftright}{head}{decor_leftright}"
    summary_out_str+=f"{decor_line}{decor_str}{decor_line}"


def summarize(do_print=False):
    cfg_sum=cfg.summary(do_print=False)
    # get_summaries(summary_func=accumulate_summary_out, include_sanity=False, header_print_func=accumulate_summary_header)
    get_summaries(summary_func=pr_accumulate_summary_out, include_sanity=False, header_print_func=accumulate_summary_header)
    mod_sum=summary_out_str
    # print(cfg_sum)
    str_out="\n".join([f"Script ran: {runned_file}",
            "CONFIGS:\n",
            cfg_sum,
            "MODELS",
            mod_sum])
    save_to=os.path.join(model_save_root,f"config-{CONFIG_NAME}.txt")
    save_to_latest=os.path.join(model_save_root,"config_latest.txt")
    save_to=uniquify_filename(save_to, sep="-")
    with open(save_to_latest, "w") as f:
        f.write(str_out)
    if not do_print and "ON_SERVER" in os.environ and "--nofile" not in sys.argv:
        color_print(f"Saving config copy to {save_to}", style="notice")
        with open(save_to, "w") as f:
            f.write(f"Copy of: {save_to}")
            f.write(str_out)
    else:
        print(str_out)
    cfg.config_save_file=save_to
    return save_to



prevent_summarize_args=set(["--discard", "--delcfg", "--nofile"]).intersection(set(sys.argv))
if __name__ == '__main__':
    config_save_file=summarize(do_print=True)
elif "--summary" in sys.argv:
    summarize(do_print=True)
    xit()
elif not prevent_summarize_args:
    config_save_file=summarize()
else:
    config_save_file=None
    color_print(f"Not summarizing: {str(list(prevent_summarize_args))} given", style="warning")



# if __name__ == '__main__':
    # get_summaries(header_print_func=lambda c: color_print(c, style="notice"))
    # summarize()
    # xit()
# ------ Saving logic ------#
# saver = tf.train.Saver()
# Zs=source_map(Xs)

# Z_s = source_map.build(Xs)
# Z_t = target_map.build(Xt)
# with tf.variable_scope("classifier") as scope:
#     Yhat_s=classifier.build(Z_s)
#     scope.reuse_variables()
# # with tf.variable_scope("classifier", reuse=True):
#     Yhat_t=classifier.build(Z_t)
#
# Z=tf.concatenate([Z_s,Z_t], axis=0)
# Y_d=tf.concatenate([tf.ones(tf.shape(Z_s)[0]),tf.zeros(tf.shape(Z_t)[0])])


# with tf.variable_scope("discriminator") as scope:

# D_s=discriminator.build(Z_s)
# scope.reuse_variables()
# with tf.variable_scope("discriminator", reuse=True):
# D_t=discriminator.build(Z_t)


# for ever:


# #------ RESTORING ------#
# if os.path.isfile(cfg.checkpoint_dir + ".index"):
#     saver.restore(sess, checkpoint_dir)
#     iter, epoch = sess.run([global_step, epoch_step])
#     print("Restored at it: {}, epoch: {}".format(iter, epoch))
