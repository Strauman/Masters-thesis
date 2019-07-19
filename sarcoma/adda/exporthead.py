from .seeding import *
from . import trainhelp as hlp
from .graphs.default import build_graph
from .helpers.cprints import color_print
from sys import exit as xit
from .trainables import Trainable
from . import util
from .configs.configs import get_config as _get_config
from .configs import ADDA_CFG
import sys
best_pretrain_val = 0
EXPORT_RESTORE_LIST=["classifier","source_map", "target_map", "discriminator"]

class States():
    conf = "abs_confusion"
    best = "best_perf"
    loss = "best_loss"
    initial = init = "initial"
    latest = "latest"

def restore_suffix_for_cfg_num(cfg_name=None, num=None):
    return f"config-{cfg_name}-{num}"

from typing import Tuple
def load_config(cfg_name, restore_list=None, set_cfg_states=True, retlist_also=False, retlist_only=False) -> Tuple[ADDA_CFG, Trainable, Trainable, Trainable]:
    restore_list=restore_list or EXPORT_RESTORE_LIST
    # global cfg, model_save_root,tm_trainer,disc_trainer,pretrainer
    # global DS_s,DS_t,cfg,tensorboard_path,CONFIG_NAME,models_dir,sess,pretrain_settings,target_settings,disc_settings
    if tf.get_default_session():
        tf.get_default_session().close()
    tf.reset_default_graph()
    # return cfg
    new_cfg,new_cfg_name=_get_config(cfg_name)
    # if cfg is None:
    cfg=new_cfg
    DS_s,DS_t,cfg,tensorboard_path,_=build_graph(new_cfg, do_shuffle=None)
    models_dir=new_cfg.models_dir
    cfg.model_save_root = os.path.join(models_dir, new_cfg.model_save_subdir)
    color_print(f"Restoring config {new_cfg_name}")
    # new_cfg._savers=util.Dottify(#pylint: disable=W0212
    # old_cfg=cfg
    cfg=new_cfg
    pretrain_settings = cfg.trainers.pretrain
    target_settings = cfg.trainers.target
    disc_settings = cfg.trainers.disc
    cfg._savers = hlp.savers_from_cfg(cfg)
    tm_trainer = cfg.trainers.target.trainable(DS_s=DS_s, DS_t=DS_t, settings=target_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.target_map,writeable=False)
    disc_trainer = cfg.trainers.disc.trainable(DS_s=DS_s, DS_t=DS_t, settings=disc_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.discriminator,writeable=False)
    pretrainer = cfg.trainers.pretrain.trainable(DS_s=DS_s, DS_t=None, settings=pretrain_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.source_map, writeable=False)
    # cfg=new_cfg
    init_session(tm_trainer,disc_trainer,pretrainer)
    if set_cfg_states:
        cfg._stateinfo=dict(tm_trainer=tm_trainer, disc_trainer=disc_trainer, pretrainer=pretrainer)
    retlist=[cfg, tm_trainer, disc_trainer, pretrainer]
    if retlist_also:
        return [*retlist, retlist]
    else:
        return retlist
def reset_states(disc_trainer,tm_trainer,pretrainer):
    sess=tf.get_default_session()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), Trainable.variables_initializer(Trainable.GraphKeys.INTRINSIC)])
    disc_trainer.sess_enter(sess)
    tm_trainer.sess_enter(sess)
    pretrainer.sess_enter(sess)

def init_session(tm_trainer,disc_trainer,pretrainer):
    sess=tf.InteractiveSession()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), Trainable.variables_initializer(Trainable.GraphKeys.INTRINSIC)])
    disc_trainer.sess_enter(sess)
    tm_trainer.sess_enter(sess)
    pretrainer.sess_enter(sess)

def _get_state(state, num, cfg, tm_trainer,disc_trainer,pretrainer,restore_list=None):
    # combos = [
    #     ("Target restart", [0, 1, 4]),
    #     ("Continue", [0, 1, 2, 3])
    # ]
    restore_list=restore_list or EXPORT_RESTORE_LIST
    print(restore_list)
    num = num or util.get_valv("--restore-num")
    suff = restore_suffix_for_cfg_num(cfg_name=cfg.name, num=num)
    for restore_name in restore_list:
        curr_state=state
        curr_suff=suff
        saver=cfg.savers[restore_name]# type: SAVOBJ
        saver.restore(sess=tf.get_default_session(), restore_dir=None, restore_state=curr_state, restore_suffix=curr_suff, try_list=None, verbose=1)
    # hlp.ask_restore(EXPORT_RESTORE_LIST, savers, tf.get_default_session(), index_combos=combos, restore_suffix=suff, restore_state=state)
    pretrainer.post_restore()
    tm_trainer.post_restore()
    disc_trainer.post_restore()

def get_state(state, num, cfg, tm_trainer=None,disc_trainer=None,pretrainer=None,restore_list=None):
    if tm_trainer is None and disc_trainer is None and pretrainer is None:
        if hasattr(cfg, "_stateinfo"):
            #pylint: disable=w0212
            tm_trainer=cfg._stateinfo["tm_trainer"]
            disc_trainer=cfg._stateinfo["disc_trainer"]
            pretrainer=cfg._stateinfo["pretrainer"]
        else:
            raise TypeError("Cannot have all trainers None when no state info found in cfg")
    return _get_state(state, num, cfg, tm_trainer, disc_trainer, pretrainer)

def get_segmentation_state(cfg, run_num=None):
    restore_list=["source_map", "classifier"]
    suff=None
    restore_args={}
    if run_num:
        suff=restore_suffix_for_cfg_num(cfg_name=cfg.name, num=run_num)
        restore_args=dict(restore_suffix=suff)
    for restore_name in restore_list:
        cfg.savers[restore_name].restore(sess=tf.get_default_session(),**restore_args)

def intersect(a,b):
    return list(set(a).intersection(set(b)))

def confirm_recache():
    args_needing_confirm=["--recache", "--clearcache", "--force-clearcache"]
    cacheargs=intersect(args_needing_confirm, sys.argv)
    if cacheargs:
        cacheargs_str=" and ".join(cacheargs)
        color_print(f"{cacheargs_str} given")
        util.ask_user(f"You sure you want to continue with {cacheargs_str}?", default=False, abort=True)
