from .seeding import *
from . import trainhelp as hlp
import sys
from sys import exit as xit
from .graphs.default import build_graph
DS_s,DS_t,cfg,tensorboard_path,CONFIG_NAME=build_graph()
savers=cfg._savers
config_save_file=cfg.config_save_file if "--nofile" not in sys.argv else None
from . import util
from .helpers.cprints import color_print
from .trainables.trainable import Trainable
from .graphbuilders.pretrain import Pretrain
from .helpers.trainhelp import c_with
from .helpers.ansi_print import NOTIFICATIONS, run_notifier
if "--cfg" in sys.argv:
    # cfg_name=sys.argv[sys.argv.index("--cfg")+1]
    NOTIFICATIONS['config_subdir']=color_print(cfg.model_save_subdir, style="notice", as_str=True)
    NOTIFICATIONS['config_name']=color_print(cfg.name, style="notice", as_str=True)
if "--run-name" in sys.argv:
    NOTIFICATIONS['run_name']=color_print(util.get_valv("--run-name"), style="notice", as_str=True)
if hasattr(cfg, "config_save_file"):
    NOTIFICATIONS["config_name"]=color_print(os.path.basename(cfg.config_save_file), style="notice", as_str=True)
if "--label" in sys.argv:
    NOTIFICATIONS["label"]=color_print(util.get_valv("--label"), style="notice", as_str=True)
if __name__ == '__main__':
    # color_print("SAVING COMMENTED OUT", style="danger")
    cfg = cfg  # type: ADDA_CFG
    pretrain_settings = util.Dottify(cfg.trainers.pretrain)
    # print(cfg.model_save_subdir)
    @hlp.SessionSaver
    def session_saver(sess):
        sess.run(tf.group([tf.global_variables_initializer(), tf.local_variables_initializer(), Trainable.variables_initializer(Trainable.GraphKeys.INTRINSIC)]))
        combos=None
        restore_num=util.get_valv("--restore-num",None)
        restore_file_suffix=None
        if restore_num is not None:
            restore_file_suffix=f"config-{CONFIG_NAME}-{restore_num}"
            color_print(f"RESTORING FROM CONFIG {restore_num} ({restore_file_suffix})", style="warning")
        # pretrain_saver_state=util.get_valv("--prefix-name", None)
        # hlp.ask_restore(cfg.trainers.pretrain.saving.restore_list, savers, sess, index_combos=combos, restore_suffix=restore_file_suffix, restore_state=cfg.trainers.pretrain.state)
        hlp.ask_restore(cfg.trainers.pretrain.saving.restore_list, savers, sess, index_combos=combos, restore_suffix=restore_file_suffix)
        set_seeds()
        yield sess
        if "--nofile" not in sys.argv:
            color_print(f"CFG_FILE:{config_save_file}", style="notice")
            copy_save_dir,cfg_save_filename=os.path.split(config_save_file)
            copy_save_file_suffix,_=os.path.splitext(cfg_save_filename)
        else:
            copy_save_file_suffix=""
        if "--val" not in sys.argv and "--tst" not in sys.argv:
            print("Updating trainable states")
            hlp.restore_graph_params(pretrainer.graph_params["best"])
            pretrainer.post_restore()
            # color_print("SAVING COMMENTED OUT", style="danger")
            hlp.ask_save(cfg.trainers.pretrain.saving.save_list, savers, sess, failsafe=True, save_copy_suffix=copy_save_file_suffix)
        elif "--val" in sys.argv:
            return
        color_print(f"SEEDS:{seed_state}", style="danger")
        if "--nofile" not in sys.argv:
            color_print(f"CFG_FILE:{config_save_file}", style="notice")
        train_results=""
        train_results+=pretrainer.cb_printer(do_print=False,check_stopping=False)
        if hasattr(pretrainer, "auc"):
            train_results+=f"\nAUC{pretrainer.auc}"
        train_results+=f"Optimal f1: {pretrainer.best_f1()}"
        _,_,val_result_string=pretrainer.validate_val(return_str=True)
        import datetime
        now=datetime.datetime.now()

        with c_with("--nofile" not in sys.argv,lambda: open(config_save_file, "a")) as f:
            f.write(f"\nRESULTS {now}:")
            f.write(train_results)
        with c_with("--nofile" not in sys.argv,lambda: open("/root/data/results.txt", "a")) as f:
            f.write(f"---{now}---")
            f.write(f"PRETRAIN {cfg.model_save_subdir}:{config_save_file}\n RESULTS:\n")
            f.write(train_results)
            f.write(val_result_string)
            f.write("\n---\n")

    # pretrainer=Pretrain(DS_s=DS_s, DS_t=DS_t, settings=dict(stop_val_dice=0.785),tensorboard_path=tensorboard_path)
    pretrainer: Pretrain = cfg.trainers.pretrain.trainable(
        DS_s=DS_s,
        DS_t=None,
        tensorboard_path=tensorboard_path,
        settings=pretrain_settings,
        saver=savers.source_map
    )
    #pylint: disable= E1120
    with session_saver() as sess:
        pretrainer.sess_enter(sess)
        pretrainer.validate_val()
        if "--val" in sys.argv:
            # color_print(f"AUC:{pretrainer.auc}", style="notice")
            if hasattr(pretrainer, "auc"):
                print(pretrainer.auc)
            xit()

        # pretrainer.validate_tst()
        # xit()
        #pylint: enable= E1120
        # pretrainer.validate_val()
        run_notifier(0.5)
        pretrainer.train()
        # pretrainer.validate_val()
        # pretrainer.validate_tst()
        # DS_s.init_handles(sess)
        # DS_t.init_handles(sess)
# os._exit(0)
