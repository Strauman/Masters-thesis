# from .ansi_print import clearscreen
# clearscreen()
from .seeding import *

from . import trainhelp as hlp
from .graphbuilders.discriminator import Discriminator
# from .trainables.targetmap import Targetmap
# from .trainables.pretrain import Pretrain
from .trainables.trainable import Trainable
from .graphbuilders.targetmap import Targetmap as Targetmap
from .graphbuilders.pretrain import Pretrain as Pretrain
from.graphs.default import build_graph
import sys
from .helpers import trainhelp as hlp
from .helpers.cprints import color_print
DS_s,DS_t,cfg,tensorboard_path,CONFIG_NAME=build_graph()
savers=cfg._savers
with hlp.c_with("--nofile" not in sys.argv, lambda: open(cfg.config_save_file,"a")) as f:
    f.write(f"SEED: {seed_state}")
from .trainables import EarlyStopping, StopTraining
from .trainables.combotrainer import ComboTrainer
from sys import exit as xit
# from . import ansi_print
from .helpers.ansi_print import NOTIFICATIONS, run_notifier
from . import util
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
    from contextlib import contextmanager
    top_val = 0.65
    delta_top_val = 0.05
    cfg = cfg  # type: ADDA_CFG

    pretrain_settings = cfg.trainers.pretrain
    # tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999)
    target_settings = cfg.trainers.target
    disc_settings = cfg.trainers.disc
    best_pretrain_val = 0
    tm_trainer: Targetmap = cfg.trainers.target.trainable(DS_s=DS_s, DS_t=DS_t, settings=target_settings, tensorboard_path=tensorboard_path, saver=savers.target_map)
    disc_trainer: Discriminator = cfg.trainers.disc.trainable(DS_s=DS_s, DS_t=DS_t, settings=disc_settings, tensorboard_path=tensorboard_path, saver=savers.discriminator)
    pretrainer: Pretrain = cfg.trainers.pretrain.trainable(DS_s=DS_s, DS_t=None, settings=pretrain_settings, tensorboard_path=tensorboard_path, saver=savers.source_map, writeable=False)
    def write_initial_results():
        train_results=""

        AUC=tm_trainer.get_epoch_value_val(tm_trainer._auc) if hasattr(tm_trainer, "_auc") else -1
        train_results+=tm_trainer.cb_printer(do_print=False,check_stopping=False, write_out=False)
        train_results+=f"Target AUC={AUC}\n"
        train_results+=f"Best f1: {tm_trainer.best_f1()}\n"
        train_results+=disc_trainer.cb_printer(do_print=False,check_stopping=False, write_out=False)
        _,_,val_result_string=pretrainer.validate_val(return_str=True, write_out=False)
        print("INITIAL RESULTS")
        print(train_results)
        print(val_result_string)
        with hlp.c_with("--nofile" not in sys.argv, lambda: open(cfg.config_save_file, "a")) as f:
            f.write(f"\nINITIAL RESULTS:\n")
            for l in train_results.split("\n"):
                f.write(l+"\n")
            for l in val_result_string.split("\n"):
                f.write(l+"\n")
            f.write("-"*10)

    @hlp.SessionSaver
    def session_saver(sess):
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer(), Trainable.variables_initializer(Trainable.GraphKeys.INTRINSIC)])
        # print("Restoring")
        combos = [
            ("Target restart", [0, 1, 4]),
            ("Continue", [0, 1, 2, 3]),
            ("Discriminator strong", [0, 1, 2, 4])
        ]
        restore_num=util.get_valv("--restore-num",None)
        restore_file_suffix=None
        if restore_num is not None:
            restore_file_suffix=f"config-{CONFIG_NAME}-{restore_num}"
            color_print(f"RESTORING FROM CONFIG {restore_num} ({restore_file_suffix})", style="warning")
            # restore_file_suffix=suffix
        hlp.ask_restore(cfg.trainers.adversarial_saving.restore_list, savers, sess, index_combos=combos, restore_suffix=restore_file_suffix)
        pretrainer.post_restore()
        tm_trainer.post_restore()
        disc_trainer.post_restore()
        set_seeds()
        yield sess
        # print("Saving")
        # return
        do_ask=False
        do_heavy="--light" not in sys.argv
        did_intervene=False
        ask_timeout=2
        if "--light" in sys.argv:
            color_print(f"--light given. Not doing heavy stuff. To change press any key within {ask_timeout} seconds")
            did_intervene=util.anykey_timeout(ask_timeout)
        elif "--heavy" in sys.argv:
            color_print(f"--heavy given. Doing heavy stuff. To change press any key within {ask_timeout} seconds")
            did_intervene=util.anykey_timeout(ask_timeout)
        else:
            color_print(f"neither --light or --heavy given. Doing heavy stuff by default. To change press any key within {ask_timeout} seconds")
            did_intervene=util.anykey_timeout(ask_timeout)
        do_ask=did_intervene
        if do_ask:
            do_heavy=util.ask_user("Calculate heavy stuff",default=True, n_args=["--light"])
        if "--nofile" not in sys.argv:
            color_print(f"CFG_FILE:{cfg.config_save_file}", style="notice")
            copy_save_dir,cfg_save_filename=os.path.split(cfg.config_save_file)
            copy_save_file_suffix,_=os.path.splitext(cfg_save_filename)
        else:
            copy_save_file_suffix=""


        def extract_result_string(prefix=None):
            if do_heavy:
                AUC=tm_trainer.get_epoch_value_val(tm_trainer._auc) if hasattr(tm_trainer, "_auc") else -1
            else:
                AUC=-1
            clf=tm_trainer.get_epoch_value_val(tm_trainer.classification_accuracy) if (do_heavy or not hasattr(tm_trainer, "f1")) else 0
            best_f1_score=current_f1_score=-1
            if hasattr(tm_trainer, "f1"):
                best_f1_score=tm_trainer.best_f1()
                current_f1_score=tm_trainer.get_epoch_value_val(tm_trainer.f1)
            train_results=""
            train_results+=tm_trainer.cb_printer(do_print=False,check_stopping=False,write_out=False)
            train_results+=f"Target AUC={AUC}\n" if do_heavy else ""
            if hasattr(tm_trainer, "f1"):
                train_results+=f"Best f1: {best_f1_score}\n" if do_heavy else "F1: Running light"

            train_results+=disc_trainer.cb_printer(do_print=False,check_stopping=False, write_out=False)
            _,_,val_result_string=pretrainer.validate_val(return_str=True, write_out=False)
            result_summary=f"{current_f1_score},{best_f1_score},AUC:{AUC},clf:{clf}"
            verbose_result=train_results+"\n"+val_result_string
            if prefix is not None:
                prefix=f"{prefix.upper()}:\n-------\n\n"
            else:
                prefix=""
            return prefix+verbose_result,prefix+result_summary

        def result_for_paramname(param_name, tr=tm_trainer):
            color_print(f"Restoring from graph_state: {param_name}", style="warning")
            try:
                hlp.restore_graph_params(tr.graph_params[param_name])
            except KeyError as e:
                color_print(f"Error restoring {param_name}", style="danger")
                print(str(e))
                return f"{param_name}:ERROR:{str(e)}", f"{param_name}:ERROR:{str(e)}"
            tm_trainer.post_restore()
            disc_trainer.post_restore()
            if "--file-only" not in sys.argv:
                hlp.ask_save(cfg.trainers.adversarial_saving.save_list, savers, sess, failsafe=True, save_copy_suffix=copy_save_file_suffix, state=param_name)
            return extract_result_string(param_name)

        tm_trainer.save_param_state("latest")
        res_latest, s_res_latest=result_for_paramname("latest")
        print(res_latest)
        # if "--file-only" not in sys.argv:
            # hlp.ask_save(cfg.trainers.adversarial_saving.save_list, savers, sess, failsafe=True, save_copy_suffix=copy_save_file_suffix+"-latest")
        state_names=["initial"]+tm_trainer.param_state_names
        d_state_names=disc_trainer.param_state_names
        result_out=[res_latest]
        summary_out=[s_res_latest]
        for sn in state_names:
            r,s=result_for_paramname(sn, tr=tm_trainer)
            print(r)
            result_out.append(r)
            summary_out.append(s)
        for sn in d_state_names:
            r,s=result_for_paramname(sn, tr=disc_trainer)
            print(r)
            result_out.append(r)
            summary_out.append(s)

        color_print(f"CFG_FILE:{cfg.config_save_file}", style="notice")
        color_print(f"SEEDS:{seed_state}", style="danger")




        # pretrainer.update_state()
        import datetime
        now=datetime.datetime.now()
        print("FINAL:")
        with hlp.c_with("--nofile" not in sys.argv, lambda: open(cfg.config_save_file, "a")) as f:
            f.write(f"\n\nRESULTS {now}:\n")
            f.write("\n-------\n".join(result_out))
            f.write("-"*10)
        with open("/root/data/results.txt", "a") as f:
            f.write(f"\n---{now}---\n")
            f.write(f"\nCOMBO {cfg.model_save_subdir}:{cfg.config_save_file}\n RESULTS:\n")
            if "--run-name" in sys.argv:
                rn=util.get_valv("--run-name")
                f.write(f"\n{rn}\n")
            if "--label" in sys.argv:
                lbl=util.get_valv("--label")
                f.write(f"\n{lbl}\n")
            # f.write(f"\nRESULTS {now}:\n")
            f.write("\n-------\n".join(summary_out))
            f.write("-"*10)

    with session_saver() as sess:  # pylint: disable=E1120
        import time
        disc_trainer.sess_enter(sess)
        tm_trainer.sess_enter(sess)
        pretrainer.sess_enter(sess)
        # print(sess.run(generate))
        # tf.summary.FileWriter(os.path.join(tensorboard_path), sess.graph)
         # = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        # init_it_trs(sess)
        # print(sess.run([Xs,Xt],feeds_tr()))
        # xit()
        # tr_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "train"), sess.graph)
        # val_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        main_trainer=ComboTrainer(target_map=tm_trainer, discriminator=disc_trainer, combo_settings=cfg.combo_settings, sess=sess)
        color_print(f"Top value is: {top_val}", style="notice")
        color_print("Pretrain VAL:")
        pretrainer.validate_val(write_out=False)
        # print(pretrainer._auc)
        # xit()
        # AUC=pretrainer.get_epoch_value_val(pretrainer._auc)
        # color_print(f"AUC:{AUC}")
        # xit()
        color_print("TM_VAL:")
        tm_trainer.validate_val()
        # xit()
        if "--skipinit" not in sys.argv:
            write_initial_results()
        # print("F1 optimal:", tm_trainer.get_optimal_f1(labels=tm_trainer.Yt,predictions=tm_trainer.soft_predictions))
        print("Target top F1:", tm_trainer.best_f1())
        #
        # AUC=tm_trainer.get_epoch_value_val(tm_trainer._auc)
        # color_print(f"AUC:{AUC}")
        # if "--val" in sys.argv:
        #     xit()
        # if "--tst" in sys.argv:
        #     pretrainer.validate_tst(write_out=False)
        #     tm_trainer.validate_tst(write_out=False)
        #     xit()
        try:
            run_notifier(0.5)
            main_trainer.train()
        except KeyboardInterrupt:
            print("Main KeyboardInterrupt")
        # print("Optimal F1:", tm_trainer.get_optimal_f1(labels=tm_trainer.Yt,predictions=tm_trainer.probabilities))
