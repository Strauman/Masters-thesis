from . import trainhelp as hlp
from .trainables.discriminator import Discriminator
# from .trainables.targetmap import Targetmap
# from .trainables.pretrain import Pretrain
from .trainables.targetmap import Targetmap as Targetmap
from .trainables.pretrain import Pretrain as Pretrain
from .build_graph import *
from .trainables import EarlyStopping, StopTraining

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
    disc_trainer: Discriminator = cfg.trainers.disc.trainable(DS_s=DS_s, DS_t=DS_t, settings=disc_settings, tensorboard_path=tensorboard_path, saver=savers.discriminator)
    tm_trainer: Targetmap = cfg.trainers.target.trainable(DS_s=DS_s, DS_t=DS_t, settings=target_settings, tensorboard_path=tensorboard_path, saver=savers.target_map)
    pretrainer: Pretrain = cfg.trainers.pretrain.trainable(DS_s=DS_s, DS_t=None, settings=pretrain_settings, tensorboard_path=tensorboard_path, saver=savers.source_map)

    @hlp.SessionSaver
    def session_saver(sess):
        sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
        # print("Restoring")
        combos = [
            ("Target restart", [0, 1, 4]),
            ("Continue", [0, 1, 2, 3])
        ]
        hlp.ask_restore(cfg.trainers.adversarial_saving.restore_list, savers, sess, index_combos=combos)
        yield sess
        # print("Saving")
        print("Updating trainable states")
        disc_trainer.update_state()
        tm_trainer.update_state()
        pretrainer.update_state()
        hlp.ask_save(cfg.trainers.adversarial_saving.save_list, savers, sess, failsafe=True)

    # @hlp.SessionSaver
    # def session_saver(sess):
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     print("Restoring")
    #
    #     for saver_name in cfg.trainers.adversarial_saving.restore_list:
    #         savers[saver_name].restore(sess)
    #     yield sess
    #     print("Saving")
    #     str_saver_names=",".join(cfg.trainers.pretrain.saving.save_list)
    #     if util.ask_user(f"Do you want to save {str_saver_names}?",default=True):
    #         for saver_name in cfg.trainers.pretrain.saving.save_list:
    #             savers[saver_name].do_save(sess)
    #     else:
    #         print("Not saving...")
    #     print("Updating trainable states")
    #     disc_trainer.update_state()
    #     tm_trainer.update_state()
    #     pretrainer.update_state()
    #     # savers.source_map.do_save(sess)
    #     # savers.classifier.do_save(sess)
    #     # savers.cross_src_target.do_save(sess)
    #     # savers.target_map.do_save(sess)
    #     # savers.discriminator.do_save(sess)

    with session_saver() as sess:  # pylint: disable=E1120
        import time
        disc_trainer.sess_enter(sess)
        tm_trainer.sess_enter(sess)
        pretrainer.sess_enter(sess)
        # tf.summary.FileWriter(os.path.join(tensorboard_path), sess.graph)
         # = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        # init_it_trs(sess)
        # print(sess.run([Xs,Xt],feeds_tr()))
        # xit()
        # tr_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "train"), sess.graph)
        # val_writer = tf.summary.FileWriter(os.path.join(tensorboard_path, "validation"), sess.graph)
        color_print(f"Top value is: {top_val}", style="notice")
        pretrainer.return_val = 0
        color_print("Pretrain VAL:")
        pretrainer.validate_val()
        color_print("TM_VAL:")
        tm_trainer.validate_val()
        try:
            while True:
                # if pretrainer.return_val <= 0.75:

                # else:
                    # print(f"skipping pretrain best val is {best_pretrain_val}")
                # pretrainer.train()
                # pretrain_val=pretrainer.return_val
                # if pretrain_val>best_pretrain_val:
                #     best_pretrain_val=pretrain_val
                # time.sleep(0.7)
                # print("pretrain_val:{pretrain_val}")

                disc_trainer.train()
                time.sleep(0.7)
                tm_trainer.train()
                time.sleep(0.7)
                # savers.source_map.do_save(sess)
                # savers.cross_src_target.restore(sess)
                # tm_trainer.validate_val()
                # break

                # if tm_trainer.opts.stop_val_acc < 0.85:
                #     tm_trainer.opts.stop_val_acc += delta_top_val
                # else:
                #     tm_trainer.opts.stop_val_acc = 0.85
                #
                # if disc_trainer.opts.stop_val_acc < 0.8:
                #     disc_trainer.opts.stop_val_acc += delta_top_val
                # else:
                #     disc_trainer.opts.stop_val_acc = 0.8
                #
                # if pretrainer.opts.stop_val_dice < 0.76:
                #     pretrainer.opts.stop_val_dice += delta_top_val
                # else:
                #     pretrainer.opts.stop_val_dice = 0.76

                # pretrainer.opts.stop_val_dice+=top_val
                print(f"New top value: {top_val}")
        except KeyboardInterrupt:
            print("Main KeyboardInterrupt")

        if "--test" in sys.argv:
            test_n_store(sess)
