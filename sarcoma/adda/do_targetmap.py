from . import trainhelp as hlp
from .build_graph import *  # pylint: disable=W
from . import util
from .trainables.targetmap import Targetmap
if __name__ == '__main__':
    cfg = cfg  # type: ADDA_CFG
    tm_settings = util.Dottify(cfg.trainers.target)
    @hlp.SessionSaver
    def session_saver(sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        combos = [
            ("Target restart", [0, 1, 2]),
            ("Continue", [0, 1, 3])
        ]
        hlp.ask_restore(cfg.trainers.target.saving.restore_list, savers, sess, index_combos=combos)
        yield sess
        if "--val" in sys.argv:
            return
        print("Saving")
        target_trainer.validate_val()
        target_trainer.update_state()
        hlp.ask_save(cfg.trainers.target.saving.save_list, savers, sess, failsafe=True)
    # pretrainer=Pretrain(DS_s=DS_s, DS_t=DS_t, settings=dict(stop_val_dice=0.785),tensorboard_path=tensorboard_path)
    target_trainer: Targetmap = cfg.trainers.target.trainable(
        DS_s=DS_s,
        DS_t=DS_t,
        tensorboard_path=tensorboard_path,
        settings=tm_settings,
        saver=savers.target_map
    )
    #pylint: disable= E1120
    with session_saver() as sess:
        target_trainer.sess_enter(sess)
        target_trainer.validate_val()
        if "--val" in sys.argv:
            xit()

        # pretrainer.validate_tst()
        # xit()
        #pylint: enable= E1120
        # pretrainer.validate_val()
        target_trainer.train()
        # pretrainer.validate_val()
        # pretrainer.validate_tst()
        # DS_s.init_handles(sess)
        # DS_t.init_handles(sess)
