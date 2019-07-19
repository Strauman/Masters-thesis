from . import trainhelp as hlp
from .build_graph import *  # pylint: disable=W
from . import util
from .trainables.discriminator import Discriminator
import sys
if __name__ == '__main__':
    cfg = cfg  # type: ADDA_CFG
    disc_settings = cfg.trainers.disc
    @hlp.SessionSaver
    def session_saver(sess):
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        combos=None
        do_ask="--restoreall" not in sys.argv
        hlp.ask_restore(cfg.trainers.disc.saving.restore_list, savers, sess, index_combos=combos, do_ask=do_ask)
        yield sess
        print("Saving")
        disc_trainer.update_state()
        hlp.ask_save(cfg.trainers.disc.saving.save_list, savers, sess, failsafe=True)
    # pretrainer=Pretrain(DS_s=DS_s, DS_t=DS_t, settings=dict(stop_val_dice=0.785),tensorboard_path=tensorboard_path)
    disc_trainer: Discriminator = cfg.trainers.disc.trainable(
        DS_s=DS_s,
        DS_t=DS_t,
        tensorboard_path=tensorboard_path,
        settings=disc_settings,
        saver=savers.discriminator
    )
    #pylint: disable= E1120
    with session_saver() as sess:
        disc_trainer.sess_enter(sess)
        disc_trainer.train()
