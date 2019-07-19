from .seeding import *

from . import trainhelp as hlp
from .trainables.discriminator import Discriminator
# from .trainables.targetmap import Targetmap
# from .trainables.pretrain import Pretrain
from .trainables.trainable import Trainable
from .trainables.targetmap import Targetmap as Targetmap
from .trainables.pretrain import Pretrain as Pretrain
from .build_graph import *
with hlp.c_with("--nofile" not in sys.argv, lambda: open(config_save_file,"a")) as f:
    f.write(f"SEED: {seed_state}")
from .trainables import EarlyStopping, StopTraining
from .trainables.combotrainer import ComboTrainer
from sys import exit as xit
# from . import ansi_print
from .helpers.ansi_print import NOTIFICATIONS, run_notifier
from . import util

pretrain_settings = cfg.trainers.pretrain
# tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999)
target_settings = cfg.trainers.target
disc_settings = cfg.trainers.disc
best_pretrain_val = 0
tm_trainer: Targetmap = cfg.trainers.target.trainable(DS_s=DS_s, DS_t=DS_t, settings=target_settings, tensorboard_path=tensorboard_path, saver=savers.target_map)
disc_trainer: Discriminator = cfg.trainers.disc.trainable(DS_s=DS_s, DS_t=DS_t, settings=disc_settings, tensorboard_path=tensorboard_path, saver=savers.discriminator)
pretrainer: Pretrain = cfg.trainers.pretrain.trainable(DS_s=DS_s, DS_t=None, settings=pretrain_settings, tensorboard_path=tensorboard_path, saver=savers.source_map, writeable=False)

restore_num=util.get_valv("--restore-num",None)
restore_file_suffix=None
if restore_num is not None:
    restore_file_suffix=f"config-{CONFIG_NAME}-{restore_num}"
    color_print(f"RESTORING FROM CONFIG {restore_num} ({restore_file_suffix})", style="warning")
print(restore_file_suffix)
def get_state(state):
    combos = [
        ("Target restart", [0, 1, 4]),
        ("Continue", [0, 1, 2, 3])
    ]
    print(cfg.trainers.adversarial_saving.restore_list)
    hlp.ask_restore(cfg.trainers.adversarial_saving.restore_list, savers, tf.get_default_session(), index_combos=combos, restore_suffix=restore_file_suffix, restore_state=state)
    pretrainer.post_restore()
    tm_trainer.post_restore()
    disc_trainer.post_restore()

def it_restore_state(states):
    def _it():
        for s in states:
            yield get_state(s)
# sess=tf.Session()
# sess.as_default()
sess=tf.InteractiveSession()
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
disc_trainer.sess_enter(sess)
tm_trainer.sess_enter(sess)
pretrainer.sess_enter(sess)
