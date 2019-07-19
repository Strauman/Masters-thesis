from .seeding import *

from . import trainhelp as hlp
# from .trainables.targetmap import Targetmap
# from .trainables.pretrain import Pretrain
from .trainables.trainable import Trainable
from .graphbuilders.discriminator import Discriminator
from .graphbuilders.targetmap import Targetmap as Targetmap
from .graphbuilders.pretrain import Pretrain as Pretrain
from . import exporthead as export
from .trainables import EarlyStopping, StopTraining
from .trainables.combotrainer import ComboTrainer
from sys import exit as xit
# from . import ansi_print
from .helpers.ansi_print import NOTIFICATIONS, run_notifier
from . import util
from pprint import pprint as pp
cfg=tm_trainer=disc_trainer=pretrainer=None
States=export.States



def xor(a,b):
    return (a or b) and not (a and b)

restore_list=export.EXPORT_RESTORE_LIST


def load_config(c=None,s=None,n=None,cfgn=None, state=None, num=None, restore_list=restore_list):
    # c,s and n are aliases to the preceding args
    global cfg, tm_trainer, disc_trainer, pretrainer, trainers
    cfgn=cfgn or c
    state=state or s
    num=num or n
    c,s,n=cfgn,state,num
    # if (not n and s) and (n or s):
    #     raise util.ArgumentError("cfgn/c argument has to be given to load_config on first call")
    # if fav in LOADED_FAVS.keys():
    #     cfg, tm_trainer, disc_trainer, pretrainer=LOADED_FAVS[fav]
    if not cfgn and not cfg:
        raise util.ArgumentError("cfgn/c argument has to be given to load_config on first call")
    if cfgn and not cfg or cfg.name!=cfgn:
        cfg, tm_trainer, disc_trainer, pretrainer = export.load_config(cfgn)
        # LOADED_FAVS[fav]=(cfg, tm_trainer, disc_trainer, pretrainer)

    trainers = [tm_trainer, disc_trainer, pretrainer]
    get_state_for_num=num or cfg.run_num
    get_state_for_state=state or cfg.state
    # if xor(num, state):# Only one is given
    #     if num:
    #
    #         export.get_state(cfg.state, cfg, n, tm_trainer, disc_trainer, pretrainer, restore_list=restore_list)
    #         cfg.state=state
    #     if state:
    #         export.get_state(cfg.state, cfg, cfg.run_num, tm_trainer, disc_trainer, pretrainer, restore_list=restore_list)
    #         cfg.run_num=num
    # elif num and state:
    export.get_state(get_state_for_state, get_state_for_num, cfg, *trainers, restore_list=restore_list)
    return cfg, tm_trainer, disc_trainer, pretrainer


    # if state!=cfg.state or num!=cfg.run_num:
        # export.get_state(state, cfg, num, *trainers)



# restore_num = util.get_valv("--restore-num", None)
# restore_file_suffix = None
# sess=tf.Session()
# sess.as_default()
# restore_list=["source_map"]
# cfg, tm_trainer, disc_trainer, pretrainer=load_config("CT_PT_ADDA", States.best, 110)
