import sys
import os
# import ..configs
from . import ADDA_CFG, util, TR_CFG, Trainer_CFG
# from ..trainables.pre
from ..trainables.pretrain import Pretrain
from ..trainables.discriminator import Discriminator
from ..trainables.targetmap import Targetmap
import tensorflow as tf


def pretrain_optim(self: Pretrain):
    return tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step, var_list=self.train_vars)

def discriminator_optim(self: Discriminator):
    return tf.train.MomentumOptimizer(0.01,0.999,use_locking=False,name='DiscOptim',use_nesterov=True)

def target_optim(self: Targetmap):
    return tf.train.MomentumOptimizer(0.05,0.999,use_locking=False,name='Momentum',use_nesterov=True)

MAIN_CFG = ADDA_CFG(  # type: ADDA_CFG
    source_map=util.Dottify(depth=2, filters=4),
    target_map=util.Dottify(depth=2, filters=4),
    classifier=util.Dottify(),
    discriminator=util.Dottify(),
    source_dataset="MNIST_SEG_16",
    target_dataset="USPS_SEG",
    trcfg=TR_CFG(source_batch=20, target_batch=20),
    input_shape=(16, 16),
    hidden_shape=(16, 16),
    model_save_subdir="USPS",
    trainers=Trainer_CFG(
        pretrain=util.Dottify(
            optimizer=pretrain_optim,
            stop_val_acc=0.96,
            scope_re="^(classifier|source_map)",
            # scope_re="^(source_map)",
            # scope_re="^(classifier)",
        ),
        target=util.Dottify(optimizer=target_optim, stop_val_acc=0.8, max_epochs=10000),
        disc=util.Dottify(optimizer=discriminator_optim, stop_val_acc=0.6)
    )
)
