import sys
import os
# import ..configs
from . import ADDA_CFG, util, TR_CFG, Trainer_CFG, Saving_CFG, CFG_Models
# from ..trainables.pre
from ..trainables.pretrain import Pretrain
from ..trainables.discriminator import Discriminator
from ..trainables.targetmap import Targetmap

import tensorflow as tf


def pretrain_optim(self: Pretrain):
    return tf.train.AdamOptimizer()


def discriminator_optim(self: Discriminator):
    # return tf.train.MomentumOptimizer(0.01, 0.999, use_locking=False, name='DiscOptim', use_nesterov=True)
    # tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    # return tf.contrib.opt.NadamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, name='Adam')
    return tf.contrib.opt.NadamOptimizer(learning_rate=0.001, beta1=0.8, beta2=0.999, epsilon=1e-01, name='Adam')
    # return tf.train.AdamOptimizer(learning_rate=0.005, beta1=0.9, beta2=0.999)
    # return tf.train.MomentumOptimizer(0.001, 0.999, use_locking=False, name='Momentum', use_nesterov=True)


def target_optim(self: Targetmap):
    return tf.contrib.opt.NadamOptimizer(learning_rate=0.001, beta1=0.8, beta2=0.999, epsilon=1e-01, name='Adam')
    # return tf.train.MomentumOptimizer(0.001, 0.999, use_locking=False, name='Momentum', use_nesterov=True)


pretrain_saver = Saving_CFG(
    restore_list=[
        "classifier",
        "source_map"
    ],
    save_list=[
        "classifier",
        "source_map"
    ]
)
target_saver = Saving_CFG()
discriminator_saver = Saving_CFG()
adversarial_saving = Saving_CFG(
    restore_list=[
        "classifier",
        "source_map",
        # ----
        # "discriminator",
        # -----
        # "target_map",
        "cross_src_target"
        # ----
    ],
    save_list=[
        # "classifier",
        "target_map",
        # "source_map",
        "discriminator",
        # "cross_src_target"
    ]
)
from ..tfmodels import unet_whole as unet
MAIN_CFG = ADDA_CFG(  # type: ADDA_CFG
    source_map=util.Dottify(depth=2, filters=8),
    target_map=util.Dottify(depth=2, filters=8),
    classifier=util.Dottify(),
    models=CFG_Models(
        classifier=unet.classifier,
        discriminator=unet.dense_discriminator,
        target_map=unet.encoder_model,
        source_map=unet.encoder_model
    ),
    discriminator=util.Dottify(),
    source_dataset="MNIST_SEG",
    target_dataset="SVHN_LBL_28",
    input_shape=(28, 28),
    hidden_shape=(28, 28),
    model_save_subdir="USPS",
    trcfg=TR_CFG(
        source_batch=20,
        target_batch=20,
        print_summaries=False
    ),

    trainers=Trainer_CFG(
        pretrain=util.Dottify(
            trainable=Pretrain,
            optimizer=pretrain_optim,
            stop_val_acc=0.87,
            max_epochs=1000000,
            scope_re="^(classifier|source_map)",
            saving=pretrain_saver,
            num_classes=10
            # scope_re="^(source_map)",
            # scope_re="^(classifier)",
        ),
        target=util.Dottify(
            trainable=Targetmap,
            optimizer=target_optim,
            stop_val_acc=1.,
            max_epochs=2,
            min_steps=0,
            has_segment_labels=False,
            scope_re="target_map",
            saving=target_saver,
            reset_optimizers=False,
            early_stopping=util.Dottify(
                loss_difference_threshold=None,
                loss_difference_memory=None,
                percentage_total_decrease=None#0.36
            )
            # early_stopping=util.Dottify(
            #     loss_difference_threshold=9E-6,
            #     loss_difference_memory=100,
            #     percentage_total_decrease=None#0.36
            # )
        ),
        disc=util.Dottify(
            trainable=Discriminator,
            optimizer=discriminator_optim,
            stop_val_acc=1.,
            max_epochs=2,
            min_steps=0,
            reset_optimizers=False,
            saving=discriminator_saver,
            scope_re="^((?!source_map|target_map|classifier).)*/",
            # scope_re="discriminator",
            early_stopping=util.Dottify(
                loss_difference_threshold=None,
                loss_difference_memory=None,
                percentage_total_decrease=None
            )
            # early_stopping=util.Dottify(
            #     loss_difference_threshold=9E-6,
            #     loss_difference_memory=100,
            #     percentage_total_decrease=None
            # )
        ),
        adversarial_saving=adversarial_saving
    )
)
