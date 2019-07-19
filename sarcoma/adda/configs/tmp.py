import sys
import os
# import ..configs
from . import ADDA_CFG, util, TR_CFG, Trainer_CFG, Saving_CFG, CFG_Models
# from ..trainables.pre
from ..graphbuilders.pretrain_lbl import Pretrain_LBL as Pretrain
from ..graphbuilders.discriminator import Discriminator
from ..graphbuilders.targetmap_lbl import Targetmap_LBL as Targetmap

import tensorflow as tf

from sys import exit as xit


def pretrain_optim(self: Pretrain):
    return tf.train.AdamOptimizer()
both_optim=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5,beta2=0.999, name="combo_adam")

def discriminator_optim(self: Discriminator):
    # return tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.9,beta2=0.999, name="disc_adam")
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5,beta2=0.999, name="disc_adam")
    # return both_optim
    # return tf.contrib.opt.NadamOptimizer(
    # learning_rate=0.001,
    # beta1=0.9,
    # beta2=0.999,
    # epsilon=1e-08,
    # use_locking=False,
    # name='Adam')
    # return tf.train.RMSPropOptimizer(
    #     2.E-4,
    #     decay=0.9,
    #     momentum=0.01,
    #     epsilon=1e-10,
    #     use_locking=False,
    #     centered=False,
    #     name='d_RMSProp'
    # )
    # return tf.train.GradientDescentOptimizer(0.001,name="SGradientDescent")


def target_optim(self: Targetmap):
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5,beta2=0.999, name="target_adam")
    # return both_optim
    # return tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.9,beta2=0.999, name="target_adam")
    # return tf.train.AdamOptimizer(learning_rate=1.E-4, beta1=0.9, beta2=0.999)
    # return tf.train.RMSPropOptimizer(
    #     2.E-4,
    #     decay=0.9,
    #     momentum=0.01,
    #     epsilon=1e-10,
    #     use_locking=False,
    #     centered=False,
    #     name='t_RMSProp'
    # )


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
target_saver = Saving_CFG(
    restore_list=[
        "classifier",
        "source_map",
        "cross_src_target"
    ],
    save_list=[
        # "target_map"
    ]
)
discriminator_saver = Saving_CFG(
    restore_list=[
        "source_map",
        "classifier",
        # "target_map",
        # "discriminator",
        "cross_src_target"
    ],
    save_list=[
        "discriminator"
    ]
)
adversarial_saving = Saving_CFG(
    restore_list=[
        "classifier",
        "source_map",
        # ----
        "discriminator",
        # -----
        "target_map",
        "cross_src_target"
        # ----
    ],
    save_list=[
        "target_map",
        "discriminator",
    ]
)


from ..tfmodels import sanity_labels
from ..tfmodels import lenet
_MNIST_SVHN_LBL = ADDA_CFG(  # type: ADDA_CFG
    source_map=util.Dottify(),
    target_map=util.Dottify(),
    classifier=util.Dottify(),
    # savers=["a","b","c"],
    # models=CFG_Models(
    #     classifier=sanity_labels.classifier,
    #     discriminator=sanity_labels.discriminator,
    #     target_map=sanity_labels.encoder,
    #     source_map=sanity_labels.encoder
    # ),
    models=CFG_Models(
        # classifier=lenet.dense_classifier,
        classifier=lenet.dense_classifier,
        discriminator=lenet.discriminator,
        target_map=lenet.lenet_encoder,
        source_map=lenet.lenet_encoder
    ),
    discriminator=util.Dottify(),
    source_dataset="SVHN_LBL_28",
    target_dataset="MNIST_LBL",
    trcfg=TR_CFG(
        source_batch=128,
        target_batch=128,
        print_summaries=False
    ),
    input_shape=(28, 28),
    # hidden_shape=(500,),
    # hidden_shape=(10,),
    # model_save_subdir="MNIST_SVHN_LBL",
    # model_save_subdir="lbl_M_S_lenet_identity",
    combo_settings=util.Dottify(
        disc_tm_train_epoch_count=(1,1),
        disc_tm_train_step_count=(1,1),
        printerval=10
    ),
    model_save_subdir="SVHN_MNIST_ADDAPAPER",
    trainers=Trainer_CFG(
        pretrain=util.Dottify(
            trainable=Pretrain,
            optimizer=pretrain_optim,
            stop_val_acc=0.87,
            max_epochs=1000000,
            scope_re="^(classifier|source_map)",
            saving=pretrain_saver,
            num_classes=10,
            label="mnist_svhn_main"
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
from.adda_template import ADDA_MAIN_TEMPLATE
from . import new_template, combine
MNIST_SVHN_LBL = new_template("MNIST_SVHN_LBL", lambda:combine(ADDA_MAIN_TEMPLATE,_MNIST_SVHN_LBL))
