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
    # WHEN DENSE
    # return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    return tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.9, beta2=0.999)

both_optim=tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5,beta2=0.999, name="combo_adam")

def discriminator_optim(self: Discriminator):
    return tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.5,beta2=0.999, name="discriminator_optim")
    # return both_optim
    # return tf.train.MomentumOptimizer(0.01, 0.999, use_locking=False, name='DiscOptim', use_nesterov=True)


def target_optim(self: Targetmap):
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5,beta2=0.999, name="target_optim")
    # return both_optim
    # return tf.train.MomentumOptimizer(0.05, 0.999, use_locking=False, name='Momentum', use_nesterov=True)

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
from ..tfmodels import lenet, unet_whole as unet, dense
mapping_unet_cfg=util.Dottify(depth=4, filters=8)
img_size=128
batch_size=5
MAIN_CFG = ADDA_CFG(  # type: ADDA_CFG
    source_map=mapping_unet_cfg,
    target_map=mapping_unet_cfg,
    classifier=util.Dottify(),
    discriminator=util.Dottify(),
    # savers=["a","b","c"],
    # models=CFG_Models(
    #     classifier=sanity_labels.classifier,
    #     discriminator=sanity_labels.discriminator,
    #     target_map=sanity_labels.encoder,
    #     source_map=sanity_labels.encoder
    # ),
    models=CFG_Models(
        # classifier=dense.identity_model,
        classifier=unet.classifier,
        # discriminator=dense.naive_dense_discriminator,
        discriminator=unet.conv_discriminator,
        target_map=unet.encoder_model,
        source_map=unet.encoder_model,
        # target_map=dense.shallow_dense_encoder,
        # source_map=dense.shallow_dense_encoder,
    ),
    trcfg=TR_CFG(
        source_batch=batch_size,
        target_batch=batch_size,
        print_summaries=False
    ),
    source_dataset=f"T2_ONLY_{img_size}",
    target_dataset=f"PT_ONLY_{img_size}",
    model_save_subdir=f"unet_MR2_PT_{img_size}",
#############
    # source_dataset=f"PT_ONLY_{img_size}",
    # target_dataset=f"T2_ONLY_{img_size}",
    # model_save_subdir=f"unet_PT_MR2_{img_size}",
#############
    input_shape=(img_size, img_size),
    # source_dataset="MNIST_SEG",
    # target_dataset="SVHN_LBL_28",
    # input_shape=(28, 28),
    combo_settings=util.Dottify(
        initial_disc_train_steps=0,
        initial_tm_train_steps=0,
        disc_train_first=False,
        simultaneous_iteration=False,
        disc_tm_train_epoch_count=(1,1),
        disc_tm_train_step_count=(1,1),
        printerval=10,
        write_out_interval=5,
    ),
    # hidden_shape=(500,),
    # hidden_shape=(10,),
    # model_save_subdir="MNIST_SVHN_LBL",
    # model_save_subdir="lbl_M_S_lenet_identity",
    trainers=Trainer_CFG(
        pretrain=util.Dottify(
            trainable=Pretrain,
            optimizer=pretrain_optim,
            stop_val_dice=0.81,
            max_epochs=1000000,
            # scope_re=".*",
            # scope_re="source_map",
            scope_re="^(classifier|source_map)",
            # scope_re="^((?!discriminator|target_map).)*/",
            saving=pretrain_saver,
            printerval=10,
            # scope_re="^(source_map)",
            # scope_re="^(classifier)",
        ),
        target=util.Dottify(
            trainable=Targetmap,
            optimizer=target_optim,
            stop_val_acc=1.,
            max_epochs=1000000,
            min_steps=0,
            scope_re="target_map",
            saving=target_saver,
            reset_optimizers=False,
            printerval=10,
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
            max_epochs=1000000,
            min_steps=0,
            reset_optimizers=False,
            saving=discriminator_saver,
            scope_re="^((?!source_map|target_map|classifier).)*/",
            printerval=10,
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
if __name__ == '__main__':
    MAIN_CFG.summary()
    # print(MAIN_CFG.__dict__)
