import sys
import os
# import ..configs
from . import ADDA_CFG, util, TR_CFG, Trainer_CFG, Saving_CFG, CFG_Models, ADDA_CFG_TEMPLATE, _CFG, TST_CFG, summary_call, COMBO_CFG, ADVERSARIAL_CFG, combine
# from ..trainables.pre

import tensorflow as tf


@summary_call
def pretrain_optim(*args, **kwargs):
    # WHEN DENSE
    # return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)
    return tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.9, beta2=0.999)


both_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999, name="combo_adam")


@summary_call
def discriminator_optim(*args, **kwargs):
    # return tf.train.GradientDescentOptimizer(learning_rate=0.0002)
    # return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.999)
    # return tf.train.MomentumOptimizer()
    # return tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.5, beta2=0.99, name="discriminator_optim")
    # BEST:
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999, name="discriminator_optim")
    # return both_optim
    # return tf.train.MomentumOptimizer(0.01, 0.999, use_locking=False, name='DiscOptim', use_nesterov=True)


@summary_call
def target_optim(*args, **kwargs):
    # return tf.train.GradientDescentOptimizer(learning_rate=0.0002)
    # return tf.train.AdamOptimizer(learning_rate=0.000002, beta1=0.5, beta2=0.999)
    # return tf.train.MomentumOptimizer()
    # return tf.train.AdamOptimizer(learning_rate=0.00002, beta1=0.5, beta2=0.99999, name="target_optim")
    # BEST:
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999, name="target_optim")
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
        "cross_src_target",
        "target_map"
    ],
    save_list=[
        "target_map"
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
mapping_unet_cfg = util.Dottify(depth=3, filters=8)
img_size = 128
batch_size = 5
relpath = lambda path: os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), path))

ADDA_MAIN_SETUP = ADDA_CFG_TEMPLATE(#type: ADDA_CFG
    models_dir="/root/data/models/",
    source_map=util.Dottify(),
    target_map=util.Dottify(),
    classifier=util.Dottify(),
    discriminator=util.Dottify(),
)
NO_RESET_SETTINGS=ADDA_CFG_TEMPLATE(
    combo_settings=COMBO_CFG(
        reset_tm_optimizer_after_cluster_step=False,
        reset_disc_optimizer_after_cluster_step=False,
        cluster_reset_after=False,  # Reset before or after stepcount
        cluster_reset_both_after=False,  # Reset both after both have finished their step count
        reset_optimizers=False,
        ####
        reset_disc_optimizer_steps=0,
        disc_steps_after_reset=0,
        ####
        reset_tm_optimizer_steps=0,
        tm_steps_after_reset=0,
    )
)
ADDA_MAIN_TEMPLATE = ADDA_CFG_TEMPLATE(  # type: ADDA_CFG
    is_template=True,
    models_dir="/root/data/models/",
    source_map=mapping_unet_cfg,
    target_map=mapping_unet_cfg,
    classifier=util.Dottify(),
    discriminator=util.Dottify(),
    trcfg=TR_CFG(
        source_batch=batch_size,
        target_batch=batch_size,
        print_summaries=False
    ),

    input_shape=(img_size, img_size),
    combo_settings=COMBO_CFG(
        initial_disc_train_steps=0,
        initial_tm_train_steps=0,
        disc_train_first=True,
        simultaneous_iteration=False,
        disc_tm_train_epoch_count=(1, 1),
        disc_tm_train_step_count=(1, 1),
        printerval=10,
        write_out_interval=5,
        printerval_seconds=True  # If not it's steps
        # printerval=500,
        # write_out_interval=500,
        # printerval_seconds=False  # If not it's steps
    )
)
from ..graphbuilders.pretrain import Pretrain
from ..graphbuilders.discriminator import Discriminator
from ..graphbuilders.targetmap import Targetmap
TRAINERS_DEFAULT_SETUP=ADDA_CFG_TEMPLATE(
    trainers=Trainer_CFG(
        pretrain=ADVERSARIAL_CFG(
            is_template=True,
            stop_val_perf=1.,
            max_epochs=float("inf"),
            scope_re="^(classifier|source_map)",
            saving=pretrain_saver
        ),
        target=ADVERSARIAL_CFG(
            is_template=True,
            stop_val_perf=1.,
            max_epochs=float("inf"),
            scope_re="target_map",
            saving=target_saver
        ),
        disc=ADVERSARIAL_CFG(
            is_template=True,
            stop_val_perf=1.,
            max_epochs=float("inf"),
            scope_re="^((?!source_map|target_map|classifier).)*/",
            saving=discriminator_saver
        ),
        adversarial_saving=adversarial_saving

    )

)
MAIN_TRAINERS = ADDA_CFG_TEMPLATE(
    trainers=Trainer_CFG(
        pretrain=util.Dottify(
            trainable=Pretrain,
            optimizer=pretrain_optim,
            stop_val_perf=0.81,
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
            stop_val_perf=1.,
            max_epochs=50,
            min_steps=0,
            scope_re="target_map",
            saving=target_saver,
            reset_optimizers=False,
            printerval=10,
            early_stopping=util.Dottify(
                loss_difference_threshold=None,
                loss_difference_memory=None,
                percentage_total_decrease=None  # 0.36
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
            stop_val_perf=1.,
            max_epochs=1000000,
            min_steps=0,
            reset_optimizers=False,
            saving=discriminator_saver,
            scope_re="^((?!source_map|target_map|classifier).)*/",
            # scope_re="discriminator",
            printerval=10,
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


NAIVE_DENSE_MODEL = ADDA_CFG_TEMPLATE(
    models=CFG_Models(
        # classifier=dense.identity_model,
        classifier=dense.identity_model,
        # discriminator=dense.naive_dense_discriminator,
        discriminator=dense.naive_dense_discriminator,
        target_map=dense.shallow_dense_encoder,
        source_map=dense.shallow_dense_encoder,
        # target_map=dense.shallow_dense_encoder,
        # source_map=dense.shallow_dense_encoder,
    ),
    model_save_subdir="dense_"
)


UNET_MAIN_MODEL = ADDA_CFG_TEMPLATE(
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
    model_save_subdir="unet_only_"
)

UNET_CLF_MODEL = ADDA_CFG_TEMPLATE(
    trainers=Trainer_CFG(
        is_template=True,
        pretrain=util.Dottify(label="CLF")
    ),
    models=CFG_Models(
        # classifier=dense.identity_model,
        classifier=unet.conv_classifier_first,
        # discriminator=dense.naive_dense_discriminator,
        discriminator=unet.conv_discriminator,
        target_map=unet.encoder_model_hidden,
        source_map=unet.encoder_model_hidden,
        # target_map=dense.shallow_dense_encoder,
        # source_map=dense.shallow_dense_encoder,
    ),
    model_save_subdir="unet_CLF_"

)

UNET_NO_CLF_MODEL = ADDA_CFG_TEMPLATE(
    models=CFG_Models(
        # classifier=dense.identity_model,
        classifier=dense.identity_model,
        # discriminator=dense.naive_dense_discriminator,
        discriminator=unet.conv_discriminator,
        target_map=unet.encoder_model,
        source_map=unet.encoder_model,
        # target_map=dense.shallow_dense_encoder,
        # source_map=dense.shallow_dense_encoder,
    ),
    model_save_subdir="unet_noclf_"
)

UNET_NO_CLF_MAXPOOL_MODEL = ADDA_CFG_TEMPLATE(
    models=CFG_Models(
        # classifier=dense.identity_model,
        classifier=dense.identity_model,
        # discriminator=dense.naive_dense_discriminator,
        discriminator=unet.conv_discriminator_mpool,
        target_map=unet.encoder_model_mpool,
        source_map=unet.encoder_model_mpool,
        # target_map=dense.shallow_dense_encoder,
        # source_map=dense.shallow_dense_encoder,
    ),
    model_save_subdir="unet_noclf_"
)


CLEAN_SETUP=combine(ADDA_MAIN_SETUP,NO_RESET_SETTINGS,TRAINERS_DEFAULT_SETUP,trigger=False)
# print(CONFIGS.__dict__)

# from sys import exit as xit
# T2_PT_UNET().summary()
# CT_T2_UNET().summary()
# xit()
