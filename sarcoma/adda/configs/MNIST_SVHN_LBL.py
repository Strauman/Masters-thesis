from . import ADDA_CFG_TEMPLATE, util, Trainer_CFG, TR_CFG, summary_call, COMBO_CFG, CFG_Models, ADVERSARIAL_CFG, cfg_label

from ..tfmodels import unet_whole as unet
import tensorflow as tf
from . import new_template, combine, make_template
from ..graphbuilders.pretrain_lbl import Pretrain_LBL
from ..graphbuilders.discriminator import Discriminator
from ..graphbuilders.targetmap_lbl import Targetmap_LBL


@summary_call
def pretrain_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)


@summary_call
def discriminator_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    # return tf.train.GradientDescentOptimizer(learning_rate=0.0002)


@summary_call
def target_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)


from ..tfmodels import lenet


# LENET_MODEL=
ADV_TMPL = make_template(ADVERSARIAL_CFG)  # type: ADVERSARIAL_CFG
# return tf.train.GradientDescentOptimizer(learning_rate=0.0002)
SVHN_MNIST_DS = ADDA_CFG_TEMPLATE(
    source_dataset="SVHN_LBL_28",
    target_dataset="MNIST_LBL",
    input_shape=(28, 28),
    model_save_subdir="SVHN_MNIST_"
)
MNIST_SVHN_DS = ADDA_CFG_TEMPLATE(
    source_dataset="MNIST_LBL",
    target_dataset="SVHN_LBL_28",
    input_shape=(28, 28),
    model_save_subdir="MNIST_SVHN_"
)

_SETUP = ADDA_CFG_TEMPLATE(

    trainers=Trainer_CFG(
        is_template=True,
        pretrain=ADV_TMPL(
            max_epochs=float("inf"),
            printerval=30,
            trainable=Pretrain_LBL,
            optimizer=pretrain_optim,
            max_summary_steps=100
        ),
        target=ADV_TMPL(
            max_epochs=5,
            printerval=5,
            trainable=Targetmap_LBL,
            optimizer=target_optim,
            max_summary_steps=5,
        ),
        disc=ADV_TMPL(
            max_epochs=5,
            printerval=5,
            trainable=Discriminator,
            optimizer=discriminator_optim,
            max_summary_steps=5,
            acc_var_window=50
        )
    ),
    trcfg=TR_CFG(
        source_batch=20,
        target_batch=20,
        print_summaries=False,
        # ds_s_val_send=[("shard",dict(num_shards=4,index=0))],
        # ds_t_val_send=[("shard",dict(num_shards=4,index=0))],
        prefetch=1000
    ),
    model_save_subdir="LBL",
    combo_settings=COMBO_CFG(
        initial_disc_train_steps=0,
        initial_tm_train_steps=0,
        disc_train_first=True,
        simultaneous_iteration=False,
        disc_tm_train_epoch_count=(1, 1),
        disc_tm_train_step_count=(10, 1),
        # max_steps=None,
        # printerval=60,
        # write_out_interval=5,
        # printerval_seconds=True,
        ###
        printerval=2000,
        write_out_interval=200,
        printerval_seconds=False,
    )
)


NOCLF = ADDA_CFG_TEMPLATE(
    models=CFG_Models(
        classifier=lenet.identity,
        discriminator=lenet.discriminator,
        target_map=lenet.lenet_classify_encoder,
        source_map=lenet.lenet_classify_encoder
    )
    )

STRAIGHT = ADDA_CFG_TEMPLATE(
    models=CFG_Models(
        classifier=lenet.dense_classifier,
        discriminator=lenet.discriminator,
        target_map=lenet.lenet_encoder,
        source_map=lenet.lenet_encoder
    )
)

NOCLF = combine(NOCLF, cfg_label("svhn-noclf"), trigger=False)
STRAIGHT = combine(STRAIGHT, cfg_label("main"), trigger=False)
_SVHN_MNIST_LBL = combine(_SETUP,NOCLF, trigger=False)
# _SVHN_MNIST_LBL=combine(STRAIGHT, _SETUP, trigger=False)
@summary_call
def reset_discriminator_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)
    # return tf.train.GradientDescentOptimizer(learning_rate=0.0002)


@summary_call
def reset_target_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.999)


ADDA_UPDATE = ADDA_CFG_TEMPLATE(
    combo_settings=COMBO_CFG(
        # initial_disc_train_steps=0,
        # initial_tm_train_steps=0,
        # disc_train_first=True,
        # simultaneous_iteration=False,
        # disc_tm_train_epoch_count=(1, 1),
        # disc_tm_train_step_count=(1, 1),
    )
)
RESET_UPDATE = ADDA_CFG_TEMPLATE(
    trainers=make_template(Trainer_CFG)(
        disc=ADV_TMPL(optimizer=reset_discriminator_optim),
        target=ADV_TMPL(optimizer=reset_target_optim)
    ),
    trcfg=TR_CFG(
        source_batch=20,
        target_batch=20,
        print_summaries=False,
        # ds_s_val_send=[("shard",dict(num_shards=4,index=0))],
        # ds_t_val_send=[("shard",dict(num_shards=4,index=0))],
        prefetch=1000
    ),
    combo_settings=COMBO_CFG(
        initial_disc_train_steps=0,
        initial_tm_train_steps=0,
        disc_tm_train_step_count=(10, 1),
        reset_tm_optimizer_after_cluster_step=False,
        reset_disc_optimizer_after_cluster_step=False,
        cluster_reset_after=False,  # Reset before or after stepcount
        cluster_reset_both_after=False,  # Reset both after both have finished their step count
        reset_optimizers=False,
        ####
        reset_disc_optimizer_steps=100,
        disc_steps_after_reset=0,
        ####
        reset_tm_optimizer_steps=0,
        tm_steps_after_reset=0,
    )
)
from .adda_template import CLEAN_SETUP

SVHN_MNIST_BASE = combine(CLEAN_SETUP, SVHN_MNIST_DS, _SVHN_MNIST_LBL)


def generate_configs():
    SVHN_MNIST_LBL_ADDA = new_template("SVHN_MNIST_LBL_ADDA", lambda: combine(SVHN_MNIST_BASE, ADDA_UPDATE))
    SVHN_MNIST_LBL_RESET = new_template("SVHN_MNIST_LBL_RESET", lambda: combine(SVHN_MNIST_BASE, RESET_UPDATE))
    CONFIGS = [SVHN_MNIST_LBL_ADDA, SVHN_MNIST_LBL_RESET]
    return CONFIGS
