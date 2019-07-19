from .. import ADDA_CFG_TEMPLATE, util, Trainer_CFG, summary_call, COMBO_CFG, CFG_Models, combine
from ...tfmodels import unet_whole as unet
import tensorflow as tf
@summary_call
def pretrain_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)


@summary_call
def discriminator_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.5, beta2=0.999)
    # return tf.train.GradientDescentOptimizer(learning_rate=0.0002)


@summary_call
def target_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999)


    # return tf.train.GradientDescentOptimizer(learning_rate=0.0002)
T2_PT_UPDATES = ADDA_CFG_TEMPLATE(
    trainers=Trainer_CFG(
        is_template=True,
        # from_dict=MAIN_TRAINERS.__dict__,
        pretrain=util.Dottify(
            optimizer=pretrain_optim,
            stop_val_perf=0.9,
            max_epochs=500,
            early_stopping=util.Dottify(
                loss_difference_threshold=None,
                loss_difference_memory=None,
                percentage_total_decrease=None
            )
        ),
        target=util.Dottify(optimizer=target_optim, max_epochs=50),
        disc=util.Dottify(optimizer=discriminator_optim)
    ),
    combo_settings=COMBO_CFG(
        initial_disc_train_steps=0,
        initial_tm_train_steps=0,
        disc_train_first=True,
        simultaneous_iteration=False,
        disc_tm_train_epoch_count=(1, 1),
        disc_tm_train_step_count=(100, 20),
        reset_disc_optimizer_steps=0,
        disc_steps_after_reset=0,
        reset_tm_optimizer_steps=0,
        tm_steps_after_reset=0,
        cluster_reset_after=True,
        # reset_disc_optimizer_steps=1060,
        # reset_tm_optimizer_steps=20,
        # printerval=10,
        # write_out_interval=5,
        # printerval_seconds=True  # If not it's steps
        printerval=100,
        write_out_interval=50,
        printerval_seconds=False,  # If not it's steps
    )
)

T2_PT_ADDA = ADDA_CFG_TEMPLATE(
    # combo_settings=COMBO_CFG(
    #     # initial_disc_train_steps=200,
    #     # initial_tm_train_steps=0,
    #     # disc_train_first=True,
    #     # disc_tm_train_epoch_count=(1, 1),
    #     # disc_tm_train_step_count=(50, 1),
    #     reset_tm_optimizer_after_cluster_step=False,
    #     reset_disc_optimizer_after_cluster_step=False,
    #     cluster_reset_after=False,  # Reset before or after stepcount
    #     cluster_reset_both_after=False,  # Reset both after both have finished their step count
    #     reset_optimizers=False,
    #     ####
    #     reset_disc_optimizer_steps=0,
    #     disc_steps_after_reset=0,
    #     ####
    #     reset_tm_optimizer_steps=0,
    #     tm_steps_after_reset=0,
    #     # reset_disc_optimizer_steps=1060,
    #     # reset_tm_optimizer_steps=20,
    # )
)
@summary_call
def disc_optim_reset(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

@summary_call
def target_optim_reset(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999)


T2_PT_RESET = ADDA_CFG_TEMPLATE(
    # trainers=Trainer_CFG(
    #     is_template=True,
    #     target=util.Dottify(optimizer=target_optim_reset),
    #     disc=util.Dottify(optimizer=disc_optim_reset)
    # ),
    combo_settings=COMBO_CFG(
        # initial_disc_train_steps=200,
        # initial_tm_train_steps=0,
        # disc_train_first=False,
        # disc_tm_train_epoch_count=(1, 1),
        # disc_tm_train_step_count=(50, 1),
        reset_tm_optimizer_after_cluster_step=True,
        reset_disc_optimizer_after_cluster_step=True,
        cluster_reset_after=False,  # Reset before or after stepcount
        cluster_reset_both_after=False,  # Reset both after both have finished their step count
        reset_optimizers=False,
        ####
        reset_disc_optimizer_steps=0,
        disc_steps_after_reset=0,
        ####
        reset_tm_optimizer_steps=0,
        tm_steps_after_reset=0,
        # reset_disc_optimizer_steps=1060,
        # reset_tm_optimizer_steps=20,
    )
)
