import tensorflow as tf
from .. import ADDA_CFG_TEMPLATE, util, Trainer_CFG, summary_call, COMBO_CFG, combine, cfg_label
from ...graphbuilders.pretrain import Pretrain


@summary_call
def ct_pretrain_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)


@summary_call
def pt_pretrain_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

@summary_call
def t2_pretrain_optim(*args, **kwargs):
    return tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999)



CT_PRETRAIN = ADDA_CFG_TEMPLATE(
    trainers=Trainer_CFG(
        is_template=True,
        # from_dict=MAIN_TRAINERS.__dict__,
        pretrain=util.Dottify(
            trainable=Pretrain,
            optimizer=ct_pretrain_optim,
            stop_val_perf=1.,
            max_epochs=1000,
            early_stopping=util.Dottify(
                loss_difference_threshold=None,
                loss_difference_memory=None,
                percentage_total_decrease=None
            )
        ),
    )
)
PT_PRETRAIN = ADDA_CFG_TEMPLATE(
    trainers=Trainer_CFG(
        is_template=True,
        pretrain=util.Dottify(
            optimizer=pt_pretrain_optim,
            stop_val_perf=1.,
            max_epochs=1000,
            early_stopping=util.Dottify(
                loss_difference_threshold=None,
                loss_difference_memory=None,
                percentage_total_decrease=None
            )
        ),
    )
)
T2_PRETRAIN = ADDA_CFG_TEMPLATE(
    trainers=Trainer_CFG(
        is_template=True,
        # from_dict=MAIN_TRAINERS.__dict__,
        # source_map=util.Dottify(
        #     dropout=0.3,
        #     l2=0.002
        # ),
        pretrain=util.Dottify(
            optimizer=t2_pretrain_optim,
            stop_val_perf=1.,
            max_epochs=1000,
            early_stopping=util.Dottify(
                loss_difference_threshold=None,
                loss_difference_memory=None,
                percentage_total_decrease=None
            )
        )
    )
)
PRETRAIN_NO_CLF = cfg_label("noclf")
PRETRAIN_MAXPOOL = cfg_label("mpool")
# CT_PRETRAIN_CLF=ADDA_CFG_TEMPLATE(trainers=Trainer_CFG(is_template=True, pretrain=util.Dottify(label="clf")))
# CT_PRETRAIN_NO_CLF=ADDA_CFG_TEMPLATE(trainers=Trainer_CFG(is_template=True, pretrain=util.Dottify(label="noclf")))
# PT_PRETRAIN_CLF=ADDA_CFG_TEMPLATE(trainers=Trainer_CFG(is_template=True, pretrain=util.Dottify(label="clf")))
# PT_PRETRAIN_NO_CLF=ADDA_CFG_TEMPLATE(trainers=Trainer_CFG(is_template=True, pretrain=util.Dottify(label="noclf")))
#

# T2_PRETRAIN_CLF=ADDA_CFG_TEMPLATE(trainers=Trainer_CFG(is_template=True, pretrain=util.Dottify(label="clf")))
# T2_PRETRAIN_NO_CLF=ADDA_CFG_TEMPLATE(trainers=Trainer_CFG(is_template=True, pretrain=util.Dottify(label="noclf")))
