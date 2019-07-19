from .seeding import *

from . import trainhelp as hlp
from .graphbuilders.discriminator import Discriminator
# from .trainables.targetmap import Targetmap
# from .trainables.pretrain import Pretrain
from .trainables.trainable import Trainable
from . import datasets
from .graphbuilders.targetmap import Targetmap as Targetmap
from .graphbuilders.pretrain import Pretrain as Pretrain
from .graphs.default import build_graph
from .helpers.cprints import color_print
from .helpers.trainhelp import new_saver, new_cross_saver
import sys
# DS_s,DS_t,cfg.config_save_file,cfg,tensorboard_path,CONFIG_NAME=build_graph()
DS_s = DS_t = cfg = tensorboard_path = CONFIG_NAME = None
pretrain_settings = target_settings = disc_settings = None
from .trainables import EarlyStopping, StopTraining
from .trainables.combotrainer import ComboTrainer
from sys import exit as xit
# from . import ansi_print
from .helpers.ansi_print import NOTIFICATIONS, run_notifier
from . import util
from .tables import tablehelper
from .configs.configs import get_config as _get_config
from .helpers.trainhelp import SaverObject as SAVOBJ
from . import exporthead
best_pretrain_val = 0
# tm_trainer: Targetmap = cfg.trainers.target.trainable(DS_s=DS_s, DS_t=DS_t, settings=target_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.target_map,writeable=False)
# disc_trainer: Discriminator = cfg.trainers.disc.trainable(DS_s=DS_s, DS_t=DS_t, settings=disc_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.discriminator,writeable=False)
# pretrainer: Pretrain = cfg.trainers.pretrain.trainable(DS_s=DS_s, DS_t=None, settings=pretrain_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.source_map, writeable=False)
tm_trainer = None  # type: Targetmap
disc_trainer = None  # type: Discriminator
pretrainer = None  # type: Pretrain
sess = None  # type: tf.InteractiveSession
EXPORT_RESTORE_LIST = ["classifier", "source_map", "target_map", "discriminator"]
States = exporthead.States
restore_num = util.get_valv("--restore-num", None)
restore_file_suffix = None
if restore_num is not None:
    restore_file_suffix = f"config-{CONFIG_NAME}-{restore_num}"
    color_print(f"RESTORING FROM CONFIG {restore_num} ({restore_file_suffix})", style="warning")


def restore_suffix_for_cfg_num(cfg_name=None, num=None):
    cfg_name = cfg_name or CONFIG_NAME
    num = num or restore_num
    return f"config-{cfg_name}-{num}"


print(restore_file_suffix)


def load_config(cfg_name):
    global cfg, model_save_root, tm_trainer, disc_trainer, pretrainer
    global DS_s, DS_t, cfg, tensorboard_path, CONFIG_NAME, models_dir, sess, pretrain_settings, target_settings, disc_settings
    if sess is not None:
        sess.close()
    tf.reset_default_graph()
    # return cfg
    new_cfg, new_cfg_name = _get_config(cfg_name)
    if cfg is None:
        cfg = new_cfg
    DS_s, DS_t, cfg, tensorboard_path, CONFIG_NAME = build_graph(new_cfg)
    models_dir = new_cfg.models_dir
    cfg.model_save_root = model_save_root = os.path.join(models_dir, new_cfg.model_save_subdir)
    color_print(f"Restoring config {new_cfg_name}")
    # new_cfg._savers=util.Dottify(#pylint: disable=W0212
    old_cfg = cfg
    cfg = new_cfg
    pretrain_settings = cfg.trainers.pretrain
    target_settings = cfg.trainers.target
    disc_settings = cfg.trainers.disc
    cfg._savers = hlp.savers_from_cfg(cfg)
    tm_trainer = cfg.trainers.target.trainable(DS_s=DS_s, DS_t=DS_t, settings=target_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.target_map, writeable=False)
    disc_trainer = cfg.trainers.disc.trainable(DS_s=DS_s, DS_t=DS_t, settings=disc_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.discriminator, writeable=False)
    pretrainer = cfg.trainers.pretrain.trainable(DS_s=DS_s, DS_t=None, settings=pretrain_settings, tensorboard_path=tensorboard_path, saver=cfg._savers.source_map, writeable=False)
    # cfg=new_cfg
    init_session()
    return cfg


def init_session():
    global tm_trainer, disc_trainer, pretrainer, sess

    sess = tf.InteractiveSession()
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), Trainable.variables_initializer(Trainable.GraphKeys.INTRINSIC)])
    disc_trainer.sess_enter(sess)
    tm_trainer.sess_enter(sess)
    pretrainer.sess_enter(sess)


def get_state(state, cfg_name=None, num=None):
    combos = [
        ("Target restart", [0, 1, 4]),
        ("Continue", [0, 1, 2, 3])
    ]
    print(EXPORT_RESTORE_LIST)
    if num != 0:
        num = num or util.get_valv("--restore-num")
    suff = restore_suffix_for_cfg_num(cfg_name=cfg_name, num=num)
    for restore_name in EXPORT_RESTORE_LIST:
        curr_state = state
        curr_suff = suff
        saver = cfg.savers[restore_name]  # type: SAVOBJ
        try_list = hlp.TRYLIST.FULL_ONLY
        if restore_name in ["source_map", "classifier"]:
            curr_suff = ""
            try_list = ["{restore_dir}", "{restore_dir}-{restore_state}"]
        saver.restore(sess=tf.get_default_session(), restore_dir=None, restore_state=curr_state, restore_suffix=curr_suff, try_list=try_list, verbose=1)
    # hlp.ask_restore(EXPORT_RESTORE_LIST, savers, tf.get_default_session(), index_combos=combos, restore_suffix=suff, restore_state=state)
    pretrainer.post_restore()
    tm_trainer.post_restore()
    disc_trainer.post_restore()


def it_restore_state(states):
    def _it():
        for s in states:
            yield get_state(s)


# sess=tf.Session()
# sess.as_default()
# sess = tf.InteractiveSession()
# sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), Trainable.variables_initializer(Trainable.GraphKeys.INTRINSIC)])
# disc_trainer.sess_enter(sess)
# tm_trainer.sess_enter(sess)
# pretrainer.sess_enter(sess)

# from .exporthead import .
current_cfg_name = None
current_num = None


def columns_to_rows(columns):
    n_rows = len(columns[0])
    rows = [[] for _ in range(n_rows)]
    for i, c in enumerate(columns):
        for j, value in enumerate(c):
            rows[j].append(value)
    return rows
column_to_rows=columns_to_rows
# For sanity testing
## cfg=util.Dottify(source_ds_name="SRC", target_ds_name="DST")
# vals=util.Dottify(
##     initial=util.Dottify(f1="1", accuracy="2", auc="3"),
##     latest=util.Dottify(f1="10", accuracy="20", auc="30")
# )


class OUT_IDX():
    order = []

    def map(self, in_columns, pre_columns=None):
        if pre_columns is None:
            extra_offset = 0
        else:
            extra_offset = len(pre_columns)
        out_columns = [[] for _ in self.order]
        for i, ident in enumerate(self.order):
            out_columns[i] = in_columns[getattr(self, ident)]
        for i, ident in enumerate(self.order):
            setattr(self, ident, i + extra_offset)
        return [*pre_columns, *out_columns]


_icIDX = {}


class columnIDC(util.Dottify):
    ADDA_initial = None
    ADDA_best = None
    ADDA_conf = None
    ADDA_latest = None
    ADDA_loss = None
    RESET_conf = None
    RESET_latest = None
    RESET_best = None
    RESET_loss = None
    order = [
        "ADDA_initial",
        "ADDA_best",
        "ADDA_conf",
        "ADDA_latest",
        "ADDA_loss",
        "RESET_conf",
        "RESET_latest",
        "RESET_best",
        "RESET_loss",
    ]
    columns = []


def get_result_columns(cfg_prefix, adda_reset_nums, export="ALL") -> columnIDC:
    global _icIDX
    icdx_key_name = "-".join(util.l2s([cfg_prefix, *adda_reset_nums]))
    if export == "ALL":
        export = ["f1", "accuracy", "auc", "seg_f1"]
    if icdx_key_name not in _icIDX:
        np.set_printoptions(precision=3, suppress=True, floatmode='fixed')
        in_columns = []
        cfg_name = cfg_prefix + "_ADDA"
        load_config(cfg_name)
        vals = set_state_values(states=["initial", "best_perf", "abs_confusion", "latest", "best_loss"], cfg_name=cfg_name, num=adda_reset_nums[0])
        for state, v in vals.items():
            in_columns.append(dict(f1=v.f1, accuracy=v.accuracy, auc=v.auc, seg_f1=v.seg_f1))
        cfg_name = cfg_prefix + "_RESET"
        load_config(cfg_name)
        vals = set_state_values(states=["abs_confusion", "latest", "best_perf", "best_loss"], cfg_name=cfg_name, num=adda_reset_nums[1])
        for state, v in vals.items():
            in_columns.append(dict(f1=v.f1, accuracy=v.accuracy, auc=v.auc, seg_f1=v.seg_f1))
        cIDX = {}
        for i, name in enumerate(columnIDC.order):
            cIDX[name] = i
        icIDX = columnIDC(from_dict=cIDX)
        export_columns = [[] for _ in in_columns]
        # for ic in in_columns:
        #     for exp in export:
        #         if
        icIDX._columns = in_columns

        _icIDX[icdx_key_name] = icIDX
    icIDX = _icIDX[icdx_key_name]
    icIDX.columns = [[ic[exp] for exp in export] for ic in icIDX._columns]
    return icIDX
# def load_state(cfg_name, state, num, restore_list=None):
#     cfg, tm_trainer, disc_trainer, pretrainer=exporthead.load_config(cfg_name, restore_list=restore_list)
#     get_state(state, cfg, num, tm_trainer,disc_trainer,pretrainer,restore_list=restore_list)
#     return cfg, tm_trainer, disc_trainer, pretrainer


def set_state_values(states, cfg_name=None, num=None):
    vals = util.Dottify()
    for state in states:
        get_state(state, cfg_name=cfg_name, num=num)
        s_vals = util.Dottify(
            f1=np.array2string(tm_trainer.get_epoch_value_val(tm_trainer.f1)),
            accuracy=np.array2string(tm_trainer.get_epoch_value_val(tm_trainer.classification_accuracy)),
            auc=np.array2string(tm_trainer.get_epoch_value_val(tm_trainer._auc)),
            seg_f1=(pretrainer.get_epoch_value_val(pretrainer.f1))
        )
        setattr(vals, state, s_vals)
    return vals


def get_svhn_results(adda_name, reset_name, nums):
    out_values = util.Dottify()
    out_values.clf = {}
    out_values.adda = {}
    out_values.reset = {}
    states = [States.init, States.best, States.conf, States.loss, States.latest]
    for state in states:
        out_values.adda[state] = -1
        out_values.adda[state + "step"] = -1
        out_values.reset[state] = -1
        out_values.reset[state + "step"] = -1

    load_config(adda_name)
    # for state in ["abs_confusion", "latest", "best_loss"]:
    for state in states:
        get_state(state, cfg_name=adda_name, num=nums[0])
        out_values.adda[state] = tm_trainer.get_epoch_value_val(tm_trainer.classification_accuracy)
        out_values.adda[state + "step"] = (tm_trainer.info.step * (tm_trainer.info.epoch + 1)) + (disc_trainer.info.step * (disc_trainer.info.epoch + 1))
    load_config(reset_name)
    for state in states:
        get_state(state, cfg_name=reset_name, num=nums[1])
        out_values.reset[state] = tm_trainer.get_epoch_value_val(tm_trainer.classification_accuracy)
        out_values.reset[state + "step"] = (tm_trainer.info.step * (tm_trainer.info.epoch + 1)) + (disc_trainer.info.step * (disc_trainer.info.epoch + 1))
    out_values.clf = pretrainer.get_epoch_value_val(pretrainer.acc)
    return out_values


def SVHN_ROWS(nums):
    base_name = "SVHN_MNIST_LBL"
    results = get_svhn_results(base_name + "_ADDA", base_name + "_RESET", nums)
    # Make columns
    columns = [
        results.clf,
        results.adda[States.best],
        results.reset[States.best],
        results.adda[States.conf],
        results.reset[States.conf],
        results.adda[States.latest],
        results.reset[States.latest],
        results.adda[States.loss],
        results.reset[States.loss]
    ]
    column_steps = [
        0,
        results.adda[States.best + "step"],
        results.reset[States.best + "step"],
        results.adda[States.conf + "step"],
        results.reset[States.conf + "step"],
        results.adda[States.latest + "step"],
        results.reset[States.latest + "step"],
        results.adda[States.loss + "step"],
        results.reset[States.loss + "step"]
    ]

    # columns=list(zip(columns,column_steps))
    # columns=[str(0) for _ in range(9)]
    # column_steps=[str(0) for _ in range(9)]
    columns = [[c] for c in columns]
    column_steps = [[c] for c in column_steps]
    rows = column_to_rows(columns)
    best_in_row = tablehelper.bf_highest_indices(rows, 3, 8)
    same_step_indices = tablehelper.idx_for_same_values(column_steps)
    column_marks = [
        tablehelper.bf_highest_indices(rows, 3, 4),
        tablehelper.bf_highest_indices(rows, 5, 6),
        tablehelper.bf_highest_indices(rows, 7, 8)
    ]
    # rows=tablehelper.bf_highest(rows, 1,2, bf_cmd=r"\underline")
    rows = tablehelper.rows_to_string(rows, precision=4, suppress=True, floatmode='fixed')
    for m in column_marks:
        rows = tablehelper.mark_indices(rows, m, cmd=r"\underline")
    rows = tablehelper.mark_indices(rows, best_in_row, cmd=r"\textbf")
    rows = tablehelper.at_indices(rows, same_step_indices, suffix=r"\*")
    rows = tablehelper.finalise(util.ensure_list(rows))
    return rows


def standard_rows(cfg_prefix, adda_reset_nums, out_idx, in_columns):
    source_name, target_name = cfg_prefix.split("_")
    # multirow = f"\multirow{{3}}{{*}}{{{source_name} $\\to$ {target_name}}}"
    modality_col = [[f"{{{source_name} $\\to$ {target_name}}}"]]
    extra_cols = modality_col
    # extra_cols = [[multirow, "", ""], ["\\F1", "Accuracy", "AUC"]]
    columns = out_idx.map(in_columns, extra_cols)
    try:
        rows = column_to_rows(columns)
    except BaseException:
        print(columns)
        raise
    rows = util.ensure_list(rows)
    return rows


def DA_ROWS(cfg_prefix, adda_reset_nums):
    source_name, target_name = cfg_prefix.split("_")
    icIDX = get_result_columns(cfg_prefix, adda_reset_nums)
    in_columns = icIDX.columns

    class DA_outcol_idx(OUT_IDX):
        initial = None
        # potential = None
        da_best = None
        reset = None
        # latest = None
        order = [
            "initial",
            # "potential",
            "da_best",
            "reset",
            # "latest"
        ]

    ocIDX = DA_outcol_idx()
    R_max = False
    ocIDX.initial = icIDX.ADDA_initial
    #pylint: disable=E1126
    # if float(in_columns[icIDX.RESET_conf][0]) >= float(in_columns[icIDX.ADDA_best][0]) and float(in_columns[icIDX.RESET_conf][0]) >= float(in_columns[icIDX.RESET_best][0]):
    #     R_max = True
    #     ocIDX.potential = icIDX.RESET_conf
    # elif float(in_columns[icIDX.RESET_best][0]) >= float(in_columns[icIDX.ADDA_best][0]):
    #     R_max = True
    #     ocIDX.potential = icIDX.RESET_best
    # else:
    #     ocIDX.potential = icIDX.ADDA_best
    ocIDX.da_best = icIDX.ADDA_conf
    ocIDX.reset = icIDX.RESET_conf
    # latest_r_max = False
    # if float(in_columns[icIDX.RESET_latest][0]) >= float(in_columns[icIDX.ADDA_latest][0]):
    #     latest_r_max = True
    #     ocIDX.latest = icIDX.RESET_latest
    # else:
    #     ocIDX.latest = icIDX.ADDA_latest

    multirow = rf"\multirow{{3}}{{*}}{{{source_name} $\\to$ {target_name}}}"
    extra_cols = [[multirow, "", ""], ["\\F1", "Accuracy", "AUC"]]
    columns = ocIDX.map(in_columns, extra_cols)
    rows = column_to_rows(columns)
    bf_offset = 4
    rows = tablehelper.bf_highest(rows, 3)
    # if R_max:
    #     rows[0][ocIDX.potential] = rows[0][ocIDX.potential] + r"\rlap{\emph{R}}"
    # if latest_r_max:
    #     rows[0][ocIDX.latest] = rows[0][ocIDX.latest] + r"\rlap{\emph{R}}"

    rows = ["&".join(r) for r in util.ensure_list(rows)]
    rows = [r + "&" for r in util.ensure_list(rows)]
    # Set the multicol
    rows[0] = rows[0] + "\\TStrutM"
    rows[-1] = rows[-1] + "\\BStrutM"
    rows = "\\\\\n".join(rows)
    rows += "\\\\\\sline\n"
    return rows


def CONFUSION_ROWS(cfg_prefix, adda_reset_nums):
    source_name, target_name = cfg_prefix.split("_")
    icIDX = get_result_columns(cfg_prefix, adda_reset_nums, export=["f1"])
    in_columns = icIDX.columns

    class CONF_OUT_IDX(OUT_IDX):
        adda_conf = None
        reset_conf = None
        adda_latest = None
        reset_latest = None
        adda_loss = None
        reset_loss = None
        adda_best = None
        reset_best = None
        order = [
            "adda_best",
            "reset_best",
            "adda_conf",
            "reset_conf",
            "adda_latest",
            "reset_latest",
            "adda_loss",
            "reset_loss",
        ]

    ocIDX = CONF_OUT_IDX()
    ocIDX.adda_conf = icIDX.ADDA_conf
    ocIDX.reset_conf = icIDX.RESET_conf
    ocIDX.adda_latest = icIDX.ADDA_latest
    ocIDX.reset_latest = icIDX.RESET_latest
    ocIDX.adda_loss = icIDX.ADDA_loss
    ocIDX.reset_loss = icIDX.RESET_loss
    ocIDX.adda_best = icIDX.ADDA_best
    ocIDX.reset_best = icIDX.RESET_best
    # multirow = f"\multirow{{3}}{{*}}{{{source_name} $\\to$ {target_name}}}"
    rows = standard_rows(cfg_prefix, adda_reset_nums, ocIDX, in_columns)
    best_in_row = tablehelper.bf_highest_indices(rows, 3, 8)
    # rows=tablehelper.bf_highest(rows, 1,2, bf_cmd=r"\underline")
    rows = tablehelper.bf_highest(rows, 3, 4, bf_cmd=r"\underline")
    rows = tablehelper.bf_highest(rows, 5, 6, bf_cmd=r"\underline")
    rows = tablehelper.bf_highest(rows, 7, 8, bf_cmd=r"\underline")
    rows = tablehelper.mark_indices(rows, best_in_row, cmd=r"\textbf")
    rows = tablehelper.finalise(util.ensure_list(rows))
    return rows


def PEAK_ROWS(cfg_prefix, adda_reset_nums):
    # adda_reset_nums=[str(n) for n in adda_reset_nums]
    icIDX = get_result_columns(cfg_prefix, adda_reset_nums, export=["f1"])
    in_columns = icIDX.columns

    class PEAK_OUT_IDX(OUT_IDX):
        ADDA_best = None
        RESET_best = None
        order = [
            "ADDA_best",
            "RESET_best"
        ]

    ocIDX = PEAK_OUT_IDX()
    ocIDX.ADDA_best = icIDX.ADDA_best
    ocIDX.RESET_best = icIDX.RESET_best
    # ocIDX.ADD =icIDX.ADDA_conf
    rows = standard_rows(cfg_prefix, adda_reset_nums, ocIDX, in_columns)
    rows = tablehelper.bf_highest(rows, 1)
    rows = tablehelper.finalise(util.ensure_list(rows))
    return rows


def get_table(rows_func, table_head, torestore, save_to):
    table_foot = r"\end{tabular}"
    ROWS = ""
    for mf in torestore:
        ROWS += rows_func(*mf)
    out_str = table_head + ROWS + table_foot
    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(out_str)
    return out_str


def DA_TABLE(torestore, save_to=None):
    table_head = tablehelper.DA_TABLE_HEAD
    table_foot = r"\end{tabular}"
    ROWS = ""
    for mf in torestore:
        ROWS += DA_ROWS(*mf)
    out_str = table_head + ROWS + table_foot
    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(out_str)
    return out_str


# ROWS+=DA_ROWS("PT_T2_RESET", 41)
if __name__ == '__main__':

    MOD_FAVS = [  # ACTUAL FAVS
        ("PT_T2", (78, 50)),
        ("PT_CT", (1, 5)),
        ("CT_PT", (110, 21)),
        ("CT_T2", (1, 4)),
        ("T2_CT", ("0", "0")),
        ("T2_PT", (41, 72)),
    ]
    MOD_MAXPOOLS = [
        ("PT_T2", ("0", "0"))
    ]

    # MOD_FAVS = [## FOR TESTING
    #     ("PT_T2", (78, 50)),
    #     ("T2_PT", (30, 65)),
    # ]
    # print(DA_TABLE(MOD_FAVS, "/root/src/tex-tables/tbl-da-reset.tex"))
    # print(CONFUSION_TABLE(MOD_FAVS, "/root/src/tex-tables/tbl-confusion-table.tex"))
    # print(get_table(CONFUSION_ROWS, table_head=tablehelper.CONF_TABLE_HEAD, torestore=MOD_FAVS, save_to="/root/src/tex-tables/tbl-confusion.tex"))
    # print(get_table(MAXPOOL_ROWS, table_head=tablehelper.MAXPOOL_TABLE_HEAD, torestore=MOD_FAVS, save_to="/root/src/tex-tables/tbl-confusion.tex"))
    svhn_numbers = [
        # [("68", "39")],
        [("66", "37")],
        [("66", "39")]
    ]
    print(get_table(SVHN_ROWS,
                    table_head=tablehelper.SVHN_TABLE_HEAD,
                    torestore=svhn_numbers,
                    save_to="/root/src/tex-tables/tbl-svhn.tex"))
    # print(Sa)
    # print(get_table(PEAK_ROWS, table_head=tablehelper.PEAK_TABLE_HEAD, torestore=MOD_FAVS, save_to="/root/src/tex-tables/tbl-peak.tex"))
