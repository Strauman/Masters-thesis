DOMAINS=[
    'CT_PT',
    'CT_T2',
    'PT_CT',
    'PT_T2',
    'T2_CT',
    'T2_PT'
]
from .adda_template import ADDA_MAIN_TEMPLATE, img_size, COMBO_CFG
from .adda_template import UNET_NO_CLF_MAXPOOL_MODEL, UNET_NO_CLF_MODEL, UNET_CLF_MODEL, MAIN_TRAINERS
from . import _CFG, util, ADDA_CFG_TEMPLATE, combine, new_template, cfg_label
from .combos import PRE_CFG
from .pretrain_defs import updates_for_mod_suff as pretr_updates
import importlib
# print(DS())
MAIN_ADDA= ADDA_CFG_TEMPLATE(
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
parpkg="adda.configs"
def generate_configs():
    configs_export=[]
    for d in DOMAINS:
        DS=getattr(importlib.import_module(".dataset_defs", parpkg), d)
        combo_pkg=importlib.import_module(f".combos.{d}", parpkg)
        DEFAULT_UPDATES=getattr(combo_pkg,f"{d}_UPDATES") # type: ADDA_CFG_TEMPLATE
        RESET_UPD=getattr(combo_pkg,f"{d}_RESET") # type: ADDA_CFG_TEMPLATE
        ADDA_UPD=getattr(combo_pkg,f"{d}_ADDA") # type: ADDA_CFG_TEMPLATE
        # DS.source_ds_name
        # eval(f"from .dataset_defs import {d} as DS")
        # eval(f"from .combos.{d} import {d}_ADDA as ADDA_UPD, {d}_RESET as RESET_UPD, import {d}_UPDATES as DEFAULT_UPDATES")
        base_config=combine(pretr_updates(DS().source_ds_name, "NO_CLF"),UNET_NO_CLF_MODEL,MAIN_TRAINERS, DS, ADDA_MAIN_TEMPLATE, DEFAULT_UPDATES, trigger=False)
        #pylint: disable=E0602
        ADDA_TMPL=new_template(f"{d}_ADDA", lambda base_config=base_config, ADDA_UPD=ADDA_UPD: combine(base_config, MAIN_ADDA, ADDA_UPD))
        RESET_TMPL=new_template(f"{d}_RESET", lambda base_config=base_config, RESET_UPD=RESET_UPD: combine(base_config, RESET_UPD))
        ####
        # clf_base_config=combine(pretr_updates(DS().source_ds_name, "CLF"),MAIN_TRAINERS,UNET_CLF_MODEL, DS, ADDA_MAIN_TEMPLATE, DEFAULT_UPDATES, trigger=False)
        # CLF_TMPL=new_template(f"{d}_CLF", lambda base_config=base_config, ADDA_UPD=ADDA_UPD: combine(clf_base_config, ADDA_UPD))
        mpool_base_config=combine(pretr_updates(DS().source_ds_name, "MAXPOOL"),MAIN_TRAINERS,UNET_NO_CLF_MAXPOOL_MODEL, DS, ADDA_MAIN_TEMPLATE, DEFAULT_UPDATES, trigger=False)
        MPOOL_TMPL_ADDA=new_template(f"{d}_MAXPOOL_ADDA", lambda mpool_base_config=mpool_base_config, ADDA_UPD=ADDA_UPD: combine(mpool_base_config, ADDA_UPD))
        MPOOL_TMPL_RESET=new_template(f"{d}_MAXPOOL_RESET", lambda mpool_base_config=mpool_base_config, RESET_UPD=RESET_UPD: combine(mpool_base_config, RESET_UPD))
        #pylint: enable=E0602
        configs_export+=[ADDA_TMPL,RESET_TMPL,MPOOL_TMPL_ADDA,MPOOL_TMPL_RESET]
    return configs_export
