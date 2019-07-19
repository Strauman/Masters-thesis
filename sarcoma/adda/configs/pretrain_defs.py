from . import combine, ADDA_CFG_TEMPLATE, new_template, Trainer_CFG
from .adda_template import img_size, ADDA_MAIN_TEMPLATE, UNET_NO_CLF_MODEL, UNET_CLF_MODEL, MAIN_TRAINERS, UNET_NO_CLF_MAXPOOL_MODEL
from .combos import PRE_CFG
from ..helpers import util
modalities=["PT","CT","T2"]
def new_pre_ds(mod):
    return lambda: ADDA_CFG_TEMPLATE(
        source_ds_name=mod,
        # target_ds_name="PT",
        target_ds_name=mod,
        source_dataset=f"{mod}_ONLY_{img_size}",
        target_dataset=f"{mod}_ONLY_{img_size}",
        model_save_subdir=f"{mod}_PRE_{img_size}",
    )
def try_get_cfg_update(name, default=ADDA_CFG_TEMPLATE, model_save_subdir=None):
    if hasattr(PRE_CFG, name):
        return getattr(PRE_CFG, name)
    else:
        if default is ADDA_CFG_TEMPLATE:
            if model_save_subdir:
                return ADDA_CFG_TEMPLATE(model_save_subdir=model_save_subdir)
            return ADDA_CFG_TEMPLATE()
        else:
            return default
def updates_for_mod_suff(mod,suffix):
    return combine(try_get_cfg_update(f"{mod}_PRETRAIN_{suffix}"), try_get_cfg_update(f"PRETRAIN_{suffix}"), trigger=False)

def generate_configs():
    PRETRAIN_CONFIGS=[]
    for mod in modalities:
        # DS=getattr(importlib.import_module(".dataset_defs", parpkg), d)
        DS=new_pre_ds(mod)
        default=try_get_cfg_update(f"{mod}_PRETRAIN")
        PRETR_DEFAULT=try_get_cfg_update(f"{mod}_PRETRAIN")
        PRETR_NO_CLF=updates_for_mod_suff(mod, "NO_CLF")
        PRETR_MAXPOOL=updates_for_mod_suff(mod,"MAXPOOL")
        base_config=combine(MAIN_TRAINERS,ADDA_MAIN_TEMPLATE,PRETR_DEFAULT,DS, trigger=False)
        NO_CLF_CFG=new_template(f"{mod}_PRE_NOCLF", lambda UNET_NO_CLF_MODEL=UNET_NO_CLF_MODEL,base_config=base_config, PRETR_NO_CLF=PRETR_NO_CLF: combine(UNET_NO_CLF_MODEL, base_config, PRETR_NO_CLF))
        NO_CLF_NEW_CFG=new_template(f"{mod}_PRE_NOCLF_NEW", lambda UNET_NO_CLF_MODEL=UNET_NO_CLF_MODEL,base_config=base_config, PRETR_NO_CLF=PRETR_NO_CLF: combine(UNET_NO_CLF_MODEL, base_config, PRETR_NO_CLF))
        MAXPOOL_CFG=new_template(f"{mod}_PRE_MAXPOOL", lambda UNET_NO_CLF_MAXPOOL_MODEL=UNET_NO_CLF_MAXPOOL_MODEL, base_config=base_config, PRETR_MAXPOOL=PRETR_MAXPOOL: combine(UNET_NO_CLF_MAXPOOL_MODEL, base_config, PRETR_MAXPOOL))
        # no_clf_config=combine(base_config, UNET_CLF_MODEL, PRETR_NO_CLF)
        PRETRAIN_CONFIGS+=[NO_CLF_CFG,MAXPOOL_CFG,NO_CLF_NEW_CFG]
    return PRETRAIN_CONFIGS
