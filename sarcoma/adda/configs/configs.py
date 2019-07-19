from .. import color_print, util
# from .config_defs import CONFIGS as tmpl_cfgs
from .method_defs import generate_configs as gen_meth_cfg
from .pretrain_defs import generate_configs as gen_pre_cfg
from .MNIST_SVHN_LBL import generate_configs as gen_mmnistsvhn_cfg
from . import Triggerable

class ConfigError(BaseException):
    pass

import click
# config_list=[]+tmpl_cfgs+METHOD_CONFIGS+PRETRAIN_CONFIGS
config_list=[]
config_list+=gen_meth_cfg()
config_list+=gen_pre_cfg()
config_list+=gen_mmnistsvhn_cfg()
# config_list=[]+tmpl_cfgs
# config_list=[]+METHOD_CONFIGS
configs={}
CONFIG=None
CONFIG_NAME=None
config_names=[ c.name for c in config_list]
for c in config_list:
    configs[c.name]=c

def ask_config():
    idx,CONFIG=util.ask_choose_one(lst=config_list, message="Choose a config", names=config_names, failsafe=False, sysarg_key="--cfg")
    CONFIG_NAME=config_names[idx]
    # CONFIG=configs[CONFIG_NAME]
    assert isinstance(CONFIG, Triggerable), "CONFIG is not Triggerable template!"
    CONFIG=CONFIG()
    return CONFIG, CONFIG_NAME

def get_config(cfg_name):
    if cfg_name not in configs.keys():
        raise ConfigError(f"Cannot find config with name {cfg_name}")
    idx=config_names.index(cfg_name)
    CONFIG=configs[cfg_name]
    CONFIG_NAME=cfg_name
    # CONFIG=configs[CONFIG_NAME]
    assert isinstance(CONFIG, Triggerable), "CONFIG is not Triggerable template!"
    CONFIG=CONFIG()
    return CONFIG, CONFIG_NAME

if __name__ == '__main__':
    import pout
    CONFIG.summary()
    # print(CONFIG.model_save_subdir)
    pout.x(0)
