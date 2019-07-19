import collections

from .adda_template import ADDA_MAIN_TEMPLATE, img_size
from . import _CFG, util, ADDA_CFG_TEMPLATE, combine, new_template

from .dataset_defs import T2_PT,T2_CT,PT_T2,PT_CT,CT_PT,CT_T2
# Template compositions
from .combos.T2_CT import T2_CT_UPDATES#,T2_CT_DENSE_UPDATES
from .combos.CT_PT import CT_PT_UPDATES#,CT_PT_DENSE_UPDATES
from .combos.PT_CT import PT_CT_UPDATES#,PT_CT_DENSE_UPDATES
from .combos.CT_T2 import CT_T2_UPDATES#,CT_T2_DENSE_UPDATES
from .combos.PT_T2 import PT_T2_UPDATES#,PT_T2_DENSE_UPDATES
from .combos.T2_PT import T2_PT_UPDATES#,T2_PT_DENSE_UPDATES
from .adda_template import MAIN_TRAINERS
from .adda_template import UNET_NO_CLF_MODEL,UNET_MAIN_MODEL,NAIVE_DENSE_MODEL,UNET_CLF_MODEL
CONFIGS = [
    new_template(name="PT_T2_UNET", callback=lambda: combine(UNET_NO_CLF_MODEL, MAIN_TRAINERS, PT_T2, ADDA_MAIN_TEMPLATE,PT_T2_UPDATES)),
    new_template(name="PT_CT_UNET", callback=lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, PT_CT, ADDA_MAIN_TEMPLATE, PT_CT_UPDATES)),

    new_template(name="T2_PT_UNET", callback=lambda: combine(UNET_NO_CLF_MODEL, MAIN_TRAINERS, T2_PT, ADDA_MAIN_TEMPLATE,T2_PT_UPDATES)),
    new_template(name="T2_CT_UNET", callback=lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, T2_CT, ADDA_MAIN_TEMPLATE, T2_CT_UPDATES)),

    new_template(name="CT_PT_UNET", callback=lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, CT_PT, ADDA_MAIN_TEMPLATE, CT_PT_UPDATES)),
    new_template(name="CT_T2_UNET", callback=lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, CT_T2, ADDA_MAIN_TEMPLATE, CT_T2_UPDATES)),
#### DENSE ####
    # new_template(name="PT_T2_DENSE", callback=lambda: combine(NAIVE_DENSE_MODEL, MAIN_TRAINERS, PT_T2, ADDA_MAIN_TEMPLATE,PT_T2_DENSE_UPDATES))
    # new_template(name="PT_CT_UNET", callback=lambda: combine(NAIVE_DENSE_MODEL, MAIN_TRAINERS, PT_CT, ADDA_MAIN_TEMPLATE, PT_CT_DENSE_UPDATES)),
    #
    # new_template(name="T2_PT_UNET", callback=lambda: combine(NAIVE_DENSE_MODEL, MAIN_TRAINERS, T2_PT, ADDA_MAIN_TEMPLATE,T2_PT_DENSE_UPDATES)),
    # new_template(name="T2_CT_UNET", callback=lambda: combine(NAIVE_DENSE_MODEL, MAIN_TRAINERS, T2_CT, ADDA_MAIN_TEMPLATE, T2_CT_DENSE_UPDATES)),
    #
    # new_template(name="CT_PT_UNET", callback=lambda: combine(NAIVE_DENSE_MODEL, MAIN_TRAINERS, CT_PT, ADDA_MAIN_TEMPLATE, CT_PT_DENSE_UPDATES)),
    # new_template(name="CT_T2_UNET", callback=lambda: combine(NAIVE_DENSE_MODEL, MAIN_TRAINERS, CT_T2, ADDA_MAIN_TEMPLATE, CT_T2_DENSE_UPDATES)),
]
# PT_T2_UNET T2_PT_UNET CT_PT_UNET PT_CT_UNET

# PT_T2_UNET = new_template(lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, PT_T2, ADDA_MAIN_TEMPLATE), "PT_T2_UNET")
# T2_PT_UNET = new_template(lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, T2_PT, ADDA_MAIN_TEMPLATE), "T2_PT_UNET")
#
# T2_CT_UNET = new_template(lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, T2_CT, ADDA_MAIN_TEMPLATE, T2_CT_UPDATES), "T2_CT_UNET")
#
# CT_PT_UNET = new_template(lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, CT_PT, ADDA_MAIN_TEMPLATE,CT_PT_UPDATES), "CT_PT_UNET")
# PT_CT_UNET = new_template(lambda: combine(UNET_MAIN_MODEL, MAIN_TRAINERS, PT_CT, ADDA_MAIN_TEMPLATE,PT_CT_UPDATES), "PT_CT_UNET")
# CONFIGS = [
#     PT_T2_UNET,
#     T2_PT_UNET,
#     CT_PT_UNET,
#     PT_CT_UNET
# ]
if __name__ == '__main__':
    import sys
    print("Available configs:")
    for cfg in CONFIGS:
        print(f"{cfg.name}")
    sys.exit(0)
