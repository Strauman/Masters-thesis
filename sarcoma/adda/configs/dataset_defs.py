from . import ADDA_CFG_TEMPLATE
from .adda_template import img_size
T2_PT = lambda: ADDA_CFG_TEMPLATE(
    source_ds_name="T2",
    target_ds_name="PT",
    source_dataset=f"T2_ONLY_{img_size}",
    target_dataset=f"PT_ONLY_{img_size}",
    model_save_subdir=f"T2_PT_{img_size}"
)
T2_CT = lambda: ADDA_CFG_TEMPLATE(
    source_ds_name="T2",
    target_ds_name="CT",
    source_dataset=f"T2_ONLY_{img_size}",
    target_dataset=f"CT_ONLY_{img_size}",
    model_save_subdir=f"T2_CT_{img_size}"
)
PT_T2 = lambda: ADDA_CFG_TEMPLATE(
    source_ds_name="PT",
    target_ds_name="T2",
    source_dataset=f"PT_ONLY_{img_size}",
    target_dataset=f"T2_ONLY_{img_size}",
    model_save_subdir=f"PT_T2_{img_size}"
)
PT_CT = lambda: ADDA_CFG_TEMPLATE(
    source_ds_name="PT",
    target_ds_name="CT",
    source_dataset=f"PT_ONLY_{img_size}",
    target_dataset=f"CT_ONLY_{img_size}",
    model_save_subdir=f"PT_CT_{img_size}"
)
CT_PT = lambda: ADDA_CFG_TEMPLATE(
    source_ds_name="CT",
    target_ds_name="PT",
    source_dataset=f"CT_ONLY_{img_size}",
    target_dataset=f"PT_ONLY_{img_size}",
    model_save_subdir=f"CT_PT_{img_size}"
)
CT_T2 = lambda: ADDA_CFG_TEMPLATE(
    source_ds_name="CT",
    target_ds_name="T2",
    source_dataset=f"CT_ONLY_{img_size}",
    target_dataset=f"T2_ONLY_{img_size}",
    model_save_subdir=f"CT_T2_{img_size}"
)
