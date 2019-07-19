from . import exporthead as export
import os
export.confirm_recache()
def get_table(rows_func, table_head, torestore, save_to, **kwargs):
    table_foot = r"\end{tabular}"
    ROWS = ""
    for mf in torestore:
        ROWS += rows_func(*mf,**kwargs)
    out_str = table_head + ROWS + table_foot
    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(out_str)
    return out_str

def save_table(out_str,save_to):
    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(out_str)
    return out_str
if __name__ == '__main__':
    svhn_numbers = [
        [("68", "39")],
        # [("66", "37")],
        # [("66", "39")]
    ]
    MOD_FAVS = [  # ACTUAL FAVS
        ("CT_PT", (110, 21)),
        ("CT_T2", (1, 4)),
        ("PT_CT", (1, 5)),
        ("PT_T2", (78, 50)),
        ("T2_CT", ("0", "0")),
        ("T2_PT", (41, 72)),
    ]
    # MOD_FAVS = [  # ACTUAL FAVS
    #     ("PT_T2", (78, 50)),
    #     ("PT_CT", (1, 5)),
    #     ("CT_PT", (110, 21)),
    #     ("CT_T2", (1, 4)),
    #     ("T2_CT", ("0", "0")),
    #     ("T2_PT", (41, 72)),
    # ]
    SVHN_FAVS = [("SVHN_MNIST_LBL", ("68","39"), "SVHN_MNIST")]
    CONF_FAVS=MOD_FAVS
    # CONF_FAVS+=SVHN_FAVS
    from .tables import svhn_mnist
    from .tables import confusion
    from .tables import adda_overview
    from .tables import zero_portions
    from .tables import segmentation
    # adda_score="F1_optim"
    adda_score="F1_best"
    # adda_score="F1"
    tbl_out_root="/root/src/tex-tables/"
    tblpath=lambda path: os.path.join(tbl_out_root, path)
    # if adda_score=="F1_best":
        # adda_reset_out=tblpath("tbl-da-reset-thresh.tex")
    # else:
    adda_reset_out=tblpath("tbl-da-reset.tex")
    segmentation_out=tblpath("tbl-segmentation.tex")
    # print(save_table(out_str=segmentation.get_table(),save_to=segmentation_out))
    # print(get_table(adda_overview.ROWS_FN, table_head=adda_overview.TABLE_HEAD, torestore=MOD_FAVS, save_to=adda_reset_out, score=adda_score))
    # print(get_table(confusion.CONFUSION_ROWS, table_head=confusion.CONF_TABLE_HEAD, torestore=CONF_FAVS, save_to="/root/src/tex-tables/tbl-confusion.tex", score=adda_score))
    print(get_table(svhn_mnist.ROWS_FN_CONF_STYLE, table_head=svhn_mnist.TABLE_HEAD_CONF_STYLE, torestore=svhn_numbers, save_to="/root/src/tex-tables/tbl-svhn.tex"))
    print(get_table(svhn_mnist.ROWS_FN_DOWN_STYLE, table_head=svhn_mnist.TABLE_HEAD_DOWN_STYLE, torestore=svhn_numbers, save_to="/root/src/tex-tables/tbl-svhn-down.tex"))
    # print(get_table(svhn_mnist.ROWS_FN_DOWN_STYLE, table_head=svhn_mnist.TABLE_HEAD_DOWN_STYLE, torestore=svhn_numbers, save_to="/root/src/tex-tables/tbl-svhn-down.tex"))
    # save_table(out_str=zero_portions.get_table(), save_to="/root/src/tex-tables/tbl-zero-portions.tex")

    # print(get_ta)
