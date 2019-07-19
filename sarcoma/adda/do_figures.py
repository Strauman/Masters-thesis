import os
from . import exporthead
from .exporting import export_helpers
from .exporting import filters
import sys
from sys import exit as xit
from .helpers.cprints import color_print
exporthead.confirm_recache()
if "--tikz" not in sys.argv:
    color_print("NB! Not exporting tikz! (--tikz not given)", style="notice")
if __name__ == '__main__':
    from .figures import demo_modalities, segmentation_output, adda_effect, boxplots
    box_perfs = ["F1_best","accuracy_best"]
    # box_perfs=["F1_optim", "accuracy_optim"]
    # box_perfs=["F1", "accuracy"]
    # demo_modalities.export_separate()
    # segmentation_output.export_segmented()
    # adda_effect.export(score_name=box_perfs[0])
    boxplots.export(box_perfs)
if "--tikz" not in sys.argv:
    color_print("NB! Not exporting tikz! (--tikz not given)", style="notice")
