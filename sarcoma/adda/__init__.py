from pathlib import Path
import sys,os
# import pout as p
# import pout
# pout.inject()
# p.inject()

def syspaths(*paths):
    for p in paths: sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)),p)))
syspaths("../hlp","../records")


# records/ imports
import datasets,trainvalfunc

# hlp/ imports
# import util
from .helpers.cprints import color_print, warning_print
# adda/ files
from . import metrics
from .helpers.trainhelp import StopTraining,EarlyStopping,FinishedEpoch
from .helpers import util, trainhelp, ansi_print
