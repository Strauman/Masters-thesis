from .helpers.trainhelp import *
from .helpers import util
from .helpers.cprints import color_print
file_tb=util.caller_id(1,2, True)
import traceback

color_print("REMOVE trainhelp import", style="warning")
print("FROM:")
print(file_tb)
# print(traceback.extract_stack())
# print(traceback.extract_tb())
