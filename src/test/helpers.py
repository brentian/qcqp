import collections

import numpy as np
import pandas as pd

###########
# display options
###########


pd.set_option("display.max_columns", None)
np.set_printoptions(
  linewidth=200,
  precision=4
)

from pyqp import bg_grb, bg_msk, bg_msk_msc, bg_msk_chordal, bg_msk_homo
from pyqp import bb_msc, bb, bb_diag, bb_socp
from pyqp.classes import QP, QPI, Bounds, BCParams
import argparse
import json

METHODS = collections.OrderedDict([
  ("grb", bg_grb.qp_gurobi),
  ("shor", bg_msk.shor),
  ("dshor", bg_msk.dshor),
  ("msc", bg_msk_msc.msc),
  ("emsc", bg_msk_msc.msc_diag),
  ("ssdp", bg_msk_chordal.ssdp),
  ("bb", bb.bb_box),
  ("bb_msc", bb_diag.bb_box),
  ("shor_homo", bg_msk_homo.shor)
])

METHOD_CODES = {
  idx + 1: m
  for idx, m in enumerate(METHODS)
}

METHOD_HELP_MSGS = {
  k: bg_msk.dshor.__doc__
  for k, v in METHODS.items()
}

parser = argparse.ArgumentParser("QCQP runner")
parser.add_argument("--dump_instance", type=int, help="if save instance", default=1)
parser.add_argument("--r", type=str, help=METHOD_CODES.__str__(), default="1,2,7")
parser.add_argument("--fpath", type=str, help="path of the instance")
parser.add_argument("--n", type=int, help="dim of x", default=5)
parser.add_argument("--m", type=int, help="if randomly generated num of constraints", default=5)
parser.add_argument("--pc", type=str, help="if randomly generated problem type", default=5)
parser.add_argument("--time_limit", default=60, type=int, help="time limit of running.")
parser.add_argument("--verbose", default=0, type=int, help="if verbose")
parser.add_argument("--bg", default='msk', type=str, help="backend used")
parser.add_argument("--bg_rd", default=0, type=int, help="backend used, rank reduction")
