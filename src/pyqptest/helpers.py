import collections
import argparse
import json
import copy

import numpy as np
import pandas as pd

###########
# display options
###########

pd.set_option("display.max_columns", None)
np.set_printoptions(linewidth=200, precision=3)

# relaxation
from pyqp import bg_grb, bg_msk, bg_msk_msc, bg_msk_admm, bg_msk_norm, bg_msk_mc
# branch and bound
from pyqp import bb, bb_diag, bb_nmsc, bb_mmsc
# helper, utilities, et cetera.
from pyqp.classes import QP, QPI, Bounds, BCParams
from pyqp.bg_msk_admm import ADMMParams

METHODS = collections.OrderedDict(
  [
    ("grb", bg_grb.qp_gurobi),  # exact benchmark by gurobi nonconvex qcqp
    # pure sdp
    ("shor", bg_msk.shor),  # relaxation: shor
    ("dshor", bg_msk.dshor),  # relaxation: shor-dual
    ("bb_sdp", bb.bb_box),
    # many small cones
    ("msc", bg_msk_msc.msc_diag),  # many small cone approach
    ("emsc", bg_msk_msc.msc_diag),
    ("bb_msc", bb_diag.bb_box),
    ("bb_nmsc", bb_nmsc.bb_box),
    ("admm_nmsc", bg_msk_admm.msc_admm),  # local method using admm
    # socp
    ("nsocp", bg_msk_norm.socp),
    ("bb_nsocp", bb_nmsc.bb_box_nsocp),
    
    # mixed-cone
    ("mcone", bg_msk_mc.socp),
    ("bb_mcone", bb_mmsc.bb_box_nsocp)
  ]
)

METHOD_CODES = {idx + 1: m for idx, m in enumerate(METHODS)}

METHOD_HELP_MSGS = {k: bg_msk.dshor.__doc__ for k, v in METHODS.items()}

QP_SPECIAL_PARAMS = {
  "msc": {"decompose_method": "eig-type1", "force_decomp": True},
  "emsc": {"decompose_method": "eig-type2", "force_decomp": True},
  "asocp": {"decompose_method": "eig-type1", "force_decomp": False}
}

QP_RANDOM_INSTANCE_TYPE = {
  0: 'normal',
  1: 'cvx',
  2: 'dnn',  # difference of doubly psd matrices
}

###################
# the argument parser
###################
parser = argparse.ArgumentParser("QCQP runner")
parser.add_argument(
  "--dump_instance", type=int, help="if save instance", default=1
)
parser.add_argument(
  "--r", type=str, help=f"solution method desc. \n {METHOD_CODES}", default="1,2,7"
)
parser.add_argument(
  "--fpath", type=str, help="path of the instance"
)
parser.add_argument(
  "--n", type=int, help="dim of x", default=5
)
parser.add_argument(
  "--m", type=int, help="if randomly generated num of constraints", default=5
)
parser.add_argument(
  "--time_limit", default=60, type=int, help="time limit of running."
)
parser.add_argument(
  "--verbose", default=0, type=int, help="if verbose"
)
parser.add_argument(
  "--bg", default='msk', type=str, help="backend used"
)
parser.add_argument(
  "--bg_rd", default=0, type=int, help="backend used, rank reduction"
)
parser.add_argument(
  "--problem_type", type=int, help=f"if randomly generated, what is the problem type?\n{QP_RANDOM_INSTANCE_TYPE}",
  default=0
)
parser.add_argument(
  "--problem_dtls",
  type=str,
  help="problem details"
)
parser.add_argument(
  "--bound_type", type=int, help=f"if randomly generated, what is the problem type?\n{QP_RANDOM_INSTANCE_TYPE}",
  default=0
)
parser.add_argument(
  "--bound_dtls", type=str,
  help=f"bound details",
)


