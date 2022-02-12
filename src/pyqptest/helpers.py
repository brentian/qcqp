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
from pyqp import bg_grb, bg_msk
from pyqp import bg_msk_msc, bg_msk_msc_admm
from pyqp import bg_msk_norm, bg_msk_norm_admm
from pyqp import bg_msk_mix
# branch and bound
from pyqp import bb, bb_diag, bb_nmsc, bb_mix
# helper, utilities, et cetera.
from pyqp.classes import QP, QPI, Bounds, BCParams, ADMMParams
from pyqp.classes import ctr, tr, DEBUG_BB

METHODS = collections.OrderedDict(
  [
    ("grb", bg_grb.qp_gurobi),  # exact benchmark by gurobi nonconvex qcqp
    # pure sdp
    ("shor", bg_msk.shor),  # relaxation: shor
    ("dshor", bg_msk.dshor),  # relaxation: shor-dual
    ("bb_sdp", bb.bb_box),
    # msc (many-small-cones)
    ("msc", bg_msk_msc.msc_diag),  # many small cone approach
    ("bb_msc_no_primal", bb_diag.bb_box),
    ("bb_msc", bb_diag.bb_box),
    ("admm_msc", bg_msk_msc_admm.msc_admm),
    # msc and socp using norm balls
    ("nsocp", bg_msk_norm.socp),
    ("bb_nmsc", bb_nmsc.bb_box),
    ("bb_nsocp", bb_nmsc.bb_box_nsocp),
    ("admm_nmsc", bg_msk_norm_admm.msc_admm),  # local method using admm
    # mixed-cone, todo.
    ("mcone", bg_msk_mix.socp),
    ("bb_mcone", bb_mix.bb_box_nsocp)
  ]
)

METHOD_CODES = {idx + 1: m for idx, m in enumerate(METHODS)}

METHOD_HELP_MSGS = {k: bg_msk.dshor.__doc__ for k, v in METHODS.items()}

QP_SPECIAL_PARAMS = {
  # relaxation
  "msc": {"decompose_method": "eig-type1", "force_decomp": True},
  "emsc": {"decompose_method": "eig-type2", "force_decomp": True},
  "asocp": {"decompose_method": "eig-type1", "force_decomp": False},
  # global method
  "bb_msc_no_primal": {"convexify_method": 1, "use_primal": False, "force_decomp": True},
  "bb_msc": {"convexify_method": 1, "use_primal": True, "force_decomp": True},
  # admm
  "admm_nmsc": {"verbose": True}
}

QP_RANDOM_INSTANCE_TYPE = {
  0: 'normal',
  1: 'rankr',
  2: 'bqp',
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
  "--bg", default='msk', type=str, help="backend used, e.g., mosek."
)
parser.add_argument(
  "--bg_pr", default=None, type=str, help="backend used, primal method"
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
