#  MIT License
#
#  Copyright (c) 2021 Cardinal Operations
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of
#  this software and associated documentation files (the "Software"), to deal in
#  the Software without restriction, including without limitation the rights to
#  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
#  of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import collections

import numpy as np
import pandas as pd
import sys

###########
# display options
###########


pd.set_option("display.max_columns", None)
np.set_printoptions(
  linewidth=200,
  precision=4
)

import pyqp.bg_msk_msc

from pyqp import bg_grb, bb, bg_msk, bg_msk_msc, bg_msk_chordal
from pyqp import bb_msc, bb_msc2
from pyqp.classes import QP, Bounds
import argparse
import json

methods = collections.OrderedDict({
  "grb": bg_grb.qp_gurobi,
  "shor": bg_msk.shor,
  "dshor": bg_msk.dshor,
  "msc": bg_msk_msc.msc,
  "emsc": bg_msk_msc.msc_diag,
  "ssdp": bg_msk_chordal.ssdp,
})

method_codes = {
  idx + 1: m
  for idx, m in enumerate(methods)
}

method_helps = {
  k: bg_msk.dshor.__doc__
  for k, v in methods.items()
}

np.random.seed(1)

parser = argparse.ArgumentParser("QCQP runner")
parser.add_argument("--fpath", type=str, help="path of the instance")
parser.add_argument("--dump_instance", type=int, help="if save instance", default=1)
parser.add_argument("--r", type=str, help=json.dumps(method_helps, indent=2), default="1,2,3")

if __name__ == '__main__':
  
  parser.print_usage()
  args = parser.parse_args()
  fpath = args.fpath
  r = map(int, args.r.split(","))
  r_methods = {method_codes[k] for k in r}
  verbose = True
  bool_use_shor = False
  evals = []
  params = bb_msc.BCParams()
  params.time_limit = 50
  kwargs = dict(
    relax=True,
    sense="max",
    verbose=verbose,
    params=params,
    bool_use_shor=bool_use_shor,
    rlt=True
  )
  
  # personal
  pkwargs = {k: {**kwargs} for k in r_methods}
  pkwargs_dtl = {
    "dshor": {**kwargs, "sense": "min"},
    "msc": {**kwargs, "decompose_method": "eig-type2"},
    "emsc": {**kwargs, "decompose_method": "eig-type2"},
    "socp": {**kwargs, "decompose_method": "eig-type2"},
  }
  pkwargs.update(pkwargs_dtl)
  pkwargs = {k: v for k, v in pkwargs.items() if k in r_methods}

  qp = QP.read(fpath)
  
  n, m = qp.n, qp.m
  # problem
  problem_id = f"{n}:{m}:{0}"
  # start
  
  bd = Bounds(xlb=np.zeros(shape=(n, 1)), xub=np.ones(shape=(n, 1)))
  
  evals = []
  results = {}
  # run methods
  for k in r_methods:
    func = methods[k]
    print(k, pkwargs[k])
    qp1 = bb_msc.QP(*qp.unpack())
    qp1.decompose(**pkwargs[k])
    r = func(qp1, bd, **pkwargs[k])
    reval = r.eval(problem_id)
    evals.append({**reval.__dict__, "method": k})
    results[k] = r
  
  for k, r in results.items():
    print(f"{k} benchmark @{r.relax_obj}")
    print(f"{k} benchmark x\n"
          f"{r.xval.round(3)}")
    r.check(qp)
  
  df_eval = pd.DataFrame.from_records(evals)
  print(df_eval)
  print(r.xval)
  print(df_eval[['prob_num', 'solve_time', 'relax_obj', 'method']].to_latex())
  
  if args.dump_instance:
    pass
