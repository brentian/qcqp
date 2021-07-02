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
import numpy as np
import pandas as pd
import sys
from pyqp import grb, bb, bg_msk, bg_msk_ex
from pyqp import bb_msc, bb_msc2
from pyqp.classes import QPI

np.random.seed(1)

if __name__ == '__main__':
  pd.set_option("display.max_columns", None)
  try:
    n, m, pc, *_ = sys.argv[1:]
  except Exception as e:
    print("usage:\n"
          "python tests/random_bb.py n (number of variables) m (num of constraints)")
    raise e
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
  methods = {
    "grb": grb.qp_gurobi,
    "shor": bg_msk.shor,
    # "msc": bg_msk.msc,
    "emsc": bg_msk.msc_diag,
    # "emscsdp": bg_msk_ex.msc_diag_sdp,
    "ssdp": bg_msk_ex.ssdp,
    "ssdpblk": bg_msk_ex.ssdpblk
  }
  
  # personal
  pkwargs = {k: {**kwargs} for k in methods}
  pkwargs_dtl = {
    "emsc": {**kwargs, "decompose_method": "eig-type2", },
    # "emscsdp": {**kwargs, "decompose_method": "eig-type2", },
    # "msc_diag": {**kwargs, "decompose_method": "eig-typae2", "lk": False},
    # "socp": {**kwargs, "decompose_method": "eig-type2"},
  }
  pkwargs.update(pkwargs_dtl)
  # problem
  problem_id = f"{n}:{m}:{0}"
  # start
  # qp = QPI.block(int(n), int(m), r=10, eps=0.5)
  qp = QPI.normal(int(n), int(m), rho=0.2)
  
  evals = []
  results = {}
  # run methods
  for k, func in methods.items():
    print(k, pkwargs[k])
    qp1 = bb_msc.QP(*qp.unpack())
    qp1.decompose(**pkwargs[k])
    r = func(qp1, **pkwargs[k])
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
  print(df_eval[['prob_num', 'solve_time', 'relax_obj', 'method']].to_latex())
