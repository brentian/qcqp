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
import pandas as pd
import sys
from pyqp import bb_msc, bb_msc2, bb, bb_msc3
from pyqp.grb import *
from .. import max_cut

np.random.seed(1)

if __name__ == '__main__':
  pd.set_option("display.max_columns", None)
  try:
    n, backend, *_ = sys.argv[1:]
  except Exception as e:
    print("usage:\n"
          "python tests/-.py n (number of variables) backend")
    raise e
  verbose = False
  bool_use_shor = True
  evals = []
  params = bb_msc.BCParams()
  params.backend_name = backend

  # problem
  problem_id = f"max-cut:{n}:{0}"
  # start
  qp = max_cut.create_random_mc(int(n))

  # benchmark by gurobi
  r_grb_relax = qp_gurobi(qp, relax=True, sense="max", verbose=True,
                          params=params)
  eval_grb = r_grb_relax.eval(problem_id)

  # shor bb
  r_bb_shor = bb.bb_box(qp, verbose=verbose, params=params)
  eval_shor = r_bb_shor.eval(problem_id)

  # # b-b
  # r_bb_msc = bb_msc.bb_box(qp, verbose=verbose, params=params, bool_use_shor=bool_use_shor, rlt=True)
  # eval_bb = r_bb_msc.eval(problem_id)

  # b-b
  r_bb_msc3 = bb_msc3.bb_box(qp, verbose=verbose, params=params, bool_use_shor=bool_use_shor, rlt=True)
  eval_bb3 = r_bb_msc3.eval(problem_id)

  print(f"gurobi benchmark @{r_grb_relax.true_obj}")
  print(f"gurobi benchmark x\n"
        f"{r_grb_relax.xval.round(3)}")
  r_grb_relax.check(qp)

  print(f"branch-and-cut @{r_bb_msc3.true_obj}")
  print(f"branch-and-cut x\n"
        f"{r_bb_msc3.xval.round(3)}")
  r_bb_msc3.check(qp)

  # print(f"branch-and-cut @{r_bb2.true_obj}")
  # print(f"branch-and-cut x\n"
  #       f"{r_bb2.xval.round(3)}")
  # r_bb2.check(qp)

  evals += [
    {**eval_grb.__dict__, "method": "gurobi_relax", },
    {**eval_shor.__dict__, "method": "qcq_shor", },
    # {**eval_bb.__dict__, "method": "qcq_bb", },
    {**eval_bb3.__dict__, "method": "qcq_bb3", },
  ]

  df_eval = pd.DataFrame.from_records(evals)
  print(df_eval)
