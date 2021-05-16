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
import numpy as np
import sys

from pyqp import bb_msc, bb, grb

np.random.seed(1)

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    try:
        n, m, backend, *_ = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/random_bb.py n (number of variables) m (num of constraints)")
        raise e
    verbose = False
    evals = []
    params = bb.BCParams()
    params.backend_name = backend

    # problem
    problem_id = f"{n}:{m}:{0}"
    # start
    qp = bb.QP.create_random_instance(int(n), int(m))
    qp.decompose()

    # benchmark by gurobi
    r_grb_relax = grb.qp_gurobi(qp, relax=True, sense="max", verbose=True,
                                params=params)
    eval_grb = r_grb_relax.eval(problem_id)
    print(eval_grb.__dict__)

    # shor bb
    r_bb = bb.bb_box(qp, verbose=verbose, params=params)
    eval_bb = r_bb.eval(problem_id)

    # msc
    r_msc = bb_msc.bb_box(qp, verbose=verbose, params=params, bool_use_shor=True, rlt=True)
    eval_msc = r_msc.eval(problem_id)
    print(eval_msc.__dict__)

    print(f"gurobi benchmark @{r_grb_relax.true_obj}")
    print(f"gurobi benchmark x\n"
          f"{r_grb_relax.xval.round(3)}")

    r_grb_relax.check(qp)
    print(f"branch-and-cut @{r_bb.true_obj}")
    print(f"branch-and-cut x\n"
          f"{r_bb.xval.round(3)}")
    r_bb.check(qp)

    evals += [
        {**eval_grb.__dict__, "method": "gurobi_relax", "size": (n, m)},
        {**eval_bb.__dict__, "method": "qcq_bb", "size": (n, m)},
        {**eval_msc.__dict__, "method": "qcq_bb_msc", "size": (n, m)},
    ]

    df_eval = pd.DataFrame.from_records(evals)
    print(df_eval)
