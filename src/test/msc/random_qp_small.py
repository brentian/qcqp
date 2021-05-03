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
from pyqp.bb_msc import *
from pyqp.grb import *

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
    params = BCParams()
    params.backend_name = backend

    # problem
    problem_id = f"{n}:{m}:{0}"
    # start
    qp = QP.create_random_instance(int(n), int(m))
    Q, q, A, a, b, sign, lb, ub, ylb, yub, diagx = qp.unpack()

    #
    r_grb_relax = qp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max", verbose=verbose)
    r_shor = bg_cvx.shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)
    r_msc = bg_cvx.msc_relaxation(qp, solver='MOSEK', verbose=verbose)
    # r_msc_msk = bg_msk.msc_relaxation(qp, solver='MOSEK', verbose=verbose)
    obj_values = {
        "gurobi_rel": r_grb_relax.true_obj,
        "cvx_shor": r_shor.true_obj,
        "cvx_msc": r_msc.true_obj,
        # "msk_msc": r_msc_msk.true_obj,
    }

    r_msc.check(qp)
    print(json.dumps(obj_values, indent=2))
