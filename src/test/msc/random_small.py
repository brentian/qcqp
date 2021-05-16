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
from pyqp import grb

np.random.seed(1)

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    try:
        n, m, *_ = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/random_bb.py n (number of variables) m (num of constraints)")
        raise e
    verbose = False
    evals = []
    params = BCParams()

    # problem
    problem_id = f"{n}:{m}:{0}"
    # start
    qp = QP.create_random_instance(int(n), int(m))
    qp.decompose()
    #
    r_grb_relax = grb.qp_gurobi(qp, sense="max", verbose=verbose)
    r_cvx_shor = bg_cvx.shor_relaxation(qp, solver='MOSEK', verbose=verbose)
    r_shor = bg_msk.shor_relaxation(qp, solver='MOSEK', verbose=verbose)
    r_msc = bg_msk.msc_relaxation(qp, bounds=None, solver='MOSEK', verbose=verbose)
    r_msc_msk = bg_msk.msc_relaxation(qp, bounds=None, solver='MOSEK', verbose=verbose, with_shor=r_shor)
    r_msc_msk2 = bg_msk.msc_relaxation(qp, bounds=None, solver='MOSEK', verbose=verbose, constr_d=True)
    r_msc_msk3 = bg_msk.msc_relaxation(qp, bounds=None, solver='MOSEK', verbose=verbose, rlt=True)

    obj_values = {
        "gurobi_rel": r_grb_relax.relax_obj,
        "cvx_shor": r_cvx_shor.relax_obj,
        "msk_shor": r_shor.relax_obj,
        "msk_msc": r_msc.relax_obj,
        "msk_msc_with_shor": r_msc_msk.relax_obj,
        "msk_msc_with_d": r_msc_msk2.relax_obj,
        "msk_msc_with_rlt": r_msc_msk3.relax_obj,
    }

    r_msc_msk.check(qp)
    print(json.dumps(obj_values, indent=2))
