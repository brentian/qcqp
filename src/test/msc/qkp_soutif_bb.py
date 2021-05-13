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
from pyqp import grb, bb_msc, bb
from ..qkp_soutif import *

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    try:
        fp, n, backend = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/qkp_soutif.py filepath n (number of variables)")
        raise e
    verbose = False
    params = bb.BCParams()
    params.backend_name = backend
    params.opt_eps = 5e-3
    # start
    Q, q, A, a, b, sign, lb, ub = read_qkp_soutif(filepath=fp, n=int(n))
    qp = QP(Q, q, A, a, b, sign, lb, ub, lb @ lb.T, ub @ ub.T)

    # benchmark by gurobi
    r_grb_relax = grb.qp_gurobi(Q, q, A, a, b, sign, lb, ub, relax=True, sense="max", verbose=True,
                                params=params)
    print(f"gurobi benchmark @{r_grb_relax.true_obj}")
    print(f"gurobi benchmark x\n"
          f"{r_grb_relax.xval}")
    # b-b
    # r_bb = bb_msc.bb_box(qp, verbose=verbose, params=params, rlt=False)
    # print(f"branch-and-cut @{r_bb.true_obj}")
    # print(f"branch-and-cut x\n"
    #       f"{r_bb.xval.round(3)}")

    r_bb_rlt = bb_msc.bb_box(qp, verbose=verbose, params=params, rlt=True, bool_use_shor=True)

    print(f"branch-and-cut @{r_bb_rlt.true_obj}")
    print(f"branch-and-cut x\n"
          f"{r_bb_rlt.xval.round(3)}")
