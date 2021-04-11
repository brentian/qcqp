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

import sys
import json
import cvxpy as cvx
import numpy as np

import qcqp
from test.qkp_soutif import qkp_gurobi, read_qkp_soutif, qkp_helberg_sdp
from test.evaluation import *

if __name__ == '__main__':

    try:
        fp, n, m = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/qkp_soutif.py filepath n:(number of variables)")
        raise e

    Q, q, A, a, b, sign, lb, ub = read_qkp_soutif(filepath=fp, n=int(n))

    # add a nonzero A to try qcqp
    np.random.seed(1)
    A = np.random.random_integers(0, 100, (1, int(n), int(n)))

    x_grb, model_grb = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, relax=False, sense="max")
    x_grb_relax, model_grb_relax = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max")
    Y_helberg, problem_helberg = qkp_helberg_sdp(Q, q, A, a, b, sign, lb, ub, solver='MOSEK')

    problem_qcqp2, y_qcqp2 = qcqp.cvx_sdp(Q, q, A, a, b, sign, lb, ub, rel_type=2, solver='MOSEK')
    problem_qcqp1, (y_qcqp1, x_qcqp1) = qcqp.cvx_sdp(Q, q, A, a, b, sign, lb, ub, rel_type=1, solver='MOSEK')

    obj_values = {
        "gurobi": model_grb.ObjVal,
        "gurobi_rel": model_grb_relax.ObjVal,
        "sdp_helberg": problem_helberg.value,
        "sdp_qcqp1": problem_qcqp1.value,
        "sdp_qcqp2": problem_qcqp2.value,
    }

    print(json.dumps(obj_values, indent=2))



