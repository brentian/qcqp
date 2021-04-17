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
import numpy as np
import itertools
import src
import pandas as pd

from .evaluation import *

if __name__ == '__main__':

    try:
        fp, n = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/qkp_soutif.py filepath n (number of variables)")
        raise e

    Q, q, A, a, b, sign, lb, ub = read_qkp_soutif(filepath=fp, n=int(n))

    x_grb, model_grb = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, relax=False, sense="max")
    x_grb_relax, model_grb_relax = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max")
    y_helberg, problem_helberg = qkp_helberg_sdp(Q, q, A, a, b, sign, lb, ub, solver='MOSEK')
    problem_qcqp2, y_qcqp2 = src.cvx_sdp(Q, q, A, a, b, sign, lb, ub, rel_type=2, solver='MOSEK')
    problem_qcqp1, (y_qcqp1, x_qcqp1) = src.cvx_sdp(Q, q, A, a, b, sign, lb, ub, rel_type=1, solver='MOSEK')

    obj_values = {
        "gurobi": model_grb.ObjVal,
        "gurobi_rel": model_grb_relax.ObjVal,
        "sdp_helberg": problem_helberg.value,
        "sdp_qcqp1": problem_qcqp1.value,
        "sdp_qcqp2": problem_qcqp2.value,
    }

    print(json.dumps(obj_values, indent=2))

    # # validate
    # # grb relaxation
    # xrelg = np.array([i.x for i in x_grb_relax.values()]).reshape((3, 1))
    # print(xrelg.T.dot(A[0]).dot(xrelg).trace() + xrelg.T.dot(a[0]).trace())
    # print(xrelg.T.dot(Q).dot(xrelg).trace() + xrelg.T.dot(q).trace())
    #
    # # sdp by method 1
    # Y, x = _
    # xrelqc = x.value
    # yrelqc = Y.value
    # print(np.abs(yrelqc.diagonal() - xrelqc.flatten()).max())
    # print(yrelqc.T.dot(A[0]).trace() + xrelqc.T.dot(a[0]).trace())
    # print((yrelqc.T @ Q).trace() + q.T.dot(xrelqc).trace())

    # evaluations
    eval_grb = evaluate(model_grb, x_grb)
    eval_grb_relax = evaluate(model_grb_relax, x_grb_relax)
    eval_qcqp1 = evaluate(problem_qcqp1, y_qcqp1, x_qcqp1)
    eval_qcqp2 = evaluate(problem_qcqp2, y_qcqp2, y_qcqp2)
    eval_helberg = evaluate(problem_helberg, y_helberg)

    evals = {
        "gurobi": eval_grb.__dict__,
        "gurobi_rel": eval_grb_relax.__dict__,
        "sdp_qcqp1": eval_qcqp1.__dict__,
        "sdp_qcqp2": eval_qcqp2.__dict__,
        "sdp_helberg": eval_helberg.__dict__,
    }

    df_eval = pd.DataFrame(evals)
    print(df_eval)


