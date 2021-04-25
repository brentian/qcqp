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

import itertools
import sys

import gurobipy as grb

from pyqp.bb import *
from pyqp.classes import Result
from pyqp.grb import qp_gurobi
from pyqp import bg_msk, bg_cvx


def read_maxcut(filepath=''):
    with open(filepath, 'r') as f:
        content = f.readlines()
        n, lines = [int(i) for i in content[0].strip("\n").strip(r" ").split(r" ")]
        data = [[int(i) for i in line.strip("\n").strip(r" ").split(r" ")]
                for line in content[1:]]
    Q = np.zeros((n, n), dtype=np.int8)
    for i, j, v in data:
        Q[i - 1, j - 1] = v
        Q[j - 1, i - 1] = v
    return Q / 2, np.zeros([n, 1]), \
           np.zeros((0, n, n)), np.zeros([0, n, 1]), np.array([]), np.array([]), \
           -np.ones([n, 1]), np.ones([n, 1])


# def maxcut_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max", relax=True, **kwargs):
#     m, n, d = a.shape
#     model = grb.Model()
#     indices = range(q.shape[0])
#
#     if relax:
#         x = model.addVars(indices, lb=lb.flatten(), ub=ub.flatten())
#     else:
#         x = model.addVars(indices, lb=lb.flatten(), ub=ub.flatten(), vtype=grb.GRB.INTEGER)
#
#     obj_expr = grb.quicksum(Q[i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices)) \
#                + grb.quicksum(q[i][0] * x[i] for i in indices)
#     for constr_num in range(m):
#         if sign[constr_num] == 0:
#             model.addConstr(
#                 grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
#                 + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
#                 == b[constr_num])
#
#         elif sign[constr_num] == -1:
#             model.addConstr(
#                 grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
#                 + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
#                 >= b[constr_num])
#
#         else:
#             model.addConstr(
#                 grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
#                 + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
#                 <= b[constr_num])
#
#     model.setParam(grb.GRB.Param.NonConvex, 2)
#     model.setParam(grb.GRB.Param.TimeLimit, 100)
#     # model.setParam(grb.GRB.Param.MIPGap, 0.05)
#     model.setObjective(obj_expr, sense=(grb.GRB.MAXIMIZE if sense == 'max' else grb.GRB.MINIMIZE))
#     model.optimize()
#     r = Result()
#     r.solve_time = model.Runtime
#     r.bound = (Q.sum() - model.ObjBoundC) / 2
#     r.relax_obj = (Q.sum() - model.ObjVal) / 2
#     r.true_obj = (Q.sum() - model.ObjVal) / 2
#     r.xval = np.array([i.x for i in x.values()]).reshape(q.shape)
#
#     return r


if __name__ == '__main__':

    try:
        fp, *_ = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/qkp_soutif.py filepath")
        raise e

    verbose = True
    params = BCParams()
    params.time_limit = 100

    Q, q, A, a, b, sign, lb, ub = read_maxcut(filepath=fp)
    qp = QP(Q, q, A, a, b, sign, lb, ub, 0, None)
    r_grb = qp_gurobi(Q, q, A, a, b, sign, lb, ub, relax=True, sense="min", verbose=verbose,
                      params=params)
    eval_grb = r_grb.eval(1)

    r_shor = bg_msk.shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', sense='min', verbose=verbose)
    eval_shor = r_shor.eval(1)
    print(eval_grb.__dict__)
    print(eval_shor.__dict__)
    # r_bb = bb_box(qp, verbose=verbose, params=params)
