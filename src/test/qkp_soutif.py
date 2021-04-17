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
import pandas as pd

from pyqp.bg_cvx import *


def read_qkp_soutif(filepath='', n=0):
    # the linear and quadratic constraints from the txt file
    size = n
    Q = np.zeros([size, size])
    capacity = 0
    # This follows the notation from Professor Ye's presentation
    q = np.zeros([size, 1])
    a = np.zeros([size, 1])
    row_counter = -2  # For cleaner indexing
    leq = 1
    with open(filepath, 'r') as f:
        for line in f:  # Read the following if statements in order. They are essentially doing f.readline()
            if row_counter < 0:
                print("Going past initial data...")
            elif row_counter == 0:  # Read the linear term in optimization function
                temp = line.strip('\n').split(' ')
                counter = 0
                for i in temp:
                    if i != '':
                        q[counter][0] = int(i)
                        counter += 1
            elif 0 < row_counter < size:  # Quadratic term in opt
                temp = line.strip('\n').split(' ')
                column_counter = row_counter
                for i in temp:
                    if i != '':
                        Q[row_counter - 1][column_counter] = int(i)
                        column_counter += 1
            elif row_counter == size:
                print("pass blank line")
            elif row_counter == size + 1:  # Define type of knapsack capacity equality. Assume always <= capacity
                leq = 1 - int(line.strip('\n'))
            elif row_counter == size + 2:  # Define knapsack capacity
                capacity = int(line.strip('\n'))
            elif row_counter == size + 3:  # Read linear terms in the constraint (knapsack capacity equation)
                temp = line.strip('\n').split(' ')
                counter = 0
                for i in temp:
                    if i != '':
                        a[counter][0] = int(i)
                        counter += 1
            else:
                print("we are done processing data")
            row_counter += 1

    # cvxpy backend
    #
    _xshape = (n, 1)
    lb, ub = np.zeros((n, 1)), np.ones((n, 1))
    return Q, q, np.zeros((1, n, n)), a.reshape((1, *_xshape)), \
           np.array([capacity]), np.array([leq]), lb.reshape(_xshape), ub.reshape(_xshape)


def qkp_helberg_sdp(Q, q, A, a, b, sign, lb, ub, solver="MOSEK", sense="max", **kwargs):
    """
    this method comes from:
    Helmberg C, Rendl F, Weismantel R (1996)
     Quadratic knapsack relaxations using cutting planes and semidefinite programming.
      International Conference on Integer Programming and Combinatorial Optimization. (Springer), 175–189.

    the relaxation is essentially about:
        x^2 = x since x is binary
    Parameters
    ----------
    Q
    q
    A
    a
    b
    sign
    lb
    ub
    solver
    sense
    kwargs

    Returns
    -------

    """
    _unused = kwargs
    m, n, d = a.shape

    Y = cvx.Variable((n, n), PSD=True)

    constrs = []

    for i in range(m):
        if sign[i] == 0:
            constrs += [cvx.trace(np.diag(a[i].flatten()) @ Y) <= b[i]]
        elif sign[i] == -1:
            constrs += [cvx.trace(np.diag(a[i].flatten()) @ Y) <= b[i]]
        else:
            constrs += [cvx.trace(np.diag(a[i].flatten()) @ Y) <= b[i]]
    # bounds
    constrs += [cvx.diag(Y) <= ub.flatten(),
                cvx.diag(Y) >= lb.flatten()]

    # objectives
    obj_expr = cvx.trace(Q @ Y) + cvx.trace(np.diag(q.flatten()) @ Y)
    obj_expr_cp = cvx.Maximize(obj_expr) if sense == 'max' else cvx.Minimize(
        obj_expr)

    problem = cvx.Problem(objective=obj_expr_cp, constraints=constrs)
    problem.solve(verbose=True, solver=solver, save_file="model.ptf")
    return Y, problem


def qkp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max", relax=True, verbose=True, **kwargs):
    """
    QCQP using Gurobi 9.1 as benchmark
    todo: works only for 1-d case now
        can we use Gurobi Matrix API?
    Parameters
    ----------
    Q
    q
    A
    a
    b
    sign
    lb
    ub
    sense
    relax
    kwargs

    Returns
    -------

    """
    import gurobipy as grb
    m, n, d = a.shape
    model = grb.Model()
    model.setParam(grb.GRB.Param.OutputFlag, verbose)

    indices = range(q.shape[0])

    if relax:
        x = model.addVars(indices, lb=lb.flatten(), ub=ub.flatten())
    else:
        x = model.addVars(indices, vtype=grb.GRB.BINARY)

    obj_expr = grb.quicksum(Q[i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices)) \
               + grb.quicksum(q[i][0] * x[i] for i in indices)
    for constr_num in range(m):
        if sign[constr_num] == 0:
            model.addConstr(
                grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
                + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
                == b[constr_num])

        elif sign[constr_num] == -1:
            model.addConstr(
                grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
                + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
                >= b[constr_num])

        else:
            model.addConstr(
                grb.quicksum(x[j] * a[constr_num][j][0] for j in indices)
                + grb.quicksum(A[constr_num][i][j] * x[i] * x[j] for i, j in itertools.product(indices, indices))
                <= b[constr_num])

    model.setParam(grb.GRB.Param.NonConvex, 2)
    model.setParam(grb.GRB.Param.TimeLimit, 500)

    model.setObjective(obj_expr, sense=(grb.GRB.MAXIMIZE if sense == 'max' else grb.GRB.MINIMIZE))
    model.optimize()
    r = Result()
    r.solve_time = model.Runtime
    r.bound = model.ObjBoundC
    r.relax_obj = model.ObjVal
    r.true_obj = model.ObjVal
    r.xval = np.array([i.x for i in x.values()]).reshape(q.shape)

    return r


if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    try:
        fp, n = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/qkp_soutif.py filepath n (number of variables)")
        raise e
    verbose = False

    # start
    Q, q, A, a, b, sign, lb, ub = read_qkp_soutif(filepath=fp, n=int(n))

    r_grb = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, relax=False, sense="max", verbose=verbose)
    r_grb_relax = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max", verbose=verbose)
    r_shor = shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)
    r_compact = compact_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)
    r_srlt = srlt_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)

    obj_values = {
        "gurobi": r_grb.true_obj,
        "gurobi_rel": r_grb_relax.true_obj,
        "sdp_qcqp1": r_shor.true_obj,
        "sdp_compact": r_compact.true_obj,
        "sdp_srlt": r_srlt.true_obj,
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
    prob_num = 0
    eval_grb = r_grb.eval(prob_num)
    eval_grb_relax = r_grb_relax.eval(prob_num)
    eval_qcqp1 = r_shor.eval(prob_num)
    eval_qcqp1_no_x = r_compact.eval(prob_num)
    eval_srlt = r_srlt.eval(prob_num)

    evals = [
        {**eval_grb.__dict__, "method": "gurobi"},
        {**eval_grb_relax.__dict__, "method": "gurobi_rel"},
        {**eval_qcqp1.__dict__, "method": "sdp_qcqp1"},
        {**eval_qcqp1_no_x.__dict__, "method": "sdp_compact"},
        {**eval_srlt.__dict__, "method": "sdp_srlt"},
    ]

    df_eval = pd.DataFrame.from_records(evals)
    print(df_eval)
