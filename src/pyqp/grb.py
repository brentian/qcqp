import itertools
import numpy as np

from .classes import Result, Params


def qp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max", relax=True, verbose=True, params: Params = Params(), **kwargs):
    """
    QCQP using Gurobi 9.1 as benchmark
    todo: works only for 1-d case now
        can we use Gurobi Matrix API?
    Parameters
    ----------
    Q
    q
    A
    ax
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
    ######################
    # === extra params
    ######################
    time_limit = params.time_limit
    import gurobipy as grb
    m, n, d = a.shape
    model = grb.Model()
    model.setParam(grb.GRB.Param.OutputFlag, verbose)

    indices = range(q.shape[0])

    if relax:
        x = model.addVars(indices, lb=lb.flatten(), ub=ub.flatten())
    else:
        x = model.addVars(indices, lb=lb.flatten(), ub=ub.flatten(), vtype=grb.GRB.INTEGER)

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
    model.setParam(grb.GRB.Param.TimeLimit, time_limit)

    model.setObjective(obj_expr, sense=(grb.GRB.MAXIMIZE if sense == 'max' else grb.GRB.MINIMIZE))
    model.optimize()
    r = Result()
    r.solve_time = model.Runtime
    r.bound = model.ObjBoundC
    r.relax_obj = model.ObjVal
    r.true_obj = model.ObjVal
    r.xval = np.array([i.x for i in x.values()]).reshape(q.shape)

    return r
