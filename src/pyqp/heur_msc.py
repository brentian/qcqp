"""
Heuristic and other primal methods to extract a feasible solution
"""
import numpy as np
from .bg_msk import *


def penalty_method(r: MSKMscResult, qp: QP) -> MSKMscResult:
    decom = qp.decom_map
    model = r.problem
    y = r.Yvar
    obj = r.obj_expr

    delta_max = 1e3
    delta = 1e-1 * np.zeros((len(y), 2))
    delta[0, 0] = 1.1
    penalty_expr = 0

    y_shape = y[0].getShape()
    pos_expr = [expr.dot(decom[i][0].reshape(y_shape), yi) for i, yi in enumerate(y)]
    neg_expr = [expr.dot(decom[i][1].reshape(y_shape), yi) for i, yi in enumerate(y)]

    for i in range(qp.m + 1):
        penalty_expr = expr.add(
            penalty_expr,
            expr.add(
                expr.mul(pos_expr[i], delta[i][0]),
                expr.mul(neg_expr[i], delta[i][0]),
            )
        )

    model.objective(mf.ObjectiveSense.Maximize,
                    expr.sub(r.obj_expr, penalty_expr))

    r.solve()

    print(1)
