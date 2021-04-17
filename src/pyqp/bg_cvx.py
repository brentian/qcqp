import cvxpy as cvx
import numpy as np
from .classes import Result


class CVXResult(Result):
    def __init__(self, problem=None, yval=0, xval=0, tval=0, relax_obj=0, true_obj=0, bound=0, solve_time=0, xvar=None,
                 yvar=None):
        super().__init__(problem, yval, xval, tval, relax_obj, true_obj, bound, solve_time)
        self.xvar = xvar
        self.yvar = yvar
        self.solved = False


def cvx_sdp(*params, sense="max", rel_type=1, **kwargs):
    Q, q, A, a, b, sign, lb, ub = params

    if rel_type == 1:
        _ = shor_relaxation(Q, q, A, a, b, sign, lb, ub, sense="max", **kwargs)
    elif rel_type == 2:
        _ = compact_relaxation(Q, q, A, a, b, sign, lb, ub, sense="max", **kwargs)
    else:
        raise ValueError("no such SDP defined")
    return _


def qp_obj_func(Q, q, xval: np.ndarray):
    return xval.T.dot(Q).dot(xval).trace() + xval.T.dot(q).trace()


def shor_relaxation(Q, q, A, a, b, sign,
                    lb, ub,
                    ylb=None, yub=None,
                    solver="MOSEK", sense="max", verbose=True, solve=True, **kwargs):
    """
    use a Y along with x in the SDP
        for basic 0 <= x <= e, diag(Y) <= x
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
    xshape = (n, d)

    Y = cvx.Variable((n, n), PSD=True)
    x = cvx.Variable(xshape)

    # bounds
    constrs = [x <= ub, x >= lb]
    if ylb is not None:
        constrs += [Y >= ylb]
    if yub is not None:
        constrs += [Y <= yub]
    constrs += [cvx.bmat([[np.eye(d), x.T], [x, Y]]) >> 0]
    constrs += [cvx.diag(Y) <= x[:, 0]]
    for i in range(m):
        if sign[i] == 0:
            constrs += [cvx.trace(A[i].T @ Y) + cvx.trace(a[i].T @ x) == b[i]]
        elif sign[i] == -1:
            constrs += [cvx.trace(A[i].T @ Y) + cvx.trace(a[i].T @ x) >= b[i]]
        else:
            constrs += [cvx.trace(A[i].T @ Y) + cvx.trace(a[i].T @ x) <= b[i]]

    # objectives
    obj_expr = cvx.trace(Q @ Y) + cvx.trace(q.T @ x)
    obj_expr_cp = cvx.Maximize(obj_expr) if sense == 'max' else cvx.Minimize(
        obj_expr)

    r = CVXResult()
    r.xvar = x
    r.yvar = Y
    problem = cvx.Problem(objective=obj_expr_cp, constraints=constrs)
    r.problem = problem
    if not solve:
        return r

    problem.solve(verbose=verbose, solver=solver, save_file="model.ptf")
    xval = x.value
    r.yval = Y.value
    r.xval = xval
    r.relax_obj = problem.value
    r.true_obj = qp_obj_func(Q, q, xval)
    r.solved = True
    return r


def srlt_relaxation(Q, q, A, a, b, sign, lb, ub, solver="MOSEK", sense="max", verbose=True, **kwargs):
    """
    use a Y along with x in the SDP
        for basic 0 <= x <= e, diag(Y) = x
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
    xshape = (n, d)

    Y = cvx.Variable((n, n), PSD=True)
    x = cvx.Variable(xshape)

    # bounds
    constrs = [x <= ub, x >= lb]
    constrs += [cvx.bmat([[np.eye(d), x.T], [x, Y]]) >> 0]

    # using srlt
    ones = np.ones(shape=xshape)
    constrs += [Y >= 0]
    constrs += [Y + np.ones(shape=Y.shape) - x @ ones.T - ones @ x.T >= 0]
    constrs += [x @ ones.T - Y >= 0]

    for i in range(m):
        if sign[i] == 0:
            constrs += [cvx.trace(A[i].T @ Y) + cvx.trace(a[i].T @ x) == b[i]]
        elif sign[i] == -1:
            constrs += [cvx.trace(A[i].T @ Y) + cvx.trace(a[i].T @ x) >= b[i]]
        else:
            constrs += [cvx.trace(A[i].T @ Y) + cvx.trace(a[i].T @ x) <= b[i]]

    # objectives
    obj_expr = cvx.trace(Q @ Y) + cvx.trace(q.T @ x)
    obj_expr_cp = cvx.Maximize(obj_expr) if sense == 'max' else cvx.Minimize(
        obj_expr)

    problem = cvx.Problem(objective=obj_expr_cp, constraints=constrs)
    problem.solve(verbose=verbose, solver=solver, save_file="model.ptf")
    xval = x.value

    r = CVXResult()
    r.problem = problem
    r.xvar = x
    r.yvar = Y
    r.yval = Y.value
    r.xval = x.value
    r.relax_obj = problem.value
    r.true_obj = qp_obj_func(Q, q, xval)
    return r


def compact_relaxation(Q, q, A, a, b, sign, lb, ub, solver="MOSEK", sense="max", verbose=True, **kwargs):
    """
     use a n+1 dimensional PSD matrix Y alone,
        without declare x in the SDP
        for basic -e <= x <= e
    todo 1. check if this work for matrix X
    todo 2. how to handle box constraints?

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
    xshape = (n, d)

    Y = cvx.Variable((n + 1, n + 1), PSD=True)
    _Q = np.bmat([[Q, q / 2], [q.T / 2, np.zeros((1, 1))]])
    constrs = []
    for i in range(m):
        # build block matrix
        _A = np.bmat([[A[i], a[i] / 2], [a[i].T / 2, np.zeros((1, 1))]])
        if sign[i] == 0:
            constrs += [cvx.trace(_A.T @ Y) == b[i]]
        elif sign[i] == -1:
            constrs += [cvx.trace(_A.T @ Y) >= b[i]]
        else:
            constrs += [cvx.trace(_A.T @ Y) <= b[i]]
    # boxes
    _Y_flatten = cvx.diag(Y)
    constrs += [_Y_flatten[:-1] >= lb.flatten(),
                _Y_flatten[:-1] <= ub.flatten(),
                _Y_flatten[-1] >= -1,
                _Y_flatten[-1] <= 1]
    obj_expr = cvx.trace(_Q @ Y)
    obj_expr_cp = cvx.Maximize(obj_expr) if sense == 'max' else cvx.Minimize(
        obj_expr)
    problem = cvx.Problem(objective=obj_expr_cp, constraints=constrs)
    problem.solve(verbose=verbose, solver=solver, save_file="model.ptf")
    xtval = np.sqrt(_Y_flatten.value)
    xval = xtval[:-1].reshape(xshape)

    r = CVXResult()
    r.problem = problem
    r.xvar = x
    r.yvar = Y
    r.yval = Y.value
    r.xval = xval
    r.relax_obj = problem.value
    r.true_obj = qp_obj_func(Q, q, xval)
    return r
