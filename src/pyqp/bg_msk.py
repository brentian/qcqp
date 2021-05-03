import numpy as np
import sys

try:
    import mosek.fusion as mf

    expr = mf.Expr
    dom = mf.Domain
    mat = mf.Matrix
except Exception as e:
    import logging

    logging.exception(e)

from .classes import Result, qp_obj_func, QP


class MSKResult(Result):
    def __init__(self, problem: mf.Model = None, yval=0, xval=0, tval=0, relax_obj=0, true_obj=0, bound=0, solve_time=0,
                 xvar: mf.Variable = None,
                 yvar: mf.Variable = None,
                 zvar: mf.Variable = None):
        super().__init__(problem, yval, xval, tval, relax_obj, true_obj, bound, solve_time)
        self.xvar = xvar
        self.yvar = yvar
        self.zvar = zvar
        self.solved = False

    def solve(self):
        self.problem.solve()
        self.yval = self.yvar.level().reshape(self.yvar.getShape())
        self.xval = self.xvar.level().reshape(self.xvar.getShape())
        self.relax_obj = self.problem.primalObjValue()
        self.solved = True


def shor_relaxation(Q, q, A, a, b, sign,
                    lb, ub,
                    ylb=None, yub=None,
                    diagx=None,
                    solver="MOSEK", sense="max", verbose=True, relax=False, solve=True, **kwargs):
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
    model = mf.Model('shor_msk')

    if verbose:
        model.setLogHandler(sys.stdout)
    # if relax:
    #     x = model.variable([n, d], dom.inRange(0, 1))
    #     Y = model.variable([n, n], dom.inPSDCone(n))
    # else:
    #     x = model.variable([n, d], dom.binary())
    #     Y = model.variable([n, n], dom.inPSDCone(n))
    #     model.setSolverParam("mioTolRelGap", 0.1)
    #     # model.setSolverParam("mioMaxTime", params.time_limit)
    #     model.setSolverParam("mioMaxNumSolutions", 20)
    #     model.acceptedSolutionStatus(mf.AccSolutionStatus.Feasible)

    Z = model.variable("Z", dom.inPSDCone(n + 1))
    Y = Z.slice([0, 0], [n, n])
    x = Z.slice([0, n], [n, n + 1])
    model.constraint(Y, dom.inPSDCone(n))
    # bounds
    # constrs = [x <= ub, x >= lb]
    if diagx is None:
        model.constraint(expr.sub(x, ub), dom.lessThan(0))
        model.constraint(expr.sub(x, lb), dom.greaterThan(0))
    if ylb is not None:
        model.constraint(expr.sub(Y, ylb), dom.greaterThan(0))
    if yub is not None:
        model.constraint(expr.sub(Y, yub), dom.lessThan(0))

    # constrs += [cvx.diag(Y) <= x[:, 0]]
    model.constraint(expr.sub(Y.diag(), x), dom.lessThan(0))
    # constrs += [cvx.bmat([[np.eye(d), x.T], [x, Y]]) >> 0]
    model.constraint(Z.index(n, n), dom.equalsTo(1.))
    for i in range(m):
        if sign[i] == 0:
            # constrs += [cvx.trace(A[i].T @ Y) + cvx.trace(a[i].T @ x) == b[i]]
            model.constraint(
                expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
                dom.equalsTo(b[i]))
        elif sign[i] == -1:
            model.constraint(
                expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
                dom.greaterThan(b[i]))
        else:
            model.constraint(
                expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
                dom.lessThan(b[i]))

    # objectives
    obj_expr = expr.add(expr.sum(expr.mulElm(Q, Y)), expr.dot(x, q))
    model.objective(mf.ObjectiveSense.Minimize
                    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)

    r = MSKResult()
    r.xvar = x
    r.yvar = Y
    r.zvar = Z
    r.problem = model
    if not solve:
        return r

    model.solve()
    xval = x.level().reshape(xshape)
    r.yval = Y.level().reshape((n, n))
    r.xval = xval
    r.relax_obj = model.primalObjValue()
    r.true_obj = qp_obj_func(Q, q, xval)
    r.solved = True
    r.solve_time = model.getSolverDoubleInfo("optimizerTime")

    return r

def msc_relaxation(qp: QP, solver="MOSEK", sense="max", verbose=True, solve=True, *args, **kwargs):
    """
    The many-small-cone approach
    Returns
    -------
    """
    _unused = kwargs
    Q, q, A, a, b, sign, lb, ub, ylb, yub, diagx = qp.unpack()
    if qp.Qpos is None:
        qp.decompose()
    m, n, d = a.shape
    xshape = (n, d)
    model = mf.Model('shor_msk')

    if verbose:
        model.setLogHandler(sys.stdout)
    qpos, ipos = qp.Qpos
    qneg, ineg = qp.Qneg
    # we construct bounds for ypos, yneg
    yposub = np.max([(qpos.T @ ub)**2, (qpos.T @ lb)**2], axis=0)
    ynegub = np.max([(qneg.T @ ub)**2, (qneg.T @ lb)**2], axis=0)

    x = model.variable("x", [*xshape], dom.inRange(lb, ub))
    y = model.variable("y+", [*xshape], dom.greaterThan(0))
    z = model.variable("z+", [*xshape], dom.unbounded())
    mpos = model.variable("m+", dom.inPSDCone(2, n))
    # if ipos.shape[0] > 0:
    #     model.constraint(expr.sub(expr.mul(qpos.T, x), zpos), dom.equalsTo(0))
    #     for idx in ipos:
    #         model.constraint(ypos.index(idx, 0), dom.lessThan(yposub[idx, 0]))
    #         model.constraint(yneg.index(idx, 0), dom.lessThan(0))
    #         model.constraint(mpos.index(idx, 0, 0))
    # if ineg.shape[0] > 0:
    #     model.constraint(expr.sub(expr.mul(qneg.T, x), zneg), dom.equalsTo(0))
    #     for idx in ineg:
    #         model.constraint(yneg.index(idx, 0), dom.lessThan(ynegub[idx, 0]))
    #         model.constraint(ypos.index(idx, 0), dom.lessThan(0))
