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

from .classes import Result, qp_obj_func, QP, MscBounds


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


class MSKMscResult(MSKResult):
    def __init__(self, problem=None, yval=0, xval=0, tval=0, relax_obj=0, true_obj=0, bound=0, solve_time=0, xvar=None,
                 yvar=None):
        super().__init__(problem, yval, xval, tval, relax_obj, true_obj, bound, solve_time, xvar, yvar)
        self.zvar = None
        self.zval = None
        self.Zvar = None
        self.Yvar = None
        self.Yval = None
        self.Zval = None
        self.solved = False
        self.obj_expr = None
        self.qel = None
        self.q = None

    def solve(self, verbose=True):
        try:
            self.problem.solve()
            status = self.problem.getProblemStatus()
        except Exception as e:
            status = 'failed'
        if status == mf.ProblemStatus.PrimalAndDualFeasible:
            self.yval = self.yvar.level().reshape(self.yvar.getShape()).round(4)
            self.xval = self.xvar.level().reshape(self.xvar.getShape()).round(4)
            self.zval = self.zvar.level().reshape(self.zvar.getShape()).round(4)
            self.Yval = np.hstack([xx.level().reshape(self.xvar.getShape()).round(4) for xx in self.Yvar])
            self.Zval = np.hstack([xx.level().reshape(self.xvar.getShape()).round(4) for xx in self.Zvar])
            self.relax_obj = self.qel.T.dot(self.yval).trace() + self.q.T.dot(self.xval).trace()
        else:  # infeasible
            self.relax_obj = -1e6
        self.solved = True


def shor_relaxation(
        Q, q, A, a, b, sign,
        lb, ub,
        ylb=None, yub=None,
        diagx=None,
        solver="MOSEK", sense="max", verbose=True, relax=False, solve=True, **kwargs
):
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


def msc_relaxation(
        qp: QP, bounds: MscBounds = None, sense="max", verbose=True, solve=True,
        *args, **kwargs
):
    """
    The many-small-cone approach
    Returns
    -------
    """
    _unused = kwargs
    Q, q, A, a, b, sign, *_ = qp.unpack()
    if qp.Qpos is None:
        qp.decompose()
    m, n, d = a.shape
    xshape = (n, d)
    model = mf.Model('many_small_cone_msk')

    if verbose:
        model.setLogHandler(sys.stdout)

    if bounds is None:
        bounds = MscBounds.construct(qp)

    zlb = bounds.zlb
    zub = bounds.zub

    qpos, qipos = qp.Qpos
    qneg, qineg = qp.Qneg
    qel = np.ones([*xshape])
    qel[qineg] = -1
    x = model.variable("x", [*xshape], dom.inRange(qp.lb, qp.ub))

    zcone = model.variable("zc", dom.inPSDCone(2, n))
    y = zcone.slice([0, 0, 0], [n, 1, 1]).reshape([n, 1])
    z = zcone.slice([0, 0, 1], [n, 1, 2]).reshape([n, 1])
    Y = [y]
    Z = [z]

    # bounds

    model.constraint(z, dom.inRange(zlb[0], zub[0]))
    if bounds.ylb is not None:
        pass
    if bounds.yub is not None:
        model.constraint(y, dom.lessThan(bounds.yub[0]))
    else:
        model.constraint(y, dom.lessThan(1e5))
    #
    model.constraint(
        expr.sub(
            expr.mul((qneg + qpos).T, x),
            z), dom.equalsTo(0))
    for idx in range(n):
        model.constraint(zcone.index([idx, 1, 1]), dom.equalsTo(1))
    # y^Te \le (q @ q.T) > 0
    qqpos = qpos @ qpos.T
    qqneg = qneg @ qneg.T
    yposs = expr.sum(y.pick([[j, 0] for j in qipos]))
    ynegs = expr.sum(y.pick([[j, 0] for j in qineg]))
    model.constraint(
        yposs, dom.lessThan((qqpos * (qqpos> 0)).sum())
    )
    model.constraint(
        ynegs, dom.lessThan((qqneg * (qqneg> 0)).sum())
    )
    model.constraint(
        expr.sub(yposs, ynegs), dom.lessThan(Q.sum())
    )

    for i in range(m):
        apos, ipos = qp.Apos[i]
        aneg, ineg = qp.Aneg[i]
        quad_expr = expr.sub(expr.dot(a[i], x), b[i])
        # yub = np.max([(apos.T @ qp.ub) ** 2, (apos.T @ qp.lb) ** 2], axis=0)
        # yub += np.max([(aneg.T @ qp.ub) ** 2, (aneg.T @ qp.lb) ** 2], axis=0)
        # yub = np.max([zub[i + 1] ** 2, zlb[i + 1] ** 2], axis=0)
        if ipos.shape[0] + ineg.shape[0] > 0:
            zconei = model.variable(f"zci@{i}", dom.inPSDCone(2, n))
            yi = zconei.slice([0, 0, 0], [n, 1, 1]).reshape([n, 1])
            zi = zconei.slice([0, 0, 1], [n, 1, 2]).reshape([n, 1])
            Y.append(yi)
            Z.append(zi)
            # create a mat
            el = np.ones([n, 1])
            el[ineg] = -1
            # bounds
            model.constraint(zi, dom.inRange(zlb[i + 1], zub[i + 1]))
            if bounds.yub is not None:
                model.constraint(yi, dom.lessThan(bounds.yub[i + 1]))
            #
            for idx in range(n):
                model.constraint(zconei.index([idx, 1, 1]), dom.equalsTo(1))
            model.constraint(
                expr.sub(
                    expr.mul((apos + aneg).T, x),
                    zi), dom.equalsTo(0))
            quad_expr = expr.add(quad_expr, expr.dot(el, yi))

        else:
            yi = np.zeros(xshape)
            zi = np.zeros(xshape)
        if sign[i] == 0:
            model.constraint(
                quad_expr,
                dom.equalsTo(0))
        elif sign[i] == -1:
            model.constraint(
                quad_expr,
                dom.greaterThan(0))
        else:
            model.constraint(
                quad_expr,
                dom.lessThan(0))

    # objectives
    penalized_obj_expr = expr.mul(1.0, expr.sum(expr.vstack(Y[1:]))) if len(Y) > 1 else 0
    true_obj_expr = expr.add(expr.dot(q, x), expr.dot(qel, y))
    obj_expr = expr.sub(true_obj_expr, penalized_obj_expr)
    # obj_expr = true_obj_expr
    model.objective(mf.ObjectiveSense.Minimize
                    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)

    r = MSKMscResult()
    r.obj_expr = true_obj_expr
    r.xvar = x
    r.yvar = y
    r.zvar = z
    r.Zvar = Z
    r.Yvar = Y
    r.qel = qel
    r.q = q
    r.problem = model
    if not solve:
        return r

    r.solve(verbose=verbose)

    return r
