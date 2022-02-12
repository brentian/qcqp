"""
Test scripts using SDP based relaxation, i.e.,
- Lift xx^T to X ⪰ 0
"""
import numpy as np
import sys
import time
from .primal_rd import PRIMAL_METHOD_ID

try:
  import mosek.fusion as mf
  
  expr = mf.Expr
  dom = mf.Domain
  mat = mf.Matrix
except Exception as e:
  import logging
  
  logging.exception(e)

from .classes import Result, qp_obj_func, QP, Bounds


class MSKResult(Result):
  def __init__(self,
               qp: QP = None,
               problem: mf.Model = None,
               xvar: mf.Variable = None,
               yvar: mf.Variable = None,
               zvar: mf.Variable = None):
    super().__init__(problem)
    self.qp = qp
    self.xvar = xvar
    self.yvar = yvar
    self.zvar = zvar
    self.xval = 0
    self.yval = 0
    self.xb = 0
    self.yb = 0
    self.res = 0
    self.res_norm = 0
    self.solved = False
  
  def solve(self, primal=0, feas_eps=1e-4):
    self.problem.solve()
    self.xval = self.xvar.level().reshape(self.xvar.getShape())
    self.yval = self.yvar.level().reshape(self.yvar.getShape())
    self.relax_obj = self.bound = self.problem.primalObjValue()
    self.solved = True
    self.unit_time = self.problem.getSolverDoubleInfo("optimizerTime")
    self.solve_time = time.time() - self.start_time
    self.res = np.abs(self.yval - self.xval @ self.xval.T)
    self.res_norm = self.res.max()
    # derive primal solution
    # if it is infeasible and primal method is assigned.
    if self.res_norm > feas_eps and primal != 0:
      func = PRIMAL_METHOD_ID[primal]
      # produce primal xb yb
      func(self)
    else:
      self.xb = self.xval
      self.yb = self.yval
    self.true_obj = qp_obj_func(self.qp.Q, self.qp.q, self.xb)


def shor(qp: QP,
         bounds: Bounds,
         sense="max",
         verbose=True,
         solve=True,
         r_parent: MSKResult = None,
         **kwargs):
  """
  use a Y along with x in the SDP
      for basic 0 <= x <= e, diag(Y) <= x
  Parameters
  ----------

  Returns
  -------

  """
  _unused = kwargs
  st_time = time.time()
  Q, q, A, a, b, sign, *_ = qp.unpack()
  lb, ub, ylb, yub, *_ = bounds.unpack()
  m, n, d = a.shape
  xshape = (n, d)
  model = mf.Model('shor_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  Y = Z.slice([0, 0], [n, n])
  x = Z.slice([0, n], [n, n + 1])
  
  # bounds
  model.constraint(expr.sub(Y.diag(), expr.mulElm(ub.flatten(), expr.flatten(x))), dom.lessThan(0))
  model.constraint(Z.index(n, n), dom.equalsTo(1.))
  if bounds.sphere:
    model.constraint(expr.sum(Y.diag()), dom.lessThan(bounds.sphere**2))
  
  for i in range(m):
    if sign[i] == 0:
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
  #
  # objectives
  obj_expr = expr.add(expr.sum(expr.mulElm(Q, Y)), expr.dot(x, q))
  model.objective(
    mf.ObjectiveSense.Minimize
    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
  model.setSolverParam("intpntSolveForm", "dual")
  r = MSKResult(qp, model, x, Y, Z)
  r.start_time = st_time
  r.build_time = time.time() - st_time
  if not solve:
    return r
  r.solve()
  if verbose:
    print(r.build_time, r.total_time, r.solve_time)
  
  return r


def dshor(qp: QP,
          bounds: Bounds,
          sense="min",
          verbose=True,
          solve=True,
          **kwargs):
  """
  dual form of SDP relaxation
    should be equal to primal form of
    the SDP.
  @note,
    simply test if I did the Lagrangians right : )
  Parameters
  -------

  """
  _unused = kwargs
  Q, q, A, a, b, sign = qp.unpack()
  lb, ub, ylb, yub = bounds.unpack()
  m, n, d = a.shape
  xshape = (n, d)
  model = mf.Model('shor_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  Y = Z.slice([0, 0], [n, n])
  y = Z.slice([0, n], [n, n + 1])
  
  # lambda, mu, v
  l = model.variable("l", [m], dom.greaterThan(0))
  mu = model.variable("m", [n], dom.greaterThan(0))
  v = model.variable("v", [n], dom.greaterThan(0))
  V = model.variable("V", [n, n], dom.greaterThan(0))
  model.constraint(expr.sub(V.diag(), v), dom.equalsTo(0))
  
  sumA = 0.0
  suma = np.zeros(q.shape)
  for idx in range(m):
    sumA = expr.add(sumA, expr.mul(A[idx], l.index(idx)))
    suma = expr.add(suma, expr.mul(a[idx], l.index(idx)))
  
  model.constraint(
    expr.add(expr.add(expr.sub(-Q, Y), expr.mulElm(np.eye(n), V)), sumA),
    dom.equalsTo(0))
  sum_temp = expr.sub(expr.sub(mu, expr.mul(2, y)), v)
  model.constraint(expr.add(expr.add(sum_temp, suma), -q), dom.greaterThan(0))
  
  # objectives
  obj_expr = expr.add(expr.add(expr.dot(l, b), Z.index(n, n)), expr.sum(mu))
  # obj_expr = expr.add(1, expr.sum(mu))
  model.objective(
    mf.ObjectiveSense.Minimize
    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
  
  r = MSKResult()
  r.xvar = y
  r.yvar = Y
  r.zvar = Z
  r.problem = model
  if not solve:
    return r
  
  model.solve()
  xval = y.level().reshape(xshape)
  r.yval = Y.level().reshape((n, n))
  r.xval = xval
  r.relax_obj = model.primalObjValue()
  r.true_obj = qp_obj_func(Q, q, xval)
  r.solved = True
  r.solve_time = model.getSolverDoubleInfo("optimizerTime")
  
  return r
