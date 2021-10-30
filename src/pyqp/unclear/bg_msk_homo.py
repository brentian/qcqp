import numpy as np
import sys
import time

from .bg_msk import MSKResult
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


class MSKHomoResult(MSKResult):
  def __init__(
      self,
      qp: QP = None,
      problem: mf.Model = None,
      xvar: mf.Variable = None,
      yvar: mf.Variable = None,
      zvar: mf.Variable = None
  ):
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
    self.relax_obj = self.problem.primalObjValue()
    self.solved = True
    self.solve_time = self.problem.getSolverDoubleInfo("optimizerTime")
    self.total_time = time.time() - self.start_time
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


def shor(
    qp: QP,
    bounds: Bounds,
    sense="max",
    verbose=True,
    solve=True,
    r_parent: MSKResult = None,
    **kwargs
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
  st_time = time.time()
  Q, q, A, a, b, sign = qp.unpack()
  Qh = qp.Qh
  Ah = qp.Ah
  lb, ub, ylb, yub = bounds.unpack()
  m, n, d = a.shape
  xshape = (n, d)
  model = mf.Model('shor_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  
  # bounds
  # model.constraint(expr.sub(x, ub), dom.lessThan(0))
  # model.constraint(expr.sub(x, lb), dom.greaterThan(0))
  model.constraint(expr.sub(Z.diag(), 1), dom.lessThan(0))
  # model.constraint(Z.index(n, n), dom.equalsTo(1.))
  
  if r_parent is not None:
    # Y.setLevel(r_parent.yval.flatten())
    # x.setLevel(np.zeros(n).tolist())
    # x.index(0, 0).setLevel([r_parent.xval[0,0]])
    # need control for the HSD model
    pass
  
  for i in range(m):
    if sign[i] == 0:
      model.constraint(
        expr.sum(expr.mulElm(Z, Ah[i])),
        dom.equalsTo(0))
    elif sign[i] == -1:
      model.constraint(
        expr.sum(expr.mulElm(Z, Ah[i])),
        dom.greaterThan(0))
    else:
      model.constraint(
        expr.sum(expr.mulElm(Z, Ah[i])),
        dom.lessThan(0))
  #
  # objectives
  obj_expr = expr.sum(expr.mulElm(Qh, Z))
  model.objective(mf.ObjectiveSense.Minimize
                  if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
  
  r = MSKResult(qp, model, zvar=Z)
  r.start_time = st_time
  r.build_time = time.time() - st_time
  # if not solve:
  #   return r
  # r.solve()
  # if verbose:
  #   print(r.build_time, r.total_time, r.solve_time)
  model.solve()
  zval = Z.level().reshape(Z.getShape())
  r.zval = zval
  t = zval[-1, -1]
  r.xval = zval[:-1, n:n + 1] / t
  r.yval = zval[:-1, :-1] / t
  
  return r
