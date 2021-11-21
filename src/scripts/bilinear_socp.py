"""
@author: Chuwen,
@date: 11'11'2021

A test for SOCP relaxation strength.
Consider,
  <x, x> = s
  then easily, we use <x, x> ≤ s
  and we wish <x, x> > s also holds.
  to approach this, let ξ such that max_ξ <x, ξ> = s
  essentially, let <x, ξ> >= s, which is equivalent to,
  
  - <x, x> ≤ s
  - t^2 = s
  - <x, ξ> ≥ s
    => |[2t, x - ξ]| ≤ |x + ξ|
    => t^2 ≤ <x, ξ>
"""
import numpy as np
import sys

from pyqp import bg_grb, bg_msk_norm
from pyqp.classes import *
from pyqp.bg_msk_norm import MSKSocpResult

try:
  import mosek.fusion as mf
  
  expr = mf.Expr
  dom = mf.Domain
  mat = mf.Matrix
except Exception as e:
  import logging
  
  logging.exception(e)


class MSKSocpResultBi(MSKSocpResult):
  def __init__(self):
    super().__init__()
    self.xivar = None
    self.xival = None
  
  def solve(self, verbose=False, qp=None):
    super(MSKSocpResultBi, self).solve(verbose, qp)
    self.xival = self.xivar.level().reshape(self.xivar.getShape()).round(4)


def socp(
    qp: QP,
    bounds: MscBounds = None,
    sense="max",
    verbose=True,
    solve=True,
    *args,
    **kwargs
):
  _unused = kwargs
  Q, q, A, a, b, sign, *_ = qp.unpack()
  if qp.Qpos is None:
    raise ValueError("decompose QP instance first")
  if qp.decom_method == 'eig-type1':
    raise ValueError(f"cannot use {qp.decom_method}")
  m, n, dim = a.shape
  xshape = (n, dim)
  model = mf.Model('socp-with-norm')
  
  model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg
  
  qel = qp.Qmul
  s_ub = np.maximum(bounds.xlb ** 2, bounds.xub ** 2).sum()
  qcones = model.variable("xr", dom.inRotatedQCone(3, n))
  ones = qcones.slice([0, 0], [1, n])
  rho = qcones.slice([1, 0], [2, n]).reshape(n, 1)
  x = qcones.slice([2, 0], [3, n]).reshape(n, 1)
  # auxiliary
  xi = model.variable("xi", [n, 1], dom.inRange(bounds.xlb, bounds.xub))
  model.constraint(ones, dom.equalsTo(0.5))
  model.constraint(x, dom.inRange(bounds.xlb, bounds.xub))
  # norm
  s = model.variable("s", dom.inRange(0, s_ub))
  # sqrt rho
  # t = model.variable("t", [n, 1], dom.inRange(0, np.sqrt(s_ub)))
  # y = x^TRR^Tx, Q = l - RR^T
  y = model.variable("y", [m + 1])
  
  # ρ^Te = s
  model.constraint(
    expr.sub(expr.sum(rho), s), dom.equalsTo(0)
  )
  
  # |x| <= t <= sqrt(s)
  # t <= sqrt(s)
  # model.constraint(
  #   expr.vstack(0.5, s, t),
  #   dom.inRotatedQCone()
  # )
  # |x| <= t
  # model.constraint(
  #   expr.vstack(t, expr.flatten(x)),
  #   dom.inQCone()
  # )
  # # |xi| <= t
  # model.constraint(
  #   expr.vstack(t, expr.flatten(xi)),
  #   dom.inQCone()
  # )
  # |x + xi|
  x_a_xi = expr.add(x, xi)
  # |x - xi|
  x_s_xi = expr.sub(x, xi)
  for i in range(n):
    model.constraint(
      expr.vstack(
        x_a_xi.index([i, 0]),
        x_s_xi.index([i, 0]),
        t.index([i, 0])
      ),
      dom.inQCone()
    )
  
  # R.T x = Z
  obj_expr = 0
  if Q is not None:
    model.constraint(
      expr.vstack(0.5, y.index(0), expr.flatten(expr.mul(qp.R[-1].T, x))),
      dom.inRotatedQCone()
    )
    obj_expr = expr.sub(obj_expr, y.index(0))
    obj_expr = expr.sub(obj_expr, expr.mul(qp.l[-1], s))
  
  # RLT for ρ = (ξ ◦ x)
  model.constraint(
    expr.sub(rho, expr.mulElm(bounds.xub + bounds.xlb, x)),
    dom.lessThan(-bounds.xlb * bounds.xub)
  )
  
  for i in range(m):
    
    quad_expr = expr.dot(a[i], x)
    Ai = qp.A[i]
    if Ai is not None:
      model.constraint(
        expr.vstack(0.5, y.index(i + 1), expr.flatten(expr.mul(qp.R[i].T, x))),
        dom.inRotatedQCone()
      )
      quad_expr = expr.add(quad_expr, y.index(i + 1))
      quad_expr = expr.sub(quad_expr, expr.mul(qp.l[i], s))
    
    if qp.sign is not None:
      # unilateral case
      quad_dom = dom.equalsTo(0) if sign[i] == 0 else (
        dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(b[i])
      )
    else:
      # bilateral case
      # todo, fix this
      # quad_dom = dom.inRange(qp.al[i], qp.au[i])
      quad_dom = dom.lessThan(qp.au[i])
    
    model.constraint(quad_expr, quad_dom)
  
  # objectives
  obj_expr = expr.add(obj_expr, expr.dot(q, x))
  
  # obj_expr = true_obj_expr
  model.objective(
    mf.ObjectiveSense.Minimize
    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr
  )
  
  r = MSKSocpResultBi()
  r.obj_expr = obj_expr
  r.xvar = x
  r.yvar = y
  r.svar = s
  r.tvar = t
  r.rhovar = rho
  r.qel = qel
  r.q = q
  r.xivar = xi
  r.problem = model
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r


if __name__ == '__main__':
  np.random.seed(1)
  m = 1
  n = 10
  qp = QPI.normal(n, m, rho=0.7)
  qp.decompose()
  bd = Bounds(xlb=np.zeros(shape=(n, 1)), xub=np.ones(shape=(n, 1)))
  r_g = bg_grb.qp_gurobi(qp, bd)
  r_n = bg_msk_norm.socp(qp, bd)
  r = socp(qp, bd, solve=True)
