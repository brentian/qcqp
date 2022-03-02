"""
Using second-order cones, small or large
"""

import numpy as np
import sys
import time

from .bg_msk import MSKResult, dom, expr, mf
from .bg_msk_msc import msc_diag, MSKMscResult
from .classes import qp_obj_func, MscBounds, Result, Bounds
from .instances import QP


class MSKMscTRSResult(MSKMscResult):
  
  def __init__(self):
    super().__init__()
    self.zvar = None
    self.zval = None
    self.Zvar = None
    self.Yvar = None
    self.Yval = 0
    self.Zval = 0
    self.Dvar = None
    self.Dval = None
    self.solved = False
    self.obj_expr = None
    self.qel = None
    self.q = None
  
  def solve(self, verbose=False, qp: QP = None, rk: MSKMscResult = None):
    start_time = time.time()
    if verbose:
      self.problem.setLogHandler(sys.stdout)
    try:
      self.problem.solve()
      status = self.problem.getProblemStatus()
    except Exception as e:
      raise ValueError(f"failed with status: {status}")
    end_time = time.time()
    if status == mf.ProblemStatus.PrimalAndDualFeasible:
      self.vval = self.xvar.level().reshape(self.xvar.getShape())  # .round(self.PRECISION)
      self.zval = self.zvar.level().reshape(self.zvar.getShape())  # .round(self.PRECISION)
      if self.yvar is not None:
        self.yval = self.yvar.level().reshape(self.yvar.getShape())  # .round(self.PRECISION)
      self.bound = self.relax_obj = self.problem.primalObjValue()
      ############################
      # add the trial step
      ############################
      self.xval = self.vval + rk.xval
      self.dval = self.zval - rk.zval
      self.rval = self.yval - rk.yval
      ############################
      # extras
      ############################
      yc = self.yval
      zc = self.zval
      resc = self.resc = np.abs(yc - zc ** 2)
      self.resc_feas = resc.max()
      if qp.m > 0:
        self.resc_feasC = qp.check(self.xval).max()
      else:
        self.resc_feasC = 0
      if self.resc_feasC <= 5e-5:
        self.true_obj = qp_obj_func(qp.Q, qp.q, self.xval)
      else:
        self.true_obj = -1e6
    else:  # infeasible
      self.bound = self.relax_obj = -1e6
      self.resc_feas = 0
      self.resc_feasC = 0
    self.solved = True
    self.unit_time = self.problem.getSolverDoubleInfo("optimizerTime")
    self.solve_time = round(end_time - start_time, 3)


def trs_msc(
    qp: QP,
    bounds: MscBounds = None,
    rk: MSKMscResult = None,
    g=None,  #
    d=None,  #
    sense="max",
    verbose=True,
    solve=True,
    *args,
    **kwargs
):
  """
  The interior point trust region for
    penalized many-small-cone
  Returns
  -------
  """
  _unused_args = args
  _unused_kwargs = kwargs
  Q, q, A, a, b, sign, *_ = qp.unpack()
  
  m, n, dim = a.shape
  xshape = (n, dim)
  model = mf.Model('msc_diagonal_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if rk is None:
    rk = msc_diag(qp, bounds, solve=True)
  
  # current point
  xk, yk, zk = rk.xval, rk.yval, rk.zval
  
  # at most n 3-d rotated cones for
  # (1, y, z = v'x) ∈ Q
  qcones = model.variable("xr", dom.inRotatedQCone(3, n))
  ones = qcones.slice([0, 0], [1, n])
  # y_+ = y_k + dy,
  # z_+ = z_k + dz,
  # x_+ = x_k + dx,
  yp = qcones.slice([1, 0], [2, n]).reshape(n, 1)
  zp = qcones.slice([2, 0], [3, n]).reshape(n, 1)
  model.constraint(ones, dom.equalsTo(0.5))
  
  # compute the step
  dz = expr.sub(zp, zk)
  dy = expr.sub(yp, yk)
  dx = model.variable("x", [*xshape], dom.inRange(bounds.xlb - xk, bounds.xub - xk))

  # second-order cones
  s = model.variable('sqr', [m + 1])
  
  # V.T x = Z
  model.constraint(expr.sub(expr.mul(qp.V, dz), dx), dom.equalsTo(0))
  
  for i in range(m):
    # for each i in m,
    #   construct the trial step constraint
    quad_expr = expr.dot(a[i] + 2 * qp.At[i + 1] @ xk, dx)
    if not qp.bool_zero_mat[i + 1]:
      si = s.index(i + 1)
      model.constraint(
        expr.vstack(0.5, si, expr.flatten(expr.mul(qp.R[i + 1].T, dx))),
        dom.inRotatedQCone()
      )
      quad_expr = expr.add(quad_expr, si)
      quad_expr = expr.sub(quad_expr, expr.dot(qp.l[i + 1] * np.ones((n, 1)), dy))
    
    if qp.sign is not None:
      # unilateral case
      quad_dom = dom.equalsTo(0) if sign[i] == 0 else dom.lessThan(0)
    
    else:
      # bilateral case
      # todo, fix this
      _l, _u = qp.al[i], qp.au[i]
      if _u < 1e6:
        if _l > -1e6:
          # bilateral
          quad_dom = dom.inRange(qp.al[i], qp.au[i])
        else:
          # LHS is inf
          quad_dom = dom.lessThan(qp.au[i])
    
    model.constraint(quad_expr, quad_dom)
  
  # objective is the penalty term
  y_sqr = model.variable('yp')
  # todo, let g = I, D = diag(zval)
  # then  σ/min(zval) D − σΓ ⪰ 0
  # objectives
  yshape = dy.getShape()
  sigma = qp.l[0] * 10
  a = qp.gamma[0] * (qp.gamma[0] > 0)
  G = np.eye(n) + np.diag(a.flatten())
  Gd = np.diag(G).reshape(yshape)
  D = np.eye(n)
  Dd = np.diag(D).reshape(yshape)
  Rd = np.sqrt(D)
  mu = np.sqrt((Gd / Dd).max() + 10)
  # -C(Γ)
  Cg = (rk.resc.T @ Gd).sum() * sigma
  
  pstack = expr.vstack(
    [y_sqr,
     expr.flatten(expr.mul(mu * Rd.T, dy)),
     expr.flatten(expr.mul(mu * D - G, dz))]
  )
  penalty = model.constraint(
    expr.vstack(
      0.5,
      pstack
    ),
    dom.inRotatedQCone()
  )
  
  obj_expr = expr.add(
    [
      expr.dot(q, dx),
      expr.dot(qp.gamma[0].reshape(yshape) - sigma * Gd, dy),
      expr.mul(- sigma, y_sqr),
      expr.dot(dz, 2 * sigma * G @ zk),
      # also add -C(Γ)
    ]
  )
  
  # obj_expr = true_obj_expr
  model.objective(
    mf.ObjectiveSense.Maximize, expr.sub(obj_expr, Cg)
  )
  
  r = MSKMscTRSResult()
  r.obj_expr = obj_expr
  r.xvar = dx
  r.yvar = yp
  r.zvar = zp
  r.qel = qp.gamma[0]
  r.q = q
  r.problem = model
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp, rk=rk)
  
  return r
