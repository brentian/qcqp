"""
Using second-order cones, small or large
"""

import numpy as np
import sys
import time

from .bg_msk import MSKResult, dom, expr, mf
from .classes import qp_obj_func, MscBounds, Result, Bounds
from .instances import QP


class MSKMscResult(MSKResult):
  
  def __init__(self):
    super().__init__()
    self.zvar = None
    self.zval = None
    self.Zvar = None
    self.Yvar = None
    self.Yval = None
    self.Zval = None
    self.Dvar = None
    self.Dval = None
    self.solved = False
    self.obj_expr = None
    self.qel = None
    self.q = None
  
  def solve(self, verbose=False, qp=None):
    start_time = time.time()
    if verbose:
      self.problem.setLogHandler(sys.stdout)
    try:
      self.problem.solve()
      status = self.problem.getProblemStatus()
    except Exception as e:
      status = 'failed'
    end_time = time.time()
    if status == mf.ProblemStatus.PrimalAndDualFeasible:
      self.xval = self.xvar.level().reshape(self.xvar.getShape()).round(self.PRECISION)
      self.zval = self.zvar.level().reshape(self.zvar.getShape()).round(self.PRECISION)
      self.Zval = np.hstack(
        [
          xx.level().reshape(self.xvar.getShape()).round(self.PRECISION)
          if xx is not None else np.zeros(self.xvar.getShape())
          for xx in self.Zvar
        ]
      )
      if self.yvar is not None:
        self.yval = self.yvar.level().reshape(self.yvar.getShape()).round(self.PRECISION)
      if self.Yvar is not None:
        self.Yval = np.hstack(
          [
            xx.level().reshape(self.xvar.getShape()).round(self.PRECISION)
            if xx is not None else np.zeros(self.xvar.getShape())
            for xx in self.Yvar
          ]
        )
      
      if self.Dvar is not None:
        self.Dval = np.hstack(
          [
            xx.level().reshape((2, 1)).round(self.PRECISION) if xx is not None else np.zeros(
              (2, 1)
            ) for xx in self.Dvar
          ]
        )
      self.bound = self.relax_obj = self.problem.primalObjValue()
      if qp is not None:
        self.true_obj = qp_obj_func(qp.Q, qp.q, self.xval)
    else:  # infeasible
      self.bound = self.relax_obj = -1e6
    self.solved = True
    self.unit_time = self.problem.getSolverDoubleInfo("optimizerTime")
    self.solve_time = round(end_time - start_time, 3)
    ############################
    # extras
    ############################
    yc = self.Yval
    zc = self.Zval
    resc = self.resc = np.abs(yc - zc ** 2)
    self.resc_feas = resc.max()
    self.resc_feasC = resc[:, 1:].max() if resc.shape[1] > 1 else 0
    


def msc_diag(
    qp: QP,
    bounds: MscBounds = None,
    sense="max",
    verbose=True,
    solve=True,
    *args,
    **kwargs
):
  """
  The many-small-cone approach (with sdp)
  Returns
  -------
  """
  _unused = kwargs
  Q, q, A, a, b, sign, *_ = qp.unpack()
  if qp.Qpos is None:
    raise ValueError("decompose QP instance first")
  if qp.decom_method == 'eig-type1':
    raise ValueError(f"cannot use {qp.decom_method}")
  m, n, dim = a.shape
  xshape = (n, dim)
  model = mf.Model('msc_diagonal_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg
  
  qel = qp.Qmul
  
  x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  zcone = model.variable("zc", dom.inPSDCone(2, n))
  y = zcone.slice([0, 0, 0], [n, 1, 1]).reshape([n, 1])
  z = zcone.slice([0, 0, 1], [n, 1, 2]).reshape([n, 1])
  Y = [y]
  Z = [z]
  for idx in range(n):
    model.constraint(zcone.index([idx, 1, 1]), dom.equalsTo(1))
  
  # Q.T x = Z
  model.constraint(expr.sub(expr.mul((qneg + qpos), z), x), dom.equalsTo(0))
  
  # RLT cuts
  # this means you can place on x directly.
  rlt_expr = expr.sub(expr.sum(y), expr.dot(bounds.xlb + bounds.xub, x))
  model.constraint(rlt_expr, dom.lessThan(-(bounds.xlb * bounds.xub).sum()))
  # else:
  model.constraint(expr.sum(y), dom.lessThan(bounds.sphere**2))
  
  for i in range(m):
    apos, ipos = qp.Apos[i]
    aneg, ineg = qp.Aneg[i]
    quad_expr = expr.sub(expr.dot(a[i], x), b[i])
    
    if ipos.shape[0] + ineg.shape[0] > 0:
      
      # if it is indeed quadratic
      zconei = model.variable(f"zci@{i}", dom.inPSDCone(2, n))
      yi = zconei.slice([0, 0, 0], [n, 1, 1]).reshape([n, 1])
      zi = zconei.slice([0, 0, 1], [n, 1, 2]).reshape([n, 1])
      Y.append(yi)
      Z.append(zi)
      
      el = qp.Amul[i]
      
      # Z[-1, -1] == 1
      for idx in range(n):
        model.constraint(zconei.index([idx, 1, 1]), dom.equalsTo(1))
      
      # A.T @ x == z
      model.constraint(
        expr.sub(expr.mul((apos + aneg), zi), x), dom.equalsTo(0)
      )
      
      if norm <= 0:
        # this means you can place on x directly.
        rlt_expr = expr.sub(expr.sum(yi), expr.dot(bounds.xlb + bounds.xub, x))
        model.constraint(
          rlt_expr, dom.lessThan(-(bounds.xlb * bounds.xub).sum())
        )
      else:
        model.constraint(expr.sum(yi), dom.lessThan(norm))
      
      quad_terms = expr.dot(el, yi)
      
      quad_expr = expr.add(quad_expr, quad_terms)
    
    else:
      Y.append(None)
      Z.append(None)
    
    quad_dom = dom.equalsTo(0) if sign[i] == 0 else (
      dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0)
    )
    
    model.constraint(quad_expr, quad_dom)
  
  # objectives
  true_obj_expr = expr.add(expr.dot(q, x), expr.dot(qel, y))
  obj_expr = true_obj_expr
  
  # obj_expr = true_obj_expr
  model.objective(
    mf.ObjectiveSense.Minimize
    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr
  )
  
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
  
  r.solve(verbose=verbose, qp=qp)
  
  return r