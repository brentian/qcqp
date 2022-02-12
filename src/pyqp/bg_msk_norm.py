"""
the norm constrained formulations,
using bilinear term:
- ξ · x = ρ · e
- ρ = (ξ ◦ x)
"""

import numpy as np
import sys
import time

from .bg_msk import MSKResult, dom, expr, mf
from .classes import qp_obj_func, MscBounds, Result, Bounds
from .instances import QP


class MSKNMscResult(MSKResult):
  
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
    self.svar = None
    self.sval = None
    self.rhovar = None
    self.rhoval = None
    self.solved = False
    self.obj_expr = None
    self.qel = None
    self.q = None
    self.unit_time = 0
  
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
      # extract sval and xi
      self.sval = self.svar.level()[0]
      self.rhoval = self.rhovar.level().reshape(self.rhovar.getShape())
      
      # other regular MSC results
      self.xval = self.xvar.level().reshape(self.xvar.getShape()).round(self.PRECISION)
      self.bound = self.relax_obj = self.problem.primalObjValue()
      if qp is not None:
        self.true_obj = qp_obj_func(qp.Q, qp.q, self.xval)
      self.res = np.abs(self.rhoval - self.xval ** 2)
      self.res_norm = self.res.max()
    else:  # infeasible
      self.relax_obj = -1e6
    self.solved = True
    self.unit_time = self.problem.getSolverDoubleInfo("optimizerTime")
    self.solve_time = round(end_time - start_time, 3)


class MSKSocpResult(MSKResult):
  
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
    self.svar = None
    self.sval = None
    self.tvar = None
    self.tval = None
    self.rhovar = None
    self.rhoval = None
    self.solved = False
    self.obj_expr = None
    self.qel = None
    self.q = None
    self.unit_time = 0
  
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
      # extract sval and xi
      self.sval = self.svar.level()[0]
      if self.tvar is not None:
        self.tval = self.tvar.level()[0]
      self.rhoval = self.rhovar.level().reshape(self.rhovar.getShape())
      # other SOCP results
      self.xval = self.xvar.level().reshape(self.xvar.getShape()).round(self.PRECISION)
      self.yval = self.yvar.level().reshape(self.yvar.getShape()).round(self.PRECISION)
      self.bound = self.relax_obj = self.problem.primalObjValue()
      if qp is not None:
        self.true_obj = qp_obj_func(qp.Q, qp.q, self.xval)
      self.res = np.abs(self.rhoval - self.xval ** 2)
      self.res_norm = self.res.max()
    else:  # infeasible
      self.relax_obj = -1e6
    self.solved = True
    self.unit_time = self.problem.getSolverDoubleInfo("optimizerTime")
    self.solve_time = round(end_time - start_time, 3)


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
  The many-small-cone approach
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
  s_ub = np.maximum(bounds.xlb ** 2, bounds.xub ** 2).sum()
  
  # x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  # rho = model.variable("rho", [*xshape], dom.greaterThan(0))
  qcones = model.variable("xr", dom.inRotatedQCone(3, n))
  ones = qcones.slice([0, 0], [1, n])
  rho = qcones.slice([1, 0], [2, n]).reshape(n, 1)
  x = qcones.slice([2, 0], [3, n]).reshape(n, 1)
  model.constraint(ones, dom.equalsTo(0.5))
  model.constraint(x, dom.inRange(bounds.xlb, bounds.xub))
  # norm
  s = model.variable("s", dom.inRange(0, s_ub))
  zcone = model.variable("zc", dom.inPSDCone(2, n))
  y = zcone.slice([0, 0, 0], [n, 1, 1]).reshape([n, 1])
  z = zcone.slice([0, 0, 1], [n, 1, 2]).reshape([n, 1])
  Y = [y]
  Z = [z]
  for idx in range(n):
    model.constraint(zcone.index([idx, 1, 1]), dom.equalsTo(1))
  
  # Q.T x = Z
  model.constraint(expr.sub(expr.mul((qneg + qpos), z), x), dom.equalsTo(0))
  
  # Norm cuts
  model.constraint(expr.sub(expr.sum(y), s), dom.lessThan(0))
  
  #
  model.constraint(
    expr.sub(expr.sum(rho), s), dom.equalsTo(0)
  )
  
  # RLT for ρ = (ξ ◦ x)
  # model.constraint(
  #   expr.sub(rho, expr.mulElm(2 * bounds.xlb, x)),
  #   dom.greaterThan(-bounds.xlb ** 2)
  # )
  # model.constraint(
  #   expr.sub(rho, expr.mulElm(2 * bounds.xub, x)),
  #   dom.greaterThan(-bounds.xub ** 2)
  # )
  model.constraint(
    expr.sub(rho, expr.mulElm(bounds.xub + bounds.xlb, x)),
    dom.lessThan(-bounds.xlb * bounds.xub)
  )
  
  for i in range(m):
    
    quad_expr = expr.dot(a[i], x)
    Ai = qp.A[i]
    if Ai is not None:
      apos, ipos = qp.Apos[i]
      aneg, ineg = qp.Aneg[i]
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
      
      # Norm cuts
      model.constraint(expr.sub(expr.sum(yi), s), dom.lessThan(0))
      
      quad_terms = expr.dot(el, yi)
      
      quad_expr = expr.add(quad_expr, quad_terms)
    
    else:
      Y.append(None)
      Z.append(None)
    
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
  true_obj_expr = expr.add(expr.dot(q, x), expr.dot(qel, y))
  obj_expr = true_obj_expr
  
  # obj_expr = true_obj_expr
  model.objective(
    mf.ObjectiveSense.Minimize
    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr
  )
  
  r = MSKNMscResult()
  r.obj_expr = true_obj_expr
  r.xvar = x
  r.yvar = y
  r.zvar = z
  r.Zvar = Z
  r.Yvar = Y
  r.svar = s
  r.rhovar = rho
  r.qel = qel
  r.q = q
  r.problem = model
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r


def socp(
    qp: QP,
    bounds: MscBounds = None,
    sense="max",
    verbose=True,
    solve=True,
    *args,
    **kwargs
):
  """
  The many-small-cone approach
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
  model = mf.Model('socp-with-norm')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg
  
  qel = qp.Qmul
  # the objective
  z = model.variable("z")
  # [1/2, rho, x] in Q
  qcones = model.variable("xr", dom.inRotatedQCone(3, n))
  ones = qcones.slice([0, 0], [1, n])
  rho = qcones.slice([1, 0], [2, n]).reshape(n, 1)
  x = qcones.slice([2, 0], [3, n]).reshape(n, 1)
  model.constraint(ones, dom.equalsTo(0.5))
  if bounds.xlb is not None:
    model.constraint(x, dom.inRange(bounds.xlb, bounds.xub))
    # s = x^Tx
    s = model.variable("s", dom.lessThan(bounds.sphere ** 2))
  else:
    raise ValueError("did not defined bounds properly")
  # y = x^TRR^Tx
  y = model.variable("y", [m + 1])
  # s = rho^Te
  model.constraint(
    expr.sub(expr.sum(rho), s), dom.equalsTo(0)
  )
  
  # RLT for ρ = (ξ ◦ x)
  model.constraint(
    expr.sub(rho, expr.mulElm(bounds.xub + bounds.xlb, x)),
    dom.lessThan(-bounds.xlb * bounds.xub)
  )
  
  # R.T x = Z
  if Q is not None:
    model.constraint(
      expr.vstack(0.5, y.index(0), expr.flatten(expr.mul(qp.R[-1].T, x))),
      dom.inRotatedQCone()
    )
    model.constraint(
      expr.add([
        expr.mul(qp.l[-1], s),
        expr.mul(-1, z),
        expr.dot(q, x),
        expr.mul(-1, y.index(0))
      ]),
      dom.greaterThan(0)
    )
  else:
    model.constraint(
      expr.add([
        z,
        expr.dot(-q, x)
      ]),
      dom.lessThan(0)
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
  
  # obj_expr = true_obj_expr
  model.objective(
    mf.ObjectiveSense.Minimize
    if sense == 'min' else mf.ObjectiveSense.Maximize, z
  )
  
  r = MSKSocpResult()
  r.obj_expr = z
  r.xvar = x
  r.yvar = y
  r.svar = s
  r.rhovar = rho
  r.qel = qel
  r.q = q
  r.problem = model
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r
