"""
The script for SOCP based relaxation

- use n-dimensional SOC, SOCr instead of SDP
- construct ellipsoid covering the box.

"""

import numpy as np
import sys
import time

from .bg_msk import MSKResult, dom, expr, mf
from .classes import qp_obj_func, MscBounds, Result, Bounds
from .instances import QP


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
      if self.rhovar is not None:
        self.rhoval = self.rhovar.level().reshape(self.rhovar.getShape())
      # other SOCP results
      self.xval = self.xvar.level().reshape(self.xvar.getShape()).round(4)
      self.yval = self.yvar.level().reshape(self.yvar.getShape()).round(4)
      self.bound = self.relax_obj = self.problem.primalObjValue()
      if qp is not None:
        self.true_obj = qp_obj_func(qp.Q, qp.q, self.xval)
      # self.res = np.abs(self.rhoval - self.xval ** 2)
      # self.res_norm = self.res.max()
    else:  # infeasible
      self.relax_obj = -1e6
    self.solved = True
    self.solve_time = round(end_time - start_time, 3)


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
  Returns
  -------
  """
  _unused = kwargs
  Q, q, A, a, b, sign, *_ = qp.unpack()
  if qp.Qpos is None:
    raise ValueError("decompose QP instance first")
  if qp.decom_method == 'eig-type2':
    raise ValueError(f"cannot use {qp.decom_method}")
  m, n, dim = a.shape
  xshape = (n, dim)
  model = mf.Model('socp-with-norm')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  Rn = qp.Qneg[0]
  Rp = qp.Qpos[0]
  # Qp = Rp @ Rp.T
  # Qp_min = Qp.min()
  # kappa = (- Qp_min + 1e-2) * (Qp_min < 0)
  # Rp[:, idxp] = np.sqrt(kappa)
  # Rn[:, idxn] = np.sqrt(kappa)
  Qp = Rp @ Rp.T
  Qn = Rn @ Rn.T
  
  assert np.abs(Qp - Qn - Q).min() < 1e-3
  
  l, u = bounds.xlb.flatten(), bounds.xub.flatten()
  rho = model.variable('rho')
  z = model.variable('z')
  x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  
  if Q is not None:
    ############################
    cone_0 = model.variable("cone_0", n + 2, dom.inRotatedQCone())
    p = cone_0.index(1)
    y = cone_0.slice(2, n + 2)
    model.constraint(cone_0.index(0), dom.equalsTo(0.5))
    # objective
    model.constraint(
      expr.sub(
        y,
        expr.mul(Rn.T, x)
      ),
      dom.equalsTo(0)
    )
    model.constraint(
      expr.add(
        [p, expr.dot(- q, x), z, expr.mul(-1, rho)]
      ),
      dom.lessThan(0)
    )
    ############################
    # convexification
    # 1. use |x|^2 to convexify
    # 2. use Q- to conv Q+
    ############################
    # use Q+
    # create the xLx - xL(l+u) + lLu <= 0
    # 1.
    cone_box = model.variable("cone_box1", n + 2, dom.inRotatedQCone())
    s = cone_box.index(1)
    v = cone_box.slice(2, n + 2)
    lmax = np.linalg.eigvalsh(Qp).max() + 1e-2
    Rpp = np.linalg.cholesky(lmax * np.eye(Qp.shape[0]) - Qp)
    model.constraint(
      expr.add(
        [s, expr.dot(- lmax * (l + u), x), rho]
      ),
      dom.lessThan(- lmax * l.T @ u)
    )
    model.constraint(cone_box.index(0), dom.equalsTo(0.5))
    model.constraint(
      expr.sub(
        v,
        expr.mul(Rpp.T, x)
      ),
      dom.equalsTo(0)
    )
    # 2.
    # cone_box = model.variable("cone_box2", n + 2, dom.inRotatedQCone())
    # s = cone_box.index(1)
    # v = cone_box.slice(2, n + 2)
    # Vn = np.linalg.inv(Rn)
    # lmax = np.linalg.eigvalsh(Vn @ Qp @ Vn.T).max() + 1e-2
    # Rpp = np.linalg.cholesky(lmax * Qn - Qp)
    # model.constraint(
    #   expr.add(
    #     [s, expr.dot(- lmax * q, x),
    #      expr.mul(lmax, z), expr.mul(-lmax + 1, rho)]
    #   ),
    #   dom.lessThan(0)
    # )
    # model.constraint(cone_box.index(0), dom.equalsTo(0.5))
    # model.constraint(
    #   expr.sub(
    #     v,
    #     expr.mul(Rpp.T, x)
    #   ),
    #   dom.equalsTo(0)
    # )
    
    
    ###########################
    # generalized RLT
    # model.constraint(
    #   expr.add(
    #     [rho, expr.dot(- Qn @ (u + l), x)]
    #   ),
    #   dom.lessThan(- l.T @ Qn @ u)
    # )
  
  for i in range(m):
    # todo, not finished
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
  obj_expr = z
  # obj_expr = true_obj_expr
  model.objective(
    mf.ObjectiveSense.Minimize
    if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr
  )
  
  r = MSKSocpResult()
  r.obj_expr = obj_expr
  r.xvar = x
  r.yvar = y
  r.svar = s
  r.problem = model
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r
