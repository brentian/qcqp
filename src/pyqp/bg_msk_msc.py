import numpy as np
import sys
import time

from .bg_msk import MSKResult, dom, expr, mf
from .classes import qp_obj_func, MscBounds, Result
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
  
  def solve(self, verbose=True, qp=None):
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
      self.xval = self.xvar.level().reshape(self.xvar.getShape()).round(4)
      self.zval = self.zvar.level().reshape(self.zvar.getShape()).round(4)
      self.Zval = np.hstack([xx.level().reshape(self.xvar.getShape()).round(4)
                             if xx is not None else np.zeros(self.xvar.getShape())
                             for xx in self.Zvar])
      if self.yvar is not None:
        self.yval = self.yvar.level().reshape(self.yvar.getShape()).round(4)
      if self.Yvar is not None:
        self.Yval = np.hstack([xx.level().reshape(self.xvar.getShape()).round(4)
                               if xx is not None else np.zeros(self.xvar.getShape())
                               for xx in self.Yvar])
      
      if self.Dvar is not None:
        self.Dval = np.hstack([xx.level().reshape((2, 1)).round(4)
                               if xx is not None else np.zeros((2, 1))
                               for xx in self.Dvar])
      self.relax_obj = self.problem.primalObjValue()
      if qp is not None:
        self.true_obj = qp_obj_func(qp.Q, qp.q, self.xval)
    else:  # infeasible
      self.relax_obj = -1e6
    self.solved = True
    self.solve_time = round(end_time - start_time, 3)


def msc(
    qp: QP, bounds: MscBounds = None,
    sense="max", verbose=True, solve=True,
    with_shor: Result = None,  # if not None then use Shor relaxation as upper bound
    rlt=False,  # True add all rlt/secant cut: yi - (li + ui) zi + li * ui <= 0
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
  m, n, dim = a.shape
  xshape = (n, dim)
  model = mf.Model('many_small_cone_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  zlb = bounds.zlb
  zub = bounds.zub
  
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
  model.constraint(
    expr.sub(
      expr.mul((qneg + qpos).T, x),
      z), dom.equalsTo(0))
  
  # RLT cuts
  if rlt:
    if qp.decom_method == 'eig-type1':
      # rlt_expr = expr.sub(y, expr.mulElm(zlb[0] + zub[0], z))
      # model.constraint(rlt_expr, dom.lessThan(- zlb[0] * zub[0]))
      rlt_expr_ex = expr.sub(expr.sum(expr.mulElm(y, np.abs(1 / qp.Qeig))),
                             expr.dot(bounds.xlb + bounds.xub, x))
      model.constraint(rlt_expr_ex, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
    elif qp.decom_method == 'eig-type2':
      # this means you can place on x directly.
      rlt_expr = expr.sub(expr.sum(y), expr.dot(bounds.xlb + bounds.xub, x))
      model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
    else:
      raise ValueError("Decompose first!")
  
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
      
      # bounds
      model.constraint(zi, dom.inRange(zlb[i + 1], zub[i + 1]))
      if bounds.yub is not None:
        model.constraint(yi, dom.lessThan(bounds.yub[i + 1]))
      
      # Z[-1, -1] == 1
      for idx in range(n):
        model.constraint(zconei.index([idx, 1, 1]), dom.equalsTo(1))
      
      # A.T @ x == z
      model.constraint(
        expr.sub(
          expr.mul((apos + aneg).T, x),
          zi), dom.equalsTo(0))
      
      if rlt:
        if qp.decom_method == 'eig-type1':
          rlt_expr = expr.sub(yi, expr.mulElm(zlb[i + 1] + zub[i + 1], zi))
          model.constraint(rlt_expr, dom.lessThan(- zlb[i + 1] * zub[i + 1]))
        elif qp.decom_method == 'eig-type2':
          # this means you can place on x directly.
          rlt_expr = expr.sub(expr.sum(yi), expr.dot(bounds.xlb + bounds.xub, x))
          model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
        else:
          raise ValueError("Decompose first!")
      
      quad_terms = expr.dot(el, yi)
      
      quad_expr = expr.add(quad_expr, quad_terms)
    
    else:
      Y.append(None)
      Z.append(None)
    
    quad_dom = dom.equalsTo(0) if sign[i] == 0 else (dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0))
    
    model.constraint(
      quad_expr, quad_dom)
  
  # objectives
  true_obj_expr = expr.add(expr.dot(q, x), expr.dot(qel, y))
  obj_expr = true_obj_expr
  
  # with shor results
  if with_shor is not None:
    # use shor as ub
    shor_ub = with_shor.relax_obj.round(4)
    model.constraint(
      true_obj_expr, dom.lessThan(shor_ub)
    )
  
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
  
  r.solve(verbose=verbose, qp=qp)
  
  return r


def msc_diag(
    qp: QP, bounds: MscBounds = None,
    sense="max", verbose=True, solve=True,
    with_shor: Result = None,  # if not None then use Shor relaxation as upper bound
    rlt=True,  # True add all rlt/secant cut: yi - (li + ui) zi + li * ui <= 0
    lk=False,  # True then add lk constraint
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
  model = mf.Model('many_small_cone_msk')
  
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
  model.constraint(
    expr.sub(
      expr.mul((qneg + qpos), z),
      x), dom.equalsTo(0))
  
  # RLT cuts
  if rlt:
    # this means you can place on x directly.
    rlt_expr = expr.sub(expr.sum(y), expr.dot(bounds.xlb + bounds.xub, x))
    model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
  
  if lk:
    lk_expr = expr.sub(expr.sum(y), expr.sum(x))
    model.constraint(lk_expr, dom.lessThan(0))
  
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
        expr.sub(
          expr.mul((apos + aneg), zi),
          x), dom.equalsTo(0))
      
      if rlt:
        # this means you can place on x directly.
        rlt_expr = expr.sub(expr.sum(yi), expr.dot(bounds.xlb + bounds.xub, x))
        model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
      
      if lk:
        lk_expr = expr.sub(expr.sum(yi), expr.sum(y))
        model.constraint(lk_expr, dom.equalsTo(0))
      
      quad_terms = expr.dot(el, yi)
      
      quad_expr = expr.add(quad_expr, quad_terms)
    
    else:
      Y.append(None)
      Z.append(None)
    
    quad_dom = dom.equalsTo(0) if sign[i] == 0 else (dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0))
    
    model.constraint(
      quad_expr, quad_dom)
  
  # objectives
  true_obj_expr = expr.add(expr.dot(q, x), expr.dot(qel, y))
  obj_expr = true_obj_expr
  
  # with shor results
  if with_shor is not None:
    # use shor as ub
    shor_ub = with_shor.relax_obj.round(4)
    model.constraint(
      true_obj_expr, dom.lessThan(shor_ub)
    )
  
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
  
  r.solve(verbose=verbose, qp=qp)
  
  return r

##################################
# USING SOCPs, should be improved
##################################


def msc_socp_relaxation(
    qp: QP, bounds: MscBounds = None,
    sense="max", verbose=True, solve=True,
    with_shor: Result = None,  # if not None then use Shor relaxation as upper bound
    rlt=False,  # True add all rlt/secant cut: yi - (li + ui) zi + li * ui <= 0
    *args,
    **kwargs
):
  _unused = kwargs
  Q, q, A, a, b, sign, *_ = qp.unpack()
  if qp.Qpos is None:
    qp.decompose()
  m, n, dim = a.shape
  xshape = (n, dim)
  model = mf.Model('msc_socp_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  zlb = bounds.zlb
  zub = bounds.zub
  ylb = bounds.ylb
  yub = bounds.yub
  
  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg
  
  # build a vector of signs
  # qel = np.ones([*xshape])
  # qel[qineg] = -1
  qel = qp.Qmul
  
  x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  y = model.variable("y", [*xshape], dom.inRange(ylb[0], yub[0]))
  z = model.variable("z", [*xshape], dom.inRange(zlb[0], zub[0]))
  
  Z = [z]
  Y = [y]
  
  #
  model.constraint(
    expr.sub(
      expr.mul((qneg + qpos).T, x),
      z), dom.equalsTo(0))
  
  # 2nd order cone
  for j in range(n):
    model.constraint(
      expr.vstack(0.5, y.index([j, 0]), z.index([j, 0])),
      dom.inRotatedQCone()
    )
  
  if rlt:
    rlt_expr = expr.sub(y, expr.mulElm(zlb[0] + zub[0], z))
    model.constraint(rlt_expr, dom.lessThan(- zlb[0] * zub[0]))
  
  for i in range(m):
    apos, ipos = qp.Apos[i]
    aneg, ineg = qp.Aneg[i]
    quad_expr = expr.sub(expr.dot(a[i], x), b[i])
    
    if ipos.shape[0] + ineg.shape[0] > 0:
      
      # if it is indeed quadratic
      zi = model.variable(f"z_{i}", [*xshape], dom.inRange(zlb[i + 1], zub[i + 1]))
      yi = model.variable(f"y_{i}", [*xshape], dom.inRange(ylb[i + 1], yub[i + 1]))
      
      Z.append(zi)
      Y.append(yi)
      
      # build a vector of signs
      # el = np.ones([n, 1])
      # el[ineg] = -1
      el = qp.Amul[i]
      
      # A.T @ x == z
      model.constraint(
        expr.sub(
          expr.mul((apos + aneg).T, x),
          zi), dom.equalsTo(0))
      
      for j in range(n):
        model.constraint(
          expr.vstack(0.5, yi.index([j, 0]), zi.index([j, 0])),
          dom.inRotatedQCone()
        )
      
      if rlt:
        rlt_expr = expr.sub(yi, expr.mulElm(zlb[i + 1] + zub[i + 1], zi))
        model.constraint(rlt_expr, dom.lessThan(- zlb[i + 1] * zub[i + 1]))
      
      # for dp, dn
      # dp = y^Te
      quad_terms = expr.dot(el, yi)
      quad_expr = expr.add(quad_expr, quad_terms)
    
    else:
      Z.append(None)
      Y.append(None)
    
    quad_dom = dom.equalsTo(0) if sign[i] == 0 else (dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0))
    
    model.constraint(
      quad_expr, quad_dom)
  
  # objectives
  true_obj_expr = expr.add(expr.dot(q, x), expr.dot(qel, y))
  obj_expr = true_obj_expr
  
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
  r.Dvar = None
  r.qel = qel
  r.q = q
  r.problem = model
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r


def socp_relaxation(
    qp: QP, bounds: MscBounds = None,
    sense="max", verbose=True, solve=True,
    rlt=False,  # True add all rlt/secant cut: yi - (li + ui) zi + li * ui <= 0
    *args,
    **kwargs
):
  _unused = kwargs
  Q, q, A, a, b, sign, *_ = qp.unpack()
  if qp.Qpos is None:
    qp.decompose()
  m, n, dim = a.shape
  xshape = (n, dim)
  model = mf.Model('socp_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  zlb = bounds.zlb
  zub = bounds.zub
  
  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg
  
  # build a vector of signs
  # qel = np.ones([*xshape])
  # qel[qineg] = -1
  qel = qp.Qmul
  
  x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  z = model.variable("z", [*xshape], dom.inRange(zlb[0], zub[0]))
  d = model.variable("d", 2, dom.inRange(bounds.dlb[0], bounds.dub[0]))
  
  Z = [z]
  D = [d]
  
  #
  model.constraint(
    expr.sub(
      expr.mul((qneg + qpos).T, x),
      z), dom.equalsTo(0))
  
  zp = z.pick([[j, 0] for j in qipos])
  zn = z.pick([[j, 0] for j in qineg])
  # 2nd order cone
  model.constraint(
    expr.vstack(0.5, d.index(0), zp),
    dom.inRotatedQCone()
  )
  model.constraint(
    expr.vstack(0.5, d.index(1), z.pick([[j, 0] for j in qineg])),
    dom.inRotatedQCone()
  )
  
  qqpos = qpos @ qpos.T
  qqneg = qneg @ qneg.T
  
  if rlt:
    lusum = zlb[0] + zub[0]
    lumul = zlb[0] * zub[0]
    rlt_exprp = expr.sub(d.index(0), expr.dot(lusum[qipos].flatten(), zp))
    model.constraint(rlt_exprp, dom.lessThan(- lumul[qipos].sum()))
    rlt_exprn = expr.sub(d.index(1), expr.dot(lusum[qineg].flatten(), zn))
    model.constraint(rlt_exprn, dom.lessThan(- lumul[qineg].sum()))
  
  for i in range(m):
    apos, ipos = qp.Apos[i]
    aneg, ineg = qp.Aneg[i]
    quad_expr = expr.sub(expr.dot(a[i], x), b[i])
    
    if ipos.shape[0] + ineg.shape[0] > 0:
      
      # if it is indeed quadratic
      zi = model.variable(f"z_{i}", [*xshape], dom.inRange(zlb[i + 1], zub[i + 1]))
      di = model.variable(f"d_{i}", 2, dom.inRange(bounds.dlb[i + 1], bounds.dub[i + 1]))
      Z.append(zi)
      D.append(di)
      
      # build a vector of signs
      # el = np.ones([n, 1])
      # el[ineg] = -1
      el = qp.Amul[i]
      
      # A.T @ x == z
      model.constraint(
        expr.sub(
          expr.mul((apos + aneg).T, x),
          zi), dom.equalsTo(0))
      
      lusum = zlb[i + 1] + zub[i + 1]
      lumul = zlb[i + 1] * zub[i + 1]
      
      if ipos.shape[0] > 0:
        zpi = zi.pick([[j, 0] for j in ipos])
        # 2nd order cone
        model.constraint(
          expr.vstack(0.5, di.index(0), zpi),
          dom.inRotatedQCone()
        )
        if rlt:
          rlt_exprp = expr.sub(di.index(0), expr.dot(lusum[ipos].flatten(), zpi))
          model.constraint(rlt_exprp, dom.lessThan(- lumul[ipos].sum()))
      
      if ineg.shape[0] > 0:
        zni = zi.pick([[j, 0] for j in ineg])
        model.constraint(
          expr.vstack(0.5, di.index(1), zni),
          dom.inRotatedQCone()
        )
        if rlt:
          rlt_exprn = expr.sub(di.index(1), expr.dot(lusum[ineg].flatten(), zni))
          model.constraint(rlt_exprn, dom.lessThan(- lumul[ineg].sum()))
      
      # for dp, dn
      # dp = y^Te
      quad_terms = expr.sub(di.index(0), di.index(1))
      quad_expr = expr.add(quad_expr, quad_terms)
    
    else:
      Z.append(None)
      D.append(None)
    
    quad_dom = dom.equalsTo(0) if sign[i] == 0 else (dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0))
    
    model.constraint(
      quad_expr, quad_dom)
  
  # objectives
  true_obj_expr = expr.add(expr.dot(q, x), expr.sub(d.index(0), d.index(1)))
  obj_expr = true_obj_expr
  
  # obj_expr = true_obj_expr
  model.objective(mf.ObjectiveSense.Minimize
                  if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
  
  r = MSKMscResult()
  r.obj_expr = true_obj_expr
  r.xvar = x
  r.zvar = z
  r.Zvar = Z
  r.Yvar = None
  r.Dvar = D
  r.qel = qel
  r.q = q
  r.problem = model
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r