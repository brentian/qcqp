import numpy as np

from .bg_msk import *


def msc_diag_sdp_relaxation(
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
  model = mf.Model('msc_sdp_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg
  
  qel = qp.Qmul
  
  x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  y = model.variable("y", [*xshape], dom.greaterThan(0))
  z = model.variable("z", [*xshape])
  Y = [y]
  Z = [z]
  
  # Q.T x = Z
  model.constraint(
    expr.sub(
      expr.mul((qneg + qpos), z),
      x), dom.equalsTo(0))
  
  # for maximization problem
  for idx in qineg:
    model.constraint(expr.vstack(0.5, y.index([idx, 0]), z.index([idx, 0])),
                     dom.inRotatedQCone())
  
  # for maximization problem
  #   use a SD matrix for positive part of
  #   [eigenvalues and eigenvectors]
  n_pos_eig = qipos.shape[0]
  if n_pos_eig > 0:
    # else the problem is convex
    arr_pos_dim = np.zeros(n_pos_eig).astype(int).tolist()
    zpos = z.pick(qipos.tolist(), arr_pos_dim)
    ypos = y.pick(qipos.tolist(), arr_pos_dim)
    ymat = model.variable("yy", dom.inPSDCone(n_pos_eig + 1))
    yy = ymat.slice([0, 0], [n_pos_eig, n_pos_eig])
    zz = ymat.slice([0, n_pos_eig], [n_pos_eig, n_pos_eig + 1])
    model.constraint(ymat.index(n_pos_eig, n_pos_eig), dom.equalsTo(1.))
    model.constraint(
      expr.sub(zpos, zz),
      dom.equalsTo(0)
    )
    model.constraint(
      expr.sub(yy.diag(), ypos),
      dom.equalsTo(0)
    )
  
  xp = expr.mul(qpos, z)
  xn = expr.mul(qneg, z)
  
  # RLT cuts
  if rlt:
    # this means you can place on x directly.
    rlt_expr = expr.sub(expr.sum(y), expr.dot(bounds.xlb + bounds.xub, x))
    model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
  
  for i in range(m):
    pass
    # apos, ipos = qp.Apos[i]
    # aneg, ineg = qp.Aneg[i]
    # quad_expr = expr.sub(expr.dot(a[i], x), b[i])
    #
    # if ipos.shape[0] + ineg.shape[0] > 0:
    #
    #   # if it is indeed quadratic
    #   zconei = model.variable(f"zci@{i}", dom.inPSDCone(2, n))
    #   yi = zconei.slice([0, 0, 0], [n, 1, 1]).reshape([n, 1])
    #   zi = zconei.slice([0, 0, 1], [n, 1, 2]).reshape([n, 1])
    #   Y.append(yi)
    #   Z.append(zi)
    #
    #   el = qp.Amul[i]
    #
    #   # Z[-1, -1] == 1
    #   for idx in range(n):
    #     model.constraint(zconei.index([idx, 1, 1]), dom.equalsTo(1))
    #
    #   # A.T @ x == z
    #   model.constraint(
    #     expr.sub(
    #       expr.mul((apos + aneg), zi),
    #       x), dom.equalsTo(0))
    #
    #   if rlt:
    #     # this means you can place on x directly.
    #     rlt_expr = expr.sub(expr.sum(yi), expr.dot(bounds.xlb + bounds.xub, x))
    #     model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
    #
    #   if lk:
    #     lk_expr = expr.sub(expr.sum(yi), expr.sum(y))
    #     model.constraint(lk_expr, dom.equalsTo(0))
    #
    #   quad_terms = expr.dot(el, yi)
    #
    #   quad_expr = expr.add(quad_expr, quad_terms)
    #
    # else:
    #   Y.append(None)
    #   Z.append(None)
    #
    # quad_dom = dom.equalsTo(0) if sign[i] == 0 else (dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0))
    #
    # model.constraint(
    #   quad_expr, quad_dom)
  
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


def msc_part_relaxation(
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
  model = mf.Model('msc_sdp_msk_partition')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  if bounds is None:
    bounds = MscBounds.construct(qp)
  
  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg
  
  qel = qp.Qmul
  
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  Ym = Z.slice([0, 0], [n, n])
  xp = Z.slice([0, n], [n, n + 1])
  x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  y = model.variable("y", [*xshape], dom.greaterThan(0))
  z = model.variable("z", [*xshape])
  Y = [y]
  Z = [z]
  
  # Q.T x = Z
  model.constraint(
    expr.sub(
      expr.mul((qneg + qpos), z),
      x), dom.equalsTo(0))
  # x+ = V+z
  model.constraint(
    expr.sub(xp, expr.mul(qpos, z)),
    dom.equalsTo(0)
  )
  # conic
  # y >= z^2
  for idx in range(n):
    model.constraint(expr.vstack(0.5, y.index([idx, 0]), z.index([idx, 0])),
                     dom.inRotatedQCone())
  # Y+ = V+^TYV+
  n_pos_eig = qipos.shape[0]
  if n_pos_eig > 0:
    # else the problem is convex
    arr_pos_dim = np.zeros(n_pos_eig).astype(int).tolist()
    ypos = y.pick(qipos.tolist(), arr_pos_dim)
    Yp = expr.mul(
      expr.mul(qpos.T, Ym),
      qpos
    )
    model.constraint(expr.sub(Yp.pick([[j, j] for j in qipos]), ypos), dom.lessThan(0))
    model.constraint(expr.mul(qneg.T, Ym), dom.equalsTo(0))
    model.constraint(expr.mul(Ym, qneg), dom.equalsTo(0))
  
  # RLT cuts
  if rlt:
    # this means you can place on x directly.
    rlt_expr = expr.sub(expr.sum(y), expr.dot(bounds.xlb + bounds.xub, x))
    model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
  
  for i in range(m):
    pass
    # apos, ipos = qp.Apos[i]
    # aneg, ineg = qp.Aneg[i]
    # quad_expr = expr.sub(expr.dot(a[i], x), b[i])
    #
    # if ipos.shape[0] + ineg.shape[0] > 0:
    #
    #   # if it is indeed quadratic
    #   zconei = model.variable(f"zci@{i}", dom.inPSDCone(2, n))
    #   yi = zconei.slice([0, 0, 0], [n, 1, 1]).reshape([n, 1])
    #   zi = zconei.slice([0, 0, 1], [n, 1, 2]).reshape([n, 1])
    #   Y.append(yi)
    #   Z.append(zi)
    #
    #   el = qp.Amul[i]
    #
    #   # Z[-1, -1] == 1
    #   for idx in range(n):
    #     model.constraint(zconei.index([idx, 1, 1]), dom.equalsTo(1))
    #
    #   # A.T @ x == z
    #   model.constraint(
    #     expr.sub(
    #       expr.mul((apos + aneg), zi),
    #       x), dom.equalsTo(0))
    #
    #   if rlt:
    #     # this means you can place on x directly.
    #     rlt_expr = expr.sub(expr.sum(yi), expr.dot(bounds.xlb + bounds.xub, x))
    #     model.constraint(rlt_expr, dom.lessThan(- (bounds.xlb * bounds.xub).sum()))
    #
    #   if lk:
    #     lk_expr = expr.sub(expr.sum(yi), expr.sum(y))
    #     model.constraint(lk_expr, dom.equalsTo(0))
    #
    #   quad_terms = expr.dot(el, yi)
    #
    #   quad_expr = expr.add(quad_expr, quad_terms)
    #
    # else:
    #   Y.append(None)
    #   Z.append(None)
    #
    # quad_dom = dom.equalsTo(0) if sign[i] == 0 else (dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0))
    #
    # model.constraint(
    #   quad_expr, quad_dom)
  
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
