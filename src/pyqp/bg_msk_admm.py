from .bg_msk_msc import *


def msc_subproblem_x(  # follows the args
    xi,
    s,
    kappa,
    mu,
    rho,
    qp: QP,
    bounds: MscBounds = None,
    verbose=True,
    solve=True,
    *args,
    **kwargs):
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
  model = mf.Model('admm_subproblem_x')

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
  rlt_expr = expr.sub(expr.sum(y), expr.dot(bounds.xlb + bounds.xub, x))
  model.constraint(rlt_expr, dom.lessThan(-(bounds.xlb * bounds.xub).sum()))

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
      model.constraint(expr.sub(expr.mul((apos + aneg), zi), x),
                       dom.equalsTo(0))

      # this means you can place on x directly.
      rlt_expr = expr.sub(expr.sum(yi), expr.dot(bounds.xlb + bounds.xub, x))
      model.constraint(rlt_expr, dom.lessThan(-(bounds.xlb * bounds.xub).sum()))

      quad_terms = expr.dot(el, yi)

      quad_expr = expr.add(quad_expr, quad_terms)

    else:
      Y.append(None)
      Z.append(None)

    quad_dom = dom.equalsTo(0) if sign[i] == 0 else (
      dom.greaterThan(0) if sign[i] == -1 else dom.lessThan(0))

    model.constraint(quad_expr, quad_dom)

  ###################
  # The above part is unchanged
  ###################
  # norm bounds on y^Te
  t = model.variable("t", dom.inRange(0, n))
  for idx, yi in enumerate(Y):
    model.constraint(expr.sub(expr.sum(yi), t), dom.lessThan(0))

  # ADMM solves the minimization problem so we reverse the max objective.
  true_obj_expr = expr.add(expr.dot(-q, x), expr.dot(-qel, y))
  # ALM terms
  expr_norm_gap = expr.sub(t, s)
  expr_norm_gap_sqr = model.variable("t_s", dom.greaterThan(0))
  model.constraint(expr.vstack(1 / 2, expr_norm_gap_sqr, expr_norm_gap),
                   dom.inRotatedQCone())
  expr_norm_x_gap = expr.sub(expr.dot(xi, x), s)
  expr_norm_x_gap_sqr = model.variable("xi_x", dom.greaterThan(0))
  model.constraint(expr.vstack(1 / 2, expr_norm_gap_sqr, expr_norm_x_gap),
                   dom.inRotatedQCone())

  # ALM objective
  obj_expr = true_obj_expr
  obj_expr = expr.add(obj_expr, expr.mul(kappa, expr_norm_gap))
  obj_expr = expr.add(obj_expr, expr.mul(mu, expr_norm_x_gap))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_gap_sqr))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_x_gap_sqr))

  model.objective(mf.ObjectiveSense.Minimize, obj_expr)

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
  r.t = t
  if not solve:
    return r

  r.solve(verbose=verbose, qp=qp)

  return r


class MSKResultXi():
  pass


def msc_subproblem_xi(  # follows the args
    x,
    y,
    z,
    t,
    kappa,
    mu,
    rho,
    qp: QP,
    bounds: MscBounds = None,
    verbose=True,
    solve=True,
    *args,
    **kwargs):
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
  model = mf.Model('many_small_cone_msk')

  if verbose:
    model.setLogHandler(sys.stdout)

  if bounds is None:
    bounds = MscBounds.construct(qp)

  qpos, qipos = qp.Qpos
  qneg, qineg = qp.Qneg

  qel = qp.Qmul

  ###################
  # The above part is unchanged
  ###################
  # norm bounds on y^Te
  xi = model.variable("xi", dom.inRange(bounds.xlb, bounds.xub))
  s = model.variable("s", dom.inRange(0, n))

  # ADMM solves the minimization problem so we reverse the max objective.
  # ALM terms
  expr_norm_gap = expr.sub(t, s)
  expr_norm_gap_sqr = model.variable("t_s", dom.greaterThan(0))
  model.constraint(expr.vstack(1 / 2, expr_norm_gap_sqr, expr_norm_gap),
                   dom.inRotatedQCone())
  expr_norm_x_gap = expr.sub(expr.dot(xi, x), s)
  expr_norm_x_gap_sqr = model.variable("xi_x", dom.greaterThan(0))
  model.constraint(expr.vstack(1 / 2, expr_norm_gap_sqr, expr_norm_x_gap),
                   dom.inRotatedQCone())

  # ALM objective
  obj_expr = 0
  obj_expr = expr.add(obj_expr, expr.mul(kappa, expr_norm_gap))
  obj_expr = expr.add(obj_expr, expr.mul(mu, expr_norm_x_gap))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_gap_sqr))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_x_gap_sqr))

  model.objective(mf.ObjectiveSense.Minimize, obj_expr)

  r = MSKResultXi()
  r.obj_expr = obj_expr
  r.xivar = xi
  r.problem = model
  r.s = s
  if not solve:
    return r

  r.solve(verbose=verbose, qp=qp)

  return r