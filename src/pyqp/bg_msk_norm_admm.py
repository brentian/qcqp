"""
admm based on the norm constrained formulations,
using bilinear term:
- ξ · x = ρ · e
- ρ = (ξ ◦ x)
"""
from .bg_msk_msc import *
import time

from .classes import ADMMParams


class MSKResultXi(MSKMscResult):
  """Result keeper for ADMM subproblem
  for (xi)
  """

  def __init__(self):
    super().__init__()

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
      self.sval = self.svar.level()[0]
      self.xival = self.xivar.level().reshape(self.xivar.getShape())
    else:  # infeasible
      self.relax_obj = -1e6
      print(status)
      if status == mf.ProblemStatus.Unknown:
        self.problem.writeTask("/tmp/dump.task.gz")
        self.problem.writeTask("/tmp/dump.mps")
    self.solved = True
    self.solve_time = round(end_time - start_time, 3)

  def check(self, qp: QP):
    pass


class MSKResultX(MSKMscResult):
  """Result keeper for ADMM subproblem
  for (x,y,z)
  """

  def __init__(self):
    super().__init__()

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
      self.xval = self.xvar.level().reshape(self.xvar.getShape()).round(6)
      self.zval = self.zvar.level().reshape(self.zvar.getShape()).round(6)
      self.Zval = np.hstack(
        [
          xx.level().reshape(self.xvar.getShape()).round(6)
          if xx is not None else np.zeros(self.xvar.getShape())
          for xx in self.Zvar
        ]
      )
      if self.yvar is not None:
        self.yval = self.yvar.level().reshape(self.yvar.getShape()).round(6)
      if self.Yvar is not None:
        self.Yval = np.hstack(
          [
            xx.level().reshape(self.xvar.getShape()).round(6)
            if xx is not None else np.zeros(self.xvar.getShape())
            for xx in self.Yvar
          ]
        )

      if self.Dvar is not None:
        self.Dval = np.hstack(
          [
            xx.level().reshape((2, 1)).round(6) if xx is not None else np.zeros(
              (2, 1)
            ) for xx in self.Dvar
          ]
        )
      self.tval = self.tvar.level()[0]
      self.relax_obj = -self.problem.primalObjValue()
      if qp is not None:
        self.true_obj = qp_obj_func(qp.Q, qp.q, self.xval)
    else:  # infeasible
      self.relax_obj = -1e6
      print(status)
      if status == mf.ProblemStatus.Unknown:
        self.problem.writeTask("/tmp/dump.task.gz")

    self.solved = True
    self.solve_time = round(end_time - start_time, 3)

    def check(self, qp: QP):
      pass


def msc_admm(
  qp: QP,  # the QP instance, must be decomposed by method II
  bounds: MscBounds = None,
  verbose=False,
  admmparams: ADMMParams = ADMMParams(),
  *args,
  **kwargs
):
  _unused = kwargs
  if qp.Qpos is None:
    raise ValueError("decompose QP instance first")
  if qp.decom_method == 'eig-type1':
    raise ValueError(f"cannot use {qp.decom_method}")
  m, n, dim = qp.a.shape
  ########################
  # initialization
  ########################
  xval = np.ones((n, dim))
  sval = (xval.T @ xval).trace()
  rho = 1
  kappa = 0
  mu = 0
  xival = xval
  # test run

  _iter = 0
  start_time = time.time()
  while _iter < admmparams.max_iteration:
    r = msc_subproblem_x(
      sval, xival, kappa, mu, rho, qp, bounds, solve=False, verbose=verbose
    )
    r.solve()
    xval = r.xval
    tval = r.tval
    r_xi = msc_subproblem_xi(
      xval, tval, kappa, mu, rho, qp, bounds, solve=False, verbose=verbose
    )
    r_xi.solve()
    sval = r_xi.sval
    xival = r_xi.xival
    residual_ts = tval - sval
    residual_xix = (xival * xval).sum() - tval
    r.bound = (r.yval.T @ qp.Qmul).trace() + (qp.q.T @ xval).trace()
    gap = abs((r.bound - r.relax_obj) / (r.bound + 1e-2))
    curr_time = time.time()
    adm_time = curr_time - start_time
    if  _iter % admmparams.logging_interval == 0:
      print(
        f"//{curr_time - start_time: .2f}, @{_iter} # alm: {r.relax_obj: .4f} gap: {gap:.3%} norm t - s: {residual_ts: .4e}, 𝜉x - t: {residual_xix: .4e}"
      )
    if (gap < admmparams.obj_gap) and (max(abs(residual_ts), abs(residual_xix)) < admmparams.res_gap):
      print(
        f"//{curr_time - start_time: .2f}, @{_iter} # alm: {r.relax_obj: .4f} gap: {gap:.3%} norm t - s: {residual_ts: .4e}, 𝜉x - t: {residual_xix: .4e}"
      )
      print(f"terminited by gap")
      break
    if adm_time >= admmparams.time_limit:
      break
    kappa += residual_ts * rho
    mu += residual_xix * rho
    _iter += 1
  r.result_xi = r_xi
  r.nodes = _iter
  r.solve_time = adm_time
  r.true_obj = qp_obj_func(qp.Q, qp.q, xval)
  r.bound = (r.yval.T @ qp.Qmul).trace() + (qp.q.T @ xval).trace()
  return r


def msc_subproblem_x(  # follows the args
    s,
    xi,
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
  model.constraint(expr.sub(expr.mul((qneg + qpos).T, x), z), dom.equalsTo(0))
  
  for i in range(m):
    apos, ipos = qp.Apos[i]
    aneg, ineg = qp.Aneg[i]
    quad_expr = expr.dot(a[i], x)

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
        expr.sub(expr.mul((apos + aneg).T, x), zi), dom.equalsTo(0)
      )
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
      quad_dom = dom.inRange(qp.al[i], qp.au[i])
      # quad_dom = dom.lessThan(qp.au[i])
    model.constraint(quad_expr, quad_dom)
  
  ###################
  # The above part is unchanged
  ###################
  # norm bounds on y^Te
  t = model.variable("t", dom.inRange(0, s))
  for idx, yi in enumerate(Y):
    if yi is not None:
      model.constraint(expr.sub(expr.sum(yi), t), dom.lessThan(0))
    else:
      print("ssss")

  # ADMM solves the minimization problem so we reverse the max objective.
  true_obj_expr = expr.add(expr.dot(-q, x), expr.dot(-qel, y))

  # ALM terms
  # the t - s gap
  expr_norm_gap = expr.sub(t, s)
  expr_norm_gap_sqr = model.variable("t_s", dom.greaterThan(0))
  model.constraint(
    expr.vstack(1 / 2, expr_norm_gap_sqr, expr_norm_gap), dom.inRotatedQCone()
  )
  # the <𝜉, x> - gap
  expr_norm_x_gap = expr.sub(expr.dot(xi, x), t)
  expr_norm_x_gap_sqr = model.variable("xi_x", dom.greaterThan(0))
  model.constraint(
    expr.vstack(1 / 2, expr_norm_x_gap_sqr, expr_norm_x_gap),
    dom.inRotatedQCone()
  )

  # ALM objective
  obj_expr = true_obj_expr
  obj_expr = expr.add(obj_expr, expr.mul(kappa, expr_norm_gap))
  obj_expr = expr.add(obj_expr, expr.mul(mu, expr_norm_x_gap))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_gap_sqr))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_x_gap_sqr))

  model.objective(mf.ObjectiveSense.Minimize, obj_expr)

  r = MSKResultX()
  r.obj_expr = true_obj_expr
  r.xvar = x
  r.yvar = y
  r.zvar = z
  r.Zvar = Z
  r.Yvar = Y
  r.qel = qel
  r.q = q
  r.problem = model
  r.tvar = t
  if not solve:
    return r

  r.solve(verbose=verbose, qp=qp)

  return r



def msc_subproblem_xi(  # follows the args
    x,
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
  
  xshape = (qp.n, qp.d)
  model = mf.Model('many_small_cone_msk')

  if verbose:
    model.setLogHandler(sys.stdout)

  ###################
  # The above part is unchanged
  ###################
  # norm bounds on y^Te
  xi = model.variable("xi", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  s = model.variable("s", dom.greaterThan(0))

  model.constraint(
    expr.vstack(1 / 2, s, expr.flatten(xi)), dom.inRotatedQCone()
  )
  # ADMM solves the minimization problem so we reverse the max objective.
  # ALM terms
  # the t - s gap
  expr_norm_gap = expr.sub(t, s)
  expr_norm_gap_sqr = model.variable("t_s", dom.greaterThan(0))
  model.constraint(
    expr.vstack(1 / 2, expr_norm_gap_sqr, expr_norm_gap), dom.inRotatedQCone()
  )
  # the <𝜉, x> - gap
  expr_norm_x_gap = expr.sub(expr.dot(xi, x), t)
  expr_norm_x_gap_sqr = model.variable("xi_x", dom.greaterThan(0))
  model.constraint(
    expr.vstack(1 / 2, expr_norm_x_gap_sqr, expr_norm_x_gap),
    dom.inRotatedQCone()
  )
  # ALM objective
  obj_expr = 0
  obj_expr = expr.add(obj_expr, expr.mul(kappa, expr_norm_gap))
  obj_expr = expr.add(obj_expr, expr.mul(mu, expr_norm_x_gap))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_gap_sqr))
  obj_expr = expr.add(obj_expr, expr.mul(rho / 2, expr_norm_x_gap_sqr))

  model.objective(mf.ObjectiveSense.Minimize, obj_expr)

  r = MSKResultXi()
  r.obj_expr = obj_expr
  r.problem = model
  r.xivar = xi
  r.svar = s
  if not solve:
    return r

  r.solve(verbose=verbose, qp=qp)

  return r