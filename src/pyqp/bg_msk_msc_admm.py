from pyqp.classes import BCParams, ADMMParams
from .bg_msk_msc import *
import time


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
      
      self.xival = self.xivar.level().reshape(self.xivar.getShape())
    else:  # infeasible
      self.relax_obj = -1e6
      print(status)
      if status == mf.ProblemStatus.Unknown:
        self.problem.writeTask("/tmp/dump.task.gz")
        self.problem.writeTask("/tmp/dump.mps")
    self.solved = True
    self.solve_time = round(end_time - start_time, 3)
    self.release()
  
  def release(self):
    self.problem.dispose()
  
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
    self.release()
  
  def release(self):
    self.problem.dispose()
  
  def check(self, qp: QP):
    pass


def msc_admm(
    qp: QP,  # the QP instance
    bounds: MscBounds = None,
    verbose=True,
    admmparams: ADMMParams = ADMMParams(),
    ws_result=None,
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
  if ws_result is None or ws_result.zval is None:
    zval = np.ones((n, dim))
    mu = np.zeros((n, dim))
  else:
    zval = ws_result.zval
    mu = ws_result.mu
  
  rho = 2
  xival = zval
  
  # test run
  
  _iter = 0
  start_time = time.time()
  while _iter < admmparams.max_iteration:
    r = msc_subproblem_x(
      xival, mu, rho, qp, bounds, solve=False, verbose=False
    )
    
    r.solve()
    xval = r.xval
    zval = r.zval
    yval = r.yval
    
    r_xi = msc_subproblem_xi(
      yval, zval, mu, rho, qp, bounds, solve=False, verbose=False
    )
    r_xi.solve()
    # compute residual. then update
    xival_res = r_xi.xival - xival
    xival = r_xi.xival
    
    # primal feasibility
    residual_xix = yval - (xival * zval)
    residual_xix_sum = np.abs(residual_xix).sum()
    # dual feasibility
    residual_dual = np.abs(zval * xival_res).sum()
    
    r.bound = (r.yval.T @ qp.Qmul).trace() + (qp.q.T @ xval).trace()
    gap = abs((r.bound - r.relax_obj) / (r.bound + 1e-2))
    curr_time = time.time()
    adm_time = curr_time - start_time
    if verbose and _iter % admmparams.logging_interval == 0:
      print(
        f"//{curr_time - start_time: .2f}, @{_iter}  #alm: {r.relax_obj:.4f} primal_eps (𝜉z - y): {residual_xix_sum:.4e}, dual_eps: {residual_dual:.4e}"
      )
    
    if residual_xix_sum < admmparams.res_gap and residual_dual < admmparams.res_gap:
      print(
        f"//{curr_time - start_time: .2f}, @{_iter}  #alm: {r.relax_obj:.4f} primal_eps (𝜉z - y): {residual_xix_sum:.4e}, dual_eps: {residual_dual:.4e}"
      )
      break
    if adm_time >= admmparams.time_limit:
      break
    mu += residual_xix * rho
    _iter += 1
  
  r.mu = mu
  r.nodes = _iter
  r.solve_time = adm_time
  r.true_obj = qp_obj_func(qp.Q, qp.q, xval)
  r.bound = (r.yval.T @ qp.Qmul).trace() + (qp.q.T @ xval).trace()
  return r


def msc_subproblem_x(  # follows the args
    xi,
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
  
  qel = qp.gamma[0].reshape(xshape)
  qcones = model.variable("xr", dom.inRotatedQCone(3, n))
  ones = qcones.slice([0, 0], [1, n])
  y = qcones.slice([1, 0], [2, n]).reshape(n, 1)
  z = qcones.slice([2, 0], [3, n]).reshape(n, 1)
  model.constraint(ones, dom.equalsTo(0.5))
  s = model.variable('sqr', [m + 1])
  x = model.variable("x", [*xshape], dom.inRange(bounds.xlb, bounds.xub))
  Y = [y]
  Z = [z]
  
  # Q.T x = Z
  model.constraint(expr.sub(expr.mul(qp.U[0].T, x), z), dom.equalsTo(0))
  if hasattr(bounds, 'zlb'):
    model.constraint(z, dom.inRange(bounds.zlb, bounds.zub))
  
  for i in range(m):
    quad_expr = expr.dot(a[i], x)
    if not qp.bool_zero_mat[i + 1]:
      si = s.index(i + 1)
      model.constraint(
        expr.vstack(0.5, si, expr.flatten(expr.mul(qp.R[i + 1].T, x))),
        dom.inRotatedQCone()
      )
      quad_expr = expr.add(quad_expr, si)
      quad_expr = expr.sub(quad_expr, expr.dot(qp.l[i + 1] * np.ones((n, 1)), y))
    
    if qp.sign is not None:
      # unilateral case
      quad_dom = dom.equalsTo(b[i]) if sign[i] == 0 else dom.lessThan(b[i])
    
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
  
  ###################
  # The above part is unchanged
  ###################
  
  # ADMM solves the minimization problem so we reverse the max objective.
  true_obj_expr = expr.add(expr.dot(-q, x), expr.dot(-qel, y))
  
  # ALM terms
  # the <𝜉, x> - gap
  expr_norm_x_gap = expr.sub(y, expr.mulElm(xi, z))
  expr_norm_x_gap_sqr = model.variable("xi_z", dom.greaterThan(0))
  model.constraint(
    expr.vstack(1 / 2, expr_norm_x_gap_sqr, expr.flatten(expr_norm_x_gap)),
    dom.inRotatedQCone()
  )
  
  # ALM objective
  obj_expr = true_obj_expr
  obj_expr = expr.add(obj_expr, expr.dot(mu, expr_norm_x_gap))
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
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r


def msc_subproblem_xi(  # follows the args
    y,
    z,
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
  
  n = qp.n
  model = mf.Model('many_small_cone_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  ###################
  # The above part is unchanged
  ###################
  # bounds on xi
  qcones = model.variable("xi_y", dom.inRotatedQCone(3, n))
  ones = qcones.slice([0, 0], [1, n])
  yr = qcones.slice([1, 0], [2, n]).reshape(n, 1)
  xi = qcones.slice([2, 0], [3, n]).reshape(n, 1)
  model.constraint(ones, dom.equalsTo(0.5))
  model.constraint(yr, dom.equalsTo(y))
  # if hasattr(bounds, 'zlb'):
  #   model.constraint(xi, dom.inRange(bounds.zlb, bounds.zub))
  
  # ADMM solves the minimization problem so we reverse the max objective.
  # ALM terms
  
  # the <𝜉, x> - gap
  expr_norm_x_gap = expr.sub(y, expr.mulElm(xi, z))
  expr_norm_x_gap_sqr = expr.add(
    [
      # (y ** 2).sum() * rho / 2,
      expr.dot(z ** 2 * rho / 2, yr),
      expr.dot(- y * z * rho, xi)
    ]
  )
  # ALM objective
  obj_expr = expr.add(
    [expr.dot(mu, expr_norm_x_gap),
     expr.sum(expr_norm_x_gap_sqr)]
  )
  
  model.objective(mf.ObjectiveSense.Minimize, obj_expr)
  
  r = MSKResultXi()
  r.obj_expr = obj_expr
  r.problem = model
  r.xivar = xi
  if not solve:
    return r
  
  r.solve(verbose=verbose, qp=qp)
  
  return r
