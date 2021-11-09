import numpy as np

from pyqptest.helpers import *
from pyqp.bg_msk import *


def shor(
    qp: QP,
    bounds: Bounds,
    sense="max",
    verbose=True,
    solve=True,
    r_parent: MSKResult = None,
    **kwargs
):
  """
  use a Y along with x in the SDP
      for basic 0 <= x <= e, diag(Y) <= x
  Parameters
  ----------
  Q
  q
  A
  a
  b
  sign
  lb
  ub
  solver
  sense
  kwargs

  Returns
  -------

  """
  _unused = kwargs
  st_time = time.time()
  
  Q, q, A, a, b, sign, *_ = qp.unpack()
  Qh = qp.Qh
  Ah = qp.Ah
  lb, ub, ylb, yub = bounds.unpack()
  m, n, d = a.shape
  xshape = (n, d)
  model = mf.Model('shor_msk')
  
  if verbose:
    model.setLogHandler(sys.stdout)
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  
  # bounds
  # model.constraint(expr.sub(x, ub), dom.lessThan(0))
  # model.constraint(expr.sub(Z.diag(), 1), dom.lessThan(0))
  model.constraint(Z.index(n, n), dom.equalsTo(1.))
  
  for i in range(m):
    if sign[i] == 0:
      model.constraint(
        expr.sum(expr.mulElm(Z, Ah[i])),
        dom.equalsTo(0))
    elif sign[i] == -1:
      model.constraint(
        expr.sum(expr.mulElm(Z, Ah[i])),
        dom.greaterThan(0))
    else:
      model.constraint(
        expr.sum(expr.mulElm(Z, Ah[i])),
        dom.lessThan(0))
  #
  # objectives
  obj_expr = expr.sum(expr.mulElm(Qh, Z))
  model.objective(mf.ObjectiveSense.Minimize
                  if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
  
  r = MSKResult(qp, model, zvar=Z)
  r.start_time = st_time
  r.build_time = time.time() - st_time
  # if not solve:
  #   return r
  # r.solve()
  # if verbose:
  #   print(r.build_time, r.total_time, r.solve_time)
  model.solve()
  zval = Z.level().reshape(Z.getShape())
  r.zval = zval
  t = zval[-1, -1]
  r.xval = zval[:-1, n:n + 1] / t
  r.yval = zval[:-1, :-1] / t
  
  return r


if __name__ == '__main__':
  params = BCParams()
  admmparams = ADMMParams()
  kwargs, r_methods = params.produce_args(parser, METHOD_CODES)
  _ = admmparams.produce_args(parser, METHOD_CODES)
  
  np.random.seed(1)
  m = 1
  n = 100
  qp = QPI.normal(n, m, rho=0.7)
  qp.A[0] = qp.A[0].T @ qp.A[0]
  qp.Q = np.random.randint(0, 10, (n, n))
  qp.Q = (qp.Q  + qp.Q.T) / 2
  qp.construct_homo()
  bd = Bounds(xlb=np.zeros(shape=(n, 1)), xub=np.ones(shape=(n, 1)))
  
  r = shor(qp, bd)
  
  # do r1 decomp
  X = r.zval

  A = qp.Ah[0]
  
  e, V = np.linalg.eig(X)
  
  Y = V @ np.diag(np.sqrt(e))
  
  # A = qp.Qh.__array__()
  rho = (A @ X).trace() / np.linalg.matrix_rank(X, 1e-11)
  v_ind = np.ndarray(n + 1)
  for i in range(n + 1):
    v_ind[i] = (Y[:, i:i + 1].T @ A @ Y[:, i:i + 1]).trace()
  
    # now compute
  y1 = Y[:, 0:1]
  for j in range(1, n + 1):
    if (v_ind[j] - rho) * (v_ind[0] - rho) < 0:
      yj = Y[:, j:j + 1]
      break
  
  # solve
  # a^2 vj + 2a v_ij + vi = (A @ X).trace()/r
  vij = (y1.T @ A @ yj).trace()
  
  alpha = ( vij + np.sqrt(vij**2 - v_ind[j] * (v_ind[0] - rho))) / v_ind[j]
  
  y = (y1  + yj * alpha) / np.sqrt(1 + alpha ** 2)
  # primal
  x = y / y[-1]
  
  print((x.T@qp.Qh @x).trace())
  print((X@qp.Qh).trace())
