"""
Mixed cone approach
"""

from .bg_msk_norm import *


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
  model.constraint(x, dom.inRange(bounds.xlb, bounds.xub))
  # s = x^Tx
  s = model.variable("s")
  # y = x^TRR^Tx
  y = model.variable("y", [m + 1])
  # s = rho^Te
  model.constraint(
    expr.sub(expr.sum(rho), s), dom.equalsTo(0)
  )
  
  # R.T x = Z
  if Q is not None:
    i, j = n - 2, n - 1
    # create a small sdp cone,
    sdpc = model.variable('x0', dom.inPSDCone(3))
    model.constraint(
      expr.sub(
        sdpc.index([0, 0]), rho.index([i, 0])
      ), dom.equalsTo(0)
    )
    model.constraint(
      expr.sub(
        sdpc.index([1, 1]), rho.index([j, 0])
      ), dom.equalsTo(0)
    )
    model.constraint(
      sdpc.index([2, 2])
      , dom.equalsTo(1)
    )
    model.constraint(
      expr.sub(
        expr.flatten(sdpc.slice([2, 0], [3, 2])),
        x.pick([[i, 0], [j, 0]])
      ), dom.equalsTo(0)
    )
    # now build new cholesky
    Q0 = Q[i:j+1, i:j+1].copy()
    Q1 = Q.copy()
    Q1[i:j+1, i:j+1] = 0
    # Q1 = Q
    l, R = QP._scaled_cholesky(n, - Q1)
    model.constraint(
      expr.vstack(0.5, y.index(0), expr.flatten(expr.mul(R.T, x))),
      dom.inRotatedQCone()
    )
    model.constraint(
      expr.add([
        expr.mul(l, s),
        expr.mul(-1, z),
        expr.dot(q, x),
        expr.mul(-1, y.index(0)),
        expr.dot(Q0, sdpc.slice([0, 0], [2, 2]))
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
  
  # RLT for ρ = (ξ ◦ x)
  model.constraint(
    expr.sub(rho, expr.mulElm(bounds.xub + bounds.xlb, x)),
    dom.lessThan(-bounds.xlb * bounds.xub)
  )
  
  for i in range(m):
    # todo,
    # todo, not finished for constrained case
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
