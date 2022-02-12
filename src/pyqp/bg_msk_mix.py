"""
Mixed cone approach
"""
import numpy as np

from .bg_msk_norm import *


class MSKMixConeResult(MSKSocpResult):
  def __init__(self):
    super(MSKMixConeResult, self).__init__()
    self.sdpc = None
    self.sdpcval = None
    # keep edges
    self.edges = set()
  
  def solve(self, verbose=False, qp=None):
    super(MSKMixConeResult, self).solve(verbose, qp)
    # sdpc
    if self.sdpc:
      self.sdpcval = self.sdpc.level().reshape(self.sdpc.getShape())


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
  #
  edges = kwargs.get('edges', set())
  n_edges = len(edges)
  sdpc = model.variable('x0', dom.inPSDCone(3, n_edges)) if n_edges > 0 else None
  # sdp cone connects to socp cone
  for idx, (i, j) in enumerate(edges):
    model.constraint(
      expr.sub(
        sdpc.index([idx, 0, 0]), rho.index([i, 0])
      ), dom.equalsTo(0)
    )
    model.constraint(
      expr.sub(
        sdpc.index([idx, 1, 1]), rho.index([j, 0])
      ), dom.equalsTo(0)
    )
    model.constraint(
      sdpc.index([idx, 2, 2])
      , dom.equalsTo(1)
    )
    model.constraint(
      expr.sub(
        expr.flatten(sdpc.slice([idx, 2, 0], [idx + 1, 3, 2])),
        x.pick([[i, 0], [j, 0]])
      ), dom.equalsTo(0)
    )
  
  if Q is not None:
    # decompose Q into
    # Q1 ⊕ ∑ Q_c
    # create the complement matrix
    Q1 = Q.copy()
    small_cone_sum = 0
    if n_edges > 0:
      Q0 = np.empty((n_edges, 2, 2))
      
      for idx, (i, j) in enumerate(edges):
        # now build new cholesky
        Q0[idx, 0, 0] = Q1[i, i]
        Q0[idx, 0, 1] = Q0[idx, 1, 0] = Q1[i, j]
        Q0[idx, 1, 1] = Q1[j, j]
        Q1[i, i] = Q1[i, j] = Q1[j, i] = Q1[j, j] = 0
      
      small_cone_sum = expr.dot(
        Q0.flatten(),
        expr.flatten(sdpc.slice([0, 0, 0], [n_edges, 2, 2]))
      )
    
    # Q1 = Q
    l, R = QP._scaled_cholesky(n, - Q1)
    model.constraint(
      expr.vstack(0.5, y.index(0), expr.flatten(expr.mul(R.T, x))),
      dom.inRotatedQCone()
    )
    partial_sum = expr.add([
      expr.mul(l, s),
      expr.mul(-1, z),
      expr.dot(q, x),
      expr.mul(-1, y.index(0)),
    ])
    model.constraint(
      expr.add(
        partial_sum,
        small_cone_sum
      ),
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
  
  r = MSKMixConeResult()
  r.obj_expr = z
  r.edges = edges
  r.sdpc = sdpc
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
