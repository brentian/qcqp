import itertools
import numpy as np
import networkx as nx
import networkx.algorithms as na
from .bg_msk import *

####################################
# Chordal/Clique based SDP
####################################
from .bg_msk_msc import MSKMscResult
from .classes import MscBounds


class MSKChordalResult(MSKResult):
  extract_arr_of_vars = \
    lambda self, x_arr: [xx.level().reshape(xx.getShape()).round(4) for xx in x_arr]
  
  def __init__(self, qp: QP, problem: mf.Model):
    super().__init__(qp, problem)
    self.Yrvar = None
    self.Yivar = None
    self.yval = None
    self.Yival = None
    self.Yrval = None
    self.solved = False
    self.obj_expr = None
    self.qel = None
    self.q = None
  
  def solve(self, verbose=False):
    if verbose:
      self.problem.setLogHandler(sys.stdout)
    try:
      self.problem.solve()
      status = self.problem.getProblemStatus()
    except Exception as e:
      status = 'failed'
    end_time = time.time()
    if status == mf.ProblemStatus.PrimalAndDualFeasible:
      self.xval = self.xvar.level().reshape(self.xvar.getShape())
      self.relax_obj = self.problem.primalObjValue()
      self.solved = True
      self.solve_time = self.problem.getSolverDoubleInfo("optimizerTime")
      self.total_time = time.time() - self.start_time
      if self.Yrvar is not None:
        self.Yrval = self.extract_arr_of_vars(self.Yrvar)
      if self.Yivar is not None:
        self.Yival = self.extract_arr_of_vars(self.Yivar)
      sum_y = 0
      for k, cr in enumerate(self.qp.cc):
        er = self.qp.Er[k]
        sum_y += er.T @ self.Yrval[k] @ er
      for k, cr in enumerate(self.qp.ic):
        er = self.qp.Eir[k]
        sum_y -= er.T @ self.Yival[k] @ er
      self.yval = sum_y
      self.true_obj = qp_obj_func(self.qp.Q, self.qp.q, self.xval)
      
    else:  # infeasible
      self.relax_obj = -1e6
    self.solved = True


def greedy_agg_cc(cc, eps=40):
  agg_cc = []
  ac = set()
  selected = set()
  while cc:
    cl = set(cc.pop())
    lenc = len(cc)
    newx = cl.difference(selected)
    ac.update(cl)
    if len(ac) >= eps or lenc == 0:
      agg_cc.append(list(ac))
      selected.update(ac)
      ac = set()
  return agg_cc


def create_er_from_clique(cr, n):
  nr = len(cr)
  Er = np.zeros((nr, n))
  for row, col in enumerate(cr):
    Er[row, col] = 1
  return Er


####################################
# Primal
####################################
# @note
# this method is based on chordal graph extension
# original graph G(V, E) standing for data matrix
# 1. find chordal extension G(V, F) for G(V, E)
# 2. for chordal graph polynomial time alg exists for maximal cliques:
#   F ⊆ {Cr}
# 3. Y ⪰ 0 is eq to Yr ⪰ 0 for each r and Cr
def ssdp(
    qp: QP,
    bounds: Bounds,
    sense="max", verbose=True, solve=True, **kwargs
):
  """
  """
  _unused = kwargs
  st_time = time.time()
  Q, q, A, a, b, sign = qp.unpack()
  lb, ub, ylb, yub = bounds.unpack()
  m, n, d = a.shape
  xshape = (n, d)
  model = mf.Model('shor_clique_msk')
  
  ic, cc, er, eir = qp.ic, qp.cc, qp.Er, qp.Eir
  ccl = [len(cl) for cl in cc]
  if verbose:
    print(f"number of cliques {len(cc)}")
    print(f"clique size: {min(ccl), max(ccl)}")
    print(cc)
  
  x = model.variable("x", [*xshape], dom.inRange(lb, ub))
  Y = model.variable("Y", [n, n])
  Yr = []
  Yi = []
  # cliques
  sum_y = 0
  for k, cr in enumerate(cc):
    Er = er[k]
    nr = len(cr)
    zr = model.variable(f"Zr_{k}", dom.inPSDCone(nr + 1))
    yr = zr.slice([0, 0], [nr, nr])
    xr = zr.slice([0, nr], [nr, nr + 1])
    model.constraint(zr.index(nr, nr), dom.equalsTo(1.))
    # model.constraint(
    #   expr.sub(
    #     expr.mul(expr.mul(Er, Y), Er.T),
    #     yr),
    #   dom.equalsTo(0)
    # )
    xr_from_x = x.pick([[idx, 0] for idx in cr])
    model.constraint(
      expr.sub(xr_from_x, xr),
      dom.equalsTo(0)
    )
    model.constraint(
      expr.sub(yr.diag(), xr),
      dom.lessThan(0)
    )
    Qr = Er @ Q @ Er.T
    sum_y = expr.add(sum_y, expr.dot(Qr, yr))
    Yr.append(yr)
  
  
  # intersections
  for k, cr in enumerate(ic):
    Er = eir[k]
    nr = len(cr)
    zr = model.variable(f"Zir_{k}", dom.inPSDCone(nr + 1))
    yr = zr.slice([0, 0], [nr, nr])
    xr = zr.slice([0, nr], [nr, nr + 1])
    model.constraint(zr.index(nr, nr), dom.equalsTo(1.))
    # model.constraint(
    #   expr.sub(
    #     expr.mul(expr.mul(Er, Y), Er.T),
    #     yr),
    #   dom.equalsTo(0)
    # )
    xr_from_x = x.pick([[idx, 0] for idx in cr])
    model.constraint(
      expr.sub(xr_from_x, xr),
      dom.equalsTo(0)
    )
    model.constraint(
      expr.sub(yr.diag(), xr),
      dom.lessThan(0)
    )
    Qr = Er @ Q @ Er.T
    sum_y = expr.sub(sum_y, expr.dot(Qr, yr))
    Yi.append(yr)
  #
  # # constraints
  # for i in range(m):
  #   if sign[i] == 0:
  #     model.constraint(
  #       expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
  #       dom.equalsTo(b[i]))
  #   elif sign[i] == -1:
  #     model.constraint(
  #       expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
  #       dom.greaterThan(b[i]))
  #   else:
  #     model.constraint(
  #       expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
  #       dom.lessThan(b[i]))
  
  # objectives
  obj_expr = expr.add(sum_y, expr.dot(x, q))
  # obj_expr = expr.add(expr.sum(expr.mulElm(Q, Y)), expr.dot(x, q))
  model.objective(
    mf.ObjectiveSense.Minimize if sense == 'min'
    else mf.ObjectiveSense.Maximize, obj_expr
  )
  
  r = MSKChordalResult(qp, model)
  r.start_time = st_time
  r.build_time = time.time() - st_time
  r.xvar = x
  r.Yrvar = Yr
  r.Yivar = Yi
  #
  r.problem = model
  if not solve:
    return r
  r.solve(verbose=verbose)
  return r


# def ssdpblk(
#     qp: QP,
#     sense="max", verbose=True, solve=True, **kwargs
# ):
#   """
#   use a Y along with x in the SDP
#       for basic 0 <= x <= e, diag(Y) <= x
#   -------
#
#   """
#   _unused = kwargs
#   Q, q, A, a, b, sign, lb, ub, ylb, yub, cc = qp.unpack()
#   m, n, d = a.shape
#   xshape = (n, d)
#   model = mf.Model('shor_clique_msk')
#
#   if verbose:
#     model.setLogHandler(sys.stdout)
#
#   ########################
#   # cliques and chordal
#   ########################
#   if cc is None:
#     g = nx.Graph()
#     g.add_edges_from([(i, j) for i, j in zip(*Q.nonzero()) if i!=j])
#     g_chordal, alpha = na.complete_to_chordal_graph(g)
#     # cc is a list of maximal cliques
#     # e.g.:
#     # [[0, 4, 1], [0, 4, 3], [0, 4, 5], [2, 1], [2, 3]]
#     cc = na.chordal_graph_cliques(g_chordal)
#   ccl = [len(cl) for cl in cc]
#   print(f"number of cliques {len(cc)}")
#   print(f"clique size: {min(ccl), max(ccl)}")
#   print(cc)
#
#   ########################
#   # compute running intersections
#   ########################
#   mc = []
#   ac = set()
#   for cl in cc:
#     newm = set(cl).difference(ac)
#     if len(newm) > 0:
#       mc.append(newm)
#       ac.update(newm)
#     if len(ac) == n:
#       break
#
#   x = model.variable("x", [*xshape])
#   # Y = model.variable("Y", [n, n])
#
#   # bounds
#   model.constraint(expr.sub(x, ub), dom.lessThan(0))
#   model.constraint(expr.sub(x, lb), dom.greaterThan(0))
#   #
#
#   # cliques
#   sum_y = 0
#   ee = 0
#   for k, cr in enumerate(cc):
#     Er = create_er_from_clique(cr, n)
#     nr = len(cr)
#     ee += Er.T @ Er
#     # Zr = model.variable(f"Zr_{k}", dom.inPSDCone(nr + 1))
#     Yr = model.variable(f"Yr_{k}", dom.inPSDCone(nr))
#     # xr = Zr.slice([0, nr], [nr, nr + 1])
#     # model.constraint(Zr.index(nr, nr), dom.equalsTo(1.))
#     # xr_from_x = x.pick([[idx, 0] for idx in cr])
#     # model.constraint(
#     #   expr.sub(xr_from_x, xr),
#     #   dom.equalsTo(0)
#     # )
#     # model.constraint(
#     #   expr.sub(Yr.diag(), xr_from_x),
#     #   dom.lessThan(0)
#     # )
#     # if k + 1 <= len(mc):
#     #   zero_cols = list(set(cr).difference(mc[k]))
#     #   # Er[:, zero_cols] = 0
#     sum_y = expr.add(sum_y, expr.mul(expr.mul(Er.T, Yr), Er))
#   model.constraint(
#     expr.sub(
#       expr.diag,
#       x), dom.lessThan(0))
#   #
#   # # constraints
#   # for i in range(m):
#   #   if sign[i] == 0:
#   #     model.constraint(
#   #       expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
#   #       dom.equalsTo(b[i]))
#   #   elif sign[i] == -1:
#   #     model.constraint(
#   #       expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
#   #       dom.greaterThan(b[i]))
#   #   else:
#   #     model.constraint(
#   #       expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
#   #       dom.lessThan(b[i]))
#
#   # objectives
#   obj_expr = expr.add(expr.sum(expr.mulElm(Q, sum_y)), expr.dot(x, q))
#   # obj_expr = expr.add(expr.sum(expr.mulElm(Q, Y)), expr.dot(x, q))
#   model.objective(mf.ObjectiveSense.Minimize
#                   if sense == 'min' else mf.ObjectiveSense.Maximize, obj_expr)
#
#   r = MSKResult()
#   r.xvar = x
#   # r.yvar = Y
#   r.problem = model
#   if not solve:
#     return r
#
#   model.solve()
#   xval = x.level().reshape(xshape)
#   # r.yval = Y.level().reshape((n, n))
#   r.xval = xval
#   r.relax_obj = model.primalObjValue()
#   r.true_obj = qp_obj_func(Q, q, xval)
#   r.solved = True
#   r.solve_time = model.getSolverDoubleInfo("optimizerTime")
#
#   return r


####################################
# Archaic naive models
#   simply rewrite positive eigenvectors
#   into a SDP instead of conic
# todo: to be improved
####################################
def msc_diag_sdp(
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
    # constraints
    model.constraint(
      ymat.index(n_pos_eig, n_pos_eig), dom.equalsTo(1.)
    )
    model.constraint(
      expr.sub(zpos, zz),
      dom.equalsTo(0)
    )
    model.constraint(
      expr.sub(yy.diag(), ypos),
      dom.equalsTo(0)
    )
  
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
  
  Zp = model.variable("Zp", dom.inPSDCone(n + 1))
  Yp = Zp.slice([0, 0], [n, n])
  xp = Zp.slice([0, n], [n, n + 1])
  xn = model.variable("xn", [*xshape])
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
  for idx in qineg:
    model.constraint(expr.vstack(0.5, y.index([idx, 0]), z.index([idx, 0])),
                     dom.inRotatedQCone())
  arr_neg_dim = np.zeros(qineg.shape[0]).astype(int).tolist()
  true_obj_expr = expr.dot(qel[qineg, :].flatten(), y.pick(qineg.tolist(), arr_neg_dim))
  
  # positive
  n_pos_eig = qipos.shape[0]
  if n_pos_eig > 0:
    # else the problem is convex
    model.constraint(expr.mul(qneg.T, Yp), dom.equalsTo(0))
    qplus = qpos @ np.diag(qel.flatten()) @ qpos.T
    true_obj_expr = expr.add(
      true_obj_expr,
      expr.sum(expr.mulElm(qplus, Yp))
    )
    model.constraint(expr.sub(Yp.diag(), x), dom.lessThan(0))
  
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
  true_obj_expr = expr.add(true_obj_expr, expr.dot(q, x))
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
