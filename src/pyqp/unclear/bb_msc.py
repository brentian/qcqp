"""
This file the branch-and-cut for MSC formulation
- use MSC in the backend
- trying to branch on y = z • z
- then branch for z ≥ √l ∨ z ≤ −√l

"""
import json
from queue import PriorityQueue

import numpy as np
import time

import pyqp.bg_msk_msc
from . import bg_msk, bg_cvx
from .bb import BCParams, BBItem, Cuts, RLTCuttingPlane
from .classes import MscBounds, Branch
from .classes import QP, qp_obj_func, Result


class MscBranch(Branch):
  def __init__(self):
    
    self.ypivot = None
    self.ypivot_val = None
    self.zpivot = None
    self.zpivot_val = None
  
  def simple_vio_branch(self, y, z, yc, zc, res):
    y_index = np.unravel_index(np.argmax(res, axis=None), res.shape)
    self.ypivot = self.zpivot = y_index
    self.ypivot_val = yc[y_index]
    self.zpivot_val = zc[y_index]
  
  def imply_bounds(self, bounds: MscBounds, left=True):
    _succeed = False
    n, m = self.zpivot
    _pivot = (m, n)
    _val = self.zpivot_val.round(6)
    newbl = MscBounds(*bounds.unpack())
    _lb, _ub = newbl.zlb[(*_pivot, 0)], newbl.zub[(*_pivot, 0)]
    if left and _val < _ub:
      # <= and a valid upper bound
      newbl.zub[(*_pivot, 0)] = _val
      _succeed = True
    if not left and _val > _lb:
      newbl.zlb[(*_pivot, 0)] = _val
      # newbl.ylb = newbl.xlb @ newbl.xlb.T
      _succeed = True
    
    # after update, check bound feasibility:
    if newbl.zlb[(*_pivot, 0)] > newbl.zub[(*_pivot, 0)]:
      _succeed = False
    return _succeed, newbl
  
  def imply_all_bounds(self, qp: QP, bounds: MscBounds):
    """
    :param bounds:
    :return:
    """
    n, m = self.ypivot
    _pivot = (m, n)
    # pivot values
    _val = self.ypivot_val
    _lb, _ub = bounds.ylb[(*_pivot, 0)], bounds.yub[(*_pivot, 0)]
    _zval = self.zpivot_val
    _zlb, _zub = bounds.zlb[(*_pivot, 0)], bounds.zub[(*_pivot, 0)]
    
    for _zzleft in (True, False):
      _succeed, newbl = self.imply_bounds(bounds, left=_zzleft)
      newbl.imply_y(qp)
      yield _succeed, newbl
  
  def imply_all_bounds2(self, qp: QP, bounds: MscBounds):
    """
    y ≤ y' (l=l') ∨ y ≥ y' (l=y')
    z ≥ √l ∨ z ≤ −√l
    :param bounds:
    :return:
    """
    n, m = self.ypivot
    _pivot = (m, n)
    # pivot values
    _val = self.ypivot_val
    _lb, _ub = bounds.ylb[(*_pivot, 0)], bounds.yub[(*_pivot, 0)]
    _zval = self.zpivot_val
    _zlb, _zub = bounds.zlb[(*_pivot, 0)], bounds.zub[(*_pivot, 0)]
    
    for _left in (True, False):
      zlb, zub, ylb, yub, *_ = bounds.unpack()
      _succeed = False
      if _left and _val < _ub:
        yub[(*_pivot, 0)] = _val
      if not _left and _val > _lb:
        ylb[(*_pivot, 0)] = _val
      if ylb[(*_pivot, 0)] > yub[(*_pivot, 0)]:
        yield False, None
      # cal √l
      for _zleft in (True, False):
        newbl = MscBounds(zlb, zub, ylb, yub, *_)
        newl = np.sqrt(newbl.ylb[(*_pivot, 0)]).round(6)
        if _zleft and - newl < _zub:
          # <= and a valid upper bound
          newbl.zub[(*_pivot, 0)] = - newl
          _succeed = True
        if not _zleft and newl > _zlb:
          newbl.zlb[(*_pivot, 0)] = newl
          _succeed = True
        
        if _succeed:
          if newbl.zlb[(*_pivot, 0)] > newbl.zub[(*_pivot, 0)]:
            _succeed = False
          newbl.imply_y(qp)
          yield _succeed, newbl
        else:
          # this means it is unchanged,
          # at least for z, the cutting planes will not work
          # this is the case for ( left x left ):
          #   y ≤ y' (l=l') z ≥ √l
          # since next time it will stops at the same solution
          for _zzleft in (True, False):
            self.imply_bounds(newbl, left=_zzleft)


class MscBBItem(BBItem):
  pass


class MscRLT(RLTCuttingPlane):
  def serialize_to_cvx(self, zvar, yvar):
    n, i, ui, li = self.data
    # (xi - li)(xi - ui) <= 0
    expr1 = yvar[n][i, 0] - zvar[n][i, 0] * ui - li * zvar[n][i, 0] + ui * li <= 0
    yield expr1
  
  def serialize_to_msk(self, zvar, yvar):
    expr = bg_msk.expr
    exprs = expr.sub
    exprm = expr.mul
    
    m, n, ui, li = self.data
    # n = yvar.getShape()[0]
    yi = yvar[m].index(n, 0)
    zi = zvar[m].index(n, 0)
    # (xi - li)(xi - ui) <= 0
    expr1, dom1 = exprs(exprs(yi, exprm(ui, zi)),
                        exprm(li, zi)), bg_msk.dom.lessThan(- ui * li)
    yield expr1, dom1


def add_rlt_cuts(branch, bounds):
  n, m = branch.zpivot
  u_i, l_i = bounds.zub[m, n, 0], bounds.zlb[m, n, 0]
  return MscRLT((m, n, u_i, l_i))


cutting_method = {
  'rlt': add_rlt_cuts
}


class MscCuts(Cuts):
  def __init__(self):
    self.cuts = {}
  
  def generate_cuts(self, branch: Branch, bounds: MscBounds, scope=None):
    
    # cuts
    if scope is None:
      scope = cutting_method
    new_cuts = MscCuts()
    for k, v in scope.items():
      val = v(branch, bounds)
      new_cuts.cuts[k] = self.cuts.get(k, []) + [val]
    
    return new_cuts
  
  def add_cuts_to_cvx(self, r: bg_cvx.CVXMscResult):
    
    _problem = r.problem
    z, y = r.Zvar, r.Yvar
    
    for cut_type, cut_list in self.cuts.items():
      for ct in cut_list:
        for expr in ct.serialize_to_cvx(z, y):
          _problem._constraints.append(expr)
  
  def add_cuts_to_msk(self, r: bg_cvx.CVXMscResult):
    
    _problem: bg_msk.mf.Model = r.problem
    z, y = r.Zvar, r.Yvar
    
    for cut_type, cut_list in self.cuts.items():
      for ct in cut_list:
        for expr, dom in ct.serialize_to_msk(z, y):
          _problem.constraint(expr, dom)


def generate_child_items(
    total_nodes, parent: MscBBItem, branch: MscBranch,
    verbose=False, backend_name='msk',
    backend_func=None, sdp_solver="MOSEK",
    with_shor: Result = None
):
  # left <=
  _ = branch.imply_all_bounds(parent.qp, parent.bound)
  _current_node = total_nodes
  for _succ, _bounds in _:
    # n, m = branch.ypivot
    # _pivot = (m, n)
    # print(_succ, _bounds.zlb[(*_pivot, 0)], _bounds.zub[(*_pivot, 0)], _bounds.ylb[(*_pivot, 0)],
    #       _bounds.yub[(*_pivot, 0)])
    if not _succ:
      # problem is infeasible:
      _r = Result()
      _r.solved = True
      _r.relax_obj = -1e6
      _cuts = MscCuts()
    else:
      _r = backend_func(parent.qp, bounds=_bounds, solver=sdp_solver, verbose=verbose, solve=False,
                        with_shor=with_shor, constr_d=False, rlt=True)
      # add cuts to cut off
      _cuts = parent.cuts.generate_cuts(branch, _bounds)
      _cuts.add_cuts(_r, backend_name)
    
    _item = MscBBItem(parent.qp, parent.depth + 1, _current_node,
                      parent.node_id, parent.result.relax_obj,
                      _r, _bounds, _cuts)
    _current_node += 1
    yield _item


def bb_box(
    qp_init: QP,
    verbose=False,
    params=BCParams(),
    bool_use_shor=False,
    constr_d=False,
    rlt=True,
    decompose_method="eig-type1",
    **kwargs
):
  print(json.dumps(params.__dict__(), indent=2))
  backend_func = kwargs.get('func')
  backend_name = params.dual_backend
  if backend_func is None:
    if backend_name == 'msk':
      backend_func = pyqp.bg_msk_msc.msc
    else:
      raise ValueError("not implemented")
  # choose branching
  
  # problems
  k = 0
  start_time = time.time()
  
  # create a copy of QP based on decomposition method
  qp = QP(*qp_init.unpack())
  qp.decompose(decompose_method=decompose_method)
  
  # root
  root_bound = MscBounds.construct(qp, imply_y=True)
  
  print("Solving root node")
  root_r = backend_func(qp, bounds=root_bound, solver=params.dual_backend, verbose=True, solve=True,
                        constr_d=constr_d, rlt=rlt)
  
  best_r = root_r
  
  # global cuts
  glc = MscCuts()
  root = MscBBItem(qp, 0, 0, -1, 1e8, result=root_r, bound=root_bound, cuts=glc)
  total_nodes = 1
  ub = root_r.relax_obj
  lb = -1e6
  
  ub_dict = {0: ub}
  queue = PriorityQueue()
  queue.put((-ub, root))
  feasible = {}
  
  while not queue.empty():
    priority, item = queue.get()
    del ub_dict[item.node_id]
    r = item.result
    
    parent_sdp_val = item.parent_bound
    
    if parent_sdp_val < lb:
      # prune this tree
      print(f"prune #{item.node_id} since parent pruned")
      continue
    
    if not r.solved:
      r.solve(verbose=verbose)
      r.solve_time = time.time() - start_time
    
    if r.relax_obj < lb:
      # prune this tree
      print(f"prune #{item.node_id} @{r.relax_obj :.4f} by bound")
      continue
    
    x = r.xval
    z = r.zval
    y = r.yval
    zc = r.Zval
    yc = r.Yval
    resc = np.abs(yc - zc ** 2)
    resc_feas = resc.max()
    resc_feasC = resc[:, 1:].max() if resc.shape[1] > 1 else 0
    
    # it is for real a lower bound by real objective
    #   if and only if it is feasible
    r.true_obj = 0 if resc_feasC > params.feas_eps \
      else qp_obj_func(item.qp.Q, item.qp.q, r.xval)
    
    r.bound = r.relax_obj
    if r.true_obj > lb:
      best_r = r
      lb = r.true_obj
    
    ub = max([r.relax_obj, max(ub_dict.values()) if len(ub_dict) > 0 else 0])
    gap = (ub - lb) / (abs(lb) + 1e-3)
    
    print(
      f"time: {r.solve_time: .2f} #{item.node_id}, "
      f"depth: {item.depth}, "
      f"feas: {resc_feas:.3e} "
      f"feasC: {resc_feasC:.3e}"
      f"obj: {r.true_obj:.4f}, "
      f"sdp_obj: {r.relax_obj:.4f}, gap:{gap:.4%} ([{lb: .2f},{ub: .2f}]")
    
    if gap <= params.opt_eps or r.solve_time >= params.time_limit:
      print(f"terminate #{item.node_id} by gap or time_limit")
      break
    
    if resc_feas <= params.feas_eps:
      print(f"prune #{item.node_id} by feasible solution")
      feasible[item.node_id] = r
      continue
    
    ## branching
    br = MscBranch()
    br.simple_vio_branch(y, z, yc, zc, resc)
    _ = generate_child_items(
      total_nodes, item, br,
      sdp_solver=params.sdp_solver,
      verbose=verbose,
      backend_name=backend_name,
      backend_func=backend_func,
      with_shor=r_shor,
    )
    for next_item in _:
      total_nodes += 1
      next_priority = - r.relax_obj.round(PRECISION_OBJVAL)
      queue.put((next_priority, next_item))
      ub_dict[next_item.node_id] = r.relax_obj
    #
    k += 1
  
  best_r.nodes = total_nodes
  best_r.bound = ub
  best_r.relax_obj = ub
  best_r.solve_time = time.time() - start_time
  return best_r
