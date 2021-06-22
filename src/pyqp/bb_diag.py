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

from . import bg_msk, bg_cvx
from .bb import BCParams, BBItem, Cuts, RLTCuttingPlane
from .classes import MscBounds, Branch
from .classes import QP, qp_obj_func, Result


class MscBranch(Branch):
  vio = 'vio'
  bound = 'bound'
  
  def __init__(self):
    self.xpivot = None
    self.xpivot_val = None
    self.xminor = None
    self.xminor_val = None
    self.ypivot = None
    self.ypivot_val = None
    self.zpivot = None
    self.zpivot_val = None
    self.type = None
    self.var_type = 0
  
  def branch(self, *args, relax=False, name=vio):
    
    x, y, z, yc, zc, res, bound = args
    if name == MscBranch.vio:
      self.simple_vio_branch(x, y, z, yc, zc, res, relax)
    elif name == MscBranch.bound:
      self.simple_bound_branch(x, y, bound, relax)
    else:
      return False
  
  def simple_vio_branch(self, x, y, z, yc, zc, res, relax=False):
    if not relax:
      raise ValueError(f"cannot branch on z for an integral instance")
    y_index = np.unravel_index(np.argmax(res, axis=None), res.shape)
    self.ypivot = self.zpivot = y_index
    self.ypivot_val = yc[y_index]
    self.zpivot_val = zc[y_index]
    self.type = MscBranch.vio
  
  def simple_bound_branch(self, x, y, bound, relax=True):
    res = np.min([x - bound.xlb, bound.xub - x], axis=0)
    x_index = np.unravel_index(np.argmax(res, axis=None), res.shape)
    # x_index = res_sum.argmax()
    self.xpivot = x_index
    self.xpivot_val = x[x_index].round(6)
    self.type = MscBranch.bound
    self.var_type = not relax
  
  def _imply_bounds(self, pivot, pivot_val, bounds: MscBounds, left=True, target="zvar"):
    _succeed = False
    _pivot = (m, n) = pivot
    _lval = _rval = _val = pivot_val
    if self.var_type:
      _lval = np.floor(_val)
      _rval = np.ceil(_val)
    newbl = MscBounds(*bounds.unpack())
    if target == 'zvar':
      _lb_arr, _ub_arr = newbl.zlb, newbl.zub
      _lb, _ub = _lb_arr[(*_pivot, 0)], _ub_arr[(*_pivot, 0)]
    elif target == 'xvar':
      _lb_arr, _ub_arr = newbl.xlb, newbl.xub
      _lb, _ub = _lb_arr[_pivot], _ub_arr[_pivot]
    else:
      raise ValueError(f"target {target} not implemented")
    if left and _lval < _ub:
      # <= valid upper bound
      _ub_arr[_pivot] = _lval
      _succeed = True
    if not left and _rval > _lb:
      # >= valid lower bound
      _lb_arr[_pivot] = _rval
      _succeed = True
    # after update, check bound feasibility:
    if _lb_arr[_pivot] > _ub_arr[_pivot]:
      _succeed = False
    return _succeed, newbl
  
  def _imply_all_bounds_yz(self, qp: QP, bounds: MscBounds):
    """
    :param bounds:
    :return:
    """
    n, m = self.ypivot
    _pivot = (m, n)
    _val = self.ypivot_val
    _zval = self.zpivot_val
    
    for _zzleft in (True, False):
      _succeed, newbl = self._imply_bounds(_pivot, _zval, bounds, left=_zzleft, target='zvar')
      newbl.imply_y(qp)
      yield _succeed, newbl
  
  def _imply_all_bounds_x(self, qp: QP, bounds: MscBounds):
    """
    :param bounds:
    :return:
    """
    n, d = self.xpivot
    _pivot = (n, d)
    _val = self.xpivot_val
    
    for _zzleft in (True, False):
      _succeed, newbl = self._imply_bounds(_pivot, _val, bounds, left=_zzleft, target='xvar')
      yield _succeed, newbl
  
  def imply_all_bounds(self, qp: QP, bounds: MscBounds):
    if self.type == MscBranch.vio:
      return self._imply_all_bounds_yz(qp, bounds)
    elif self.type == MscBranch.bound:
      return self._imply_all_bounds_x(qp, bounds)
    else:
      pass


class MscBBItem(BBItem):
  pass


class MscRLT(RLTCuttingPlane):
  
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


class MscRLTForX(RLTCuttingPlane):
  
  def serialize_to_msk(self, yvar, xvar):
    pass


def add_rlt_cuts(branch, bounds):
  n, m = branch.zpivot
  u_i, l_i = bounds.zub[m, n, 0], bounds.zlb[m, n, 0]
  return MscRLT((m, n, u_i, l_i))


cutting_method = {
  "vio": {
    'rlt': add_rlt_cuts
  }
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
  if verbose:
    print(branch.__dict__)
  _current_node = total_nodes
  _scope = cutting_method.get(branch.type, {})
  for _succ, _bounds in _:
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
      _cuts = parent.cuts.generate_cuts(branch, _bounds, scope=_scope)
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
  branch_name = kwargs.get('branch_name', MscBranch.vio)
  backend_name = params.backend_name
  logging_interval = params.logging_interval
  # choose backend
  if backend_func is None:
    if backend_name == 'msk':
      backend_func = bg_msk.msc_diag
    elif backend_name == 'cvx':
      backend_func = bg_cvx.msc_relaxation
    else:
      raise ValueError("not implemented")
  
  # problems
  k = 0
  start_time = time.time()
  
  # create a copy of QP based on decomposition method
  qp = QP(*qp_init.unpack())
  qp.decompose(decompose_method=decompose_method)
  
  # root
  root_bound = MscBounds.construct(qp, imply_y=True)
  
  if bool_use_shor:
    print("Solving the Shor relaxation")
    r_shor = bg_msk.shor(qp, solver='MOSEK', verbose=False)
  else:
    r_shor = None
  
  print("Solving root node")
  root_r = backend_func(qp, bounds=root_bound, solver=params.sdp_solver, verbose=True, solve=True, with_shor=r_shor,
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
    try:
      priority, item = queue.get()
      del ub_dict[item.node_id]
      r = item.result
      
      parent_sdp_val = item.parent_bound
      
      # if parent_sdp_val < lb:
      #   # prune this tree
      #   print(f"prune #{item.node_id} since parent pruned")
      #   continue
      
      if not r.solved:
        r.solve(verbose=verbose)
        r.solve_time = time.time() - start_time
      
      if r.relax_obj < lb:
        # prune this tree
        if k % logging_interval == 0:
          print(f"prune #{item.node_id} @{r.relax_obj :.4f} by bound")
        continue
      
      x = r.xval
      z = r.zval
      y = r.yval
      zc = r.Zval
      yc = r.Yval
      try:
        resc = np.abs(yc - zc ** 2)
      except:
        print(1)
      resc_feas = resc.max()
      resc_feasC = resc[:, 1:].max() if resc.shape[1] > 1 else 0
      
      # it is for real a lower bound by real objective
      #   if and only if it is feasible
      bool_integral_feasible = True if params.relax else (r.xval.round() - r.xval).max() <= params.feas_eps
      bool_sol_feasible = resc_feasC <= params.feas_eps and bool_integral_feasible
      bool_feasible = resc_feas <= params.feas_eps and bool_integral_feasible
      r.true_obj = qp_obj_func(item.qp.Q, item.qp.q, r.xval) if bool_sol_feasible else 0
      
      r.bound = r.relax_obj
      if r.true_obj > lb:
        best_r = r
        lb = r.true_obj
      
      ub = max([r.relax_obj, max(ub_dict.values()) if len(ub_dict) > 0 else 0])
      gap = (ub - lb) / (abs(lb) + 1e-3)
      
      if k % logging_interval == 0:
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
      
      if bool_feasible:
        print(f"prune #{item.node_id} by feasible solution")
        feasible[item.node_id] = r
        continue
      
      ## branching
      br = MscBranch()
      branch_args = (
        x, y, z, yc, zc, resc, item.bound
      )
      br.branch(*branch_args, relax=params.relax, name=branch_name)
      
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
        next_priority = - r.relax_obj.round(3)
        queue.put((next_priority, next_item))
        ub_dict[next_item.node_id] = r.relax_obj
      #
      k += 1
    except KeyboardInterrupt as e:
      print("optimization terminated")
      break
  
  best_r.nodes = total_nodes
  best_r.bound = min(ub, best_r.bound)
  best_r.relax_obj = best_r.relax_obj
  best_r.solve_time = time.time() - start_time
  return best_r
