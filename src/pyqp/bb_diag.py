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
from . import bg_msk, bg_cvx, bg_msk_msc, bg_msk_msc_admm, bg_msk_norm_admm
from .bb import BCParams, BBItem, Cuts, RLTCuttingPlane
from .classes import MscBounds, Branch, Bounds, ADMMParams
from .classes import QP, qp_obj_func, Result
from .classes import PRECISION_OBJVAL, PRECISION_SOL


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
    self.xpivot_val = x[x_index].round(self.PRECISION)
    self.type = MscBranch.bound
    self.var_type = not relax
  
  def _imply_bounds(self, pivot, pivot_val, bounds: MscBounds, left=True, target="zvar"):
    _succeed = False
    _pivot = (m, n) = pivot
    _lval = _rval = _val = pivot_val
    if self.var_type:
      _lval = np.floor(_val)
      _rval = np.ceil(_val)
    newbl = MscBounds(**bounds.__dict__)
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
    yi = yvar.index(n, 0)
    zi = zvar.index(n, 0)
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
    z, y = r.zvar, r.yvar
    
    for cut_type, cut_list in self.cuts.items():
      for ct in cut_list:
        for expr in ct.serialize_to_cvx(z, y):
          _problem._constraints.append(expr)
  
  def add_cuts_to_msk(self, r: bg_cvx.CVXMscResult):
    
    _problem: bg_msk.mf.Model = r.problem
    z, y = r.zvar, r.yvar
    
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
    bounds: Bounds,
    verbose=False,
    params=BCParams(),
    decompose_method="eig-type2",
    **kwargs
):
  print(json.dumps(params.__dict__(), indent=2))
  branch_name = kwargs.get('branch_name', MscBranch.vio)
  backend_func = kwargs.get('func')
  backend_name = params.dual_backend
  primal_func = kwargs.get('primal_func')
  primal_name = params.primal_backend
  bool_use_primal = kwargs.get("use_primal", True)
  logging_interval = params.logging_interval
  
  admmparams = ADMMParams()
  
  # choose backend
  if backend_func is None:
    if backend_name == 'msk':
      backend_func = bg_msk_msc.msc_diag
    elif backend_name == 'cvx':
      backend_func = bg_cvx.msc_relaxation
    else:
      raise ValueError("not implemented")
  # choose backend
  primal_interval = params.primal_interval
  if primal_func is None and bool_use_primal:
    if primal_name == 'admm':
      primal_func = bg_msk_msc_admm.msc_admm
      admmparams.max_iteration = 20
    elif primal_name == 'nadmm':
      # more expensive
      primal_func = bg_msk_norm_admm.msc_admm
      admmparams.max_iteration = 200
      primal_interval = params.primal_interval * 10
    else:
      primal_interval = 1
  
  # problems
  k = 0
  start_time = time.time()
  
  # create a copy of QP based on decomposition method
  qp = QP(*qp_init.unpack())
  qp.decompose(decompose_method=decompose_method)
  
  # root
  root_bound = MscBounds(**bounds.__dict__, qp=qp)
  
  print("Solving root node")
  root_r = backend_func(qp, bounds=root_bound, solver=params.dual_backend, verbose=False, solve=True)
  
  best_r = root_r
  
  # global cuts
  glc = MscCuts()
  root = MscBBItem(qp, 0, 0, -1, 1e8, result=root_r, bound=root_bound, cuts=glc)
  ub = root_r.relax_obj
  lb = -1e6
  
  #############
  # counters
  #############
  counter_primal_improv = 0
  counter_primal_invoke = 0
  counter_total_nodes = 1
  counter_solved_nodes = 1
  
  ub_dict = {0: ub}
  queue = PriorityQueue()
  queue.put((-ub, root))
  feasible = {}
  feasible_via_primal = {}  # solution by heuristics
  feasible_node_id = {}
  
  while not queue.empty():
    
    priority, item = queue.get()
    
    r = item.result
    
    parent_sdp_val = item.parent_bound
    
    ub = max(ub_dict.values())
    ub_dict.pop(item.node_id)
    
    if parent_sdp_val < lb:
      # prune this tree
      print(f"prune #{item.node_id} since parent pruned")
      continue
    
    if not r.solved:
      r.solve(verbose=verbose, qp=qp)
      r.solve_time = time.time() - start_time
      counter_solved_nodes += 1
    
    if r.relax_obj < lb:
      # prune this tree
      if k % logging_interval == 0:
        print(f"prune #{item.node_id} @{r.relax_obj :.4f} by bound")
      continue
    
    # it is essentially a lower bound by real objective
    #   if and only if it is feasible
    bool_integral_feasible = True if params.relax else (r.xval.round() - r.xval).max() <= params.feas_eps
    bool_sol_feasible = r.resc_feasC <= params.feas_eps and bool_integral_feasible
    bool_feasible = r.resc_feas <= params.feas_eps and bool_integral_feasible
    
    ##########################
    # branching
    ##########################
    x = r.xval
    zc = z = r.zval
    yc = y = r.yval
    br = MscBranch()
    branch_args = (
      x, y, z, yc, zc, r.resc, item.bound
    )
    br.branch(*branch_args, relax=params.relax, name=branch_name)
    
    ##########################
    # primal solution condition
    ##########################
    bool_pass_primal = False
    bool_has_primal = primal_func is not None
    bool_start_primal = k % primal_interval == 0
    if bool_has_primal and k > 0:
      if bool_start_primal:
        _parent_primal_id = feasible_node_id[item.parent_id]
        _parent_primal_r = feasible_via_primal[_parent_primal_id]
        # check if still feasible, if so then the solution is valid
        #  there is no need for this.
        _zval = _parent_primal_r.zval
        if np.all(_zval <= item.bound.zub) and np.all(_zval >= item.bound.zub):
          bool_pass_primal = True
      else:
        bool_pass_primal = True
    
    if bool_start_primal and bool_has_primal:
      try:
        r_primal = primal_func(qp, item.bound, True, admmparams, r)
        feasible_via_primal[item.node_id] = r_primal
        feasible_node_id[item.node_id] = item.node_id
        counter_primal_invoke += 1
        if not bool_sol_feasible or r_primal.true_obj > r.true_obj:
          r.xval = r_primal.xval
          r.yval = r_primal.yval
          r.zval = r_primal.zval
          print(f"primal {primal_func.__name__} is better: {r_primal.true_obj, r.true_obj}")
          r.true_obj = r_primal.true_obj
          counter_primal_improv += 1
      except Exception as e:
        print(f"primal {primal_func.__name__} failed ")
    
    if bool_pass_primal:
      # do not need to start primal
      # still using parent's primal
      print("primal passed")
      feasible_node_id[item.node_id] = feasible_node_id[item.parent_id]
    
    if r.true_obj > lb:
      best_r = r
      lb = r.true_obj
    
    gap = (ub - lb) / (abs(ub) + 1e-3)
    gap_primal = abs(r.true_obj - r.relax_obj) / (abs(r.relax_obj) + 1e-3)
    
    if k % logging_interval == 0:
      print(
        f"time: {r.solve_time:.2e} #{item.node_id:.2e}, "
        f"depth: {item.depth:.2e}, "
        f"feas: {r.resc_feas:.3e} "
        f"feasC: {r.resc_feasC:.3e} "
        f"obj: {r.true_obj:.3e}, "
        f"sdp_obj: {r.relax_obj:.3e}, "
        f"gap_*: {gap_primal:.2%} ",
        f"gap:{gap:.2%} ([{lb:.2e},{r.relax_obj:.2e},{ub:.2e}]")
    
    if gap <= params.opt_eps or r.solve_time >= params.time_limit:
      print(f"terminate #{item.node_id} by gap or time_limit")
      break
    
    if bool_feasible or gap_primal < params.opt_eps:
      print(f"prune #{item.node_id} by feasible solution")
      feasible[item.node_id] = r
      continue
    
    _ = generate_child_items(
      counter_total_nodes, item, br,
      sdp_solver=params.dual_backend,
      verbose=verbose,
      backend_name=backend_name,
      backend_func=backend_func,
    )
    for next_item in _:
      counter_total_nodes += 1
      next_priority = - r.relax_obj.round(PRECISION_OBJVAL)
      queue.put((next_priority, next_item))
      ub_dict[next_item.node_id] = r.relax_obj
    #
    k += 1
  
  best_r.nodes = counter_solved_nodes
  best_r.relax_obj = min([best_r.relax_obj, *ub_dict.values()])
  best_r.solve_time = time.time() - start_time
  print(f"//finished with {best_r.solve_time: .3f} s \n"
        f"//primal improvement {counter_primal_improv} \n"
        f"//primal invocation {counter_primal_invoke}")
  return best_r
