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

from . import bg_msk, bg_cvx, bg_msk_msc, bg_msk_msc_admm, bg_msk_norm_admm
from .bg_msk_msc import mf, MSKMscResult
from .bb import BCParams, BBItem, Cuts, RLTCuttingPlane
from .classes import MscBounds, Branch, Bounds, ADMMParams
from .classes import QP, qp_obj_func, Result
from .classes import PRECISION_OBJVAL, PRECISION_SOL
from .classes import DEBUG_BB, ctr, tr

if DEBUG_BB:
  ctr.track_class(mf.Model)
  ctr.track_class(MscBounds)
  ctr.track_class(MSKMscResult)


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
      # self.simple_vio_branch(x, y, z, yc, zc, res, relax, bound)
      self.simple_bls_branch(x, y, z, yc, zc, res, relax, bound)
    elif name == MscBranch.bound:
      self.simple_bound_branch(x, y, bound, relax)
    else:
      return False
  
  def simple_vio_branch(self, x, y, z, yc, zc, res, relax=False, bound=None):
    if not relax:
      raise ValueError(f"cannot branch on z for an integral instance")
    y_index = np.unravel_index(np.argmax(res, axis=None), res.shape)
    self.ypivot = self.zpivot = y_index
    self.ypivot_val = yc[y_index]
    self.zpivot_val = zc[y_index]
    self.type = MscBranch.vio
  
  def simple_bls_branch(self, x, y, z, yc, zc, res, relax=False, bound=None):
    """
    simply create a disjunctive, by ``balancing''
      z <= (l+u)/2 or z >= (l+u)/2
    may be unable to cut off x*
    """
    if not relax:
      raise ValueError(f"cannot branch on z for an integral instance")
    y_index = np.unravel_index(np.argmax(res, axis=None), res.shape)
    self.ypivot = self.zpivot = y_index
    
    self.zpivot_val = bound.zlb[y_index] / 2 + bound.zub[y_index] / 2
    
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
      _lb, _ub = _lb_arr[_pivot], _ub_arr[_pivot]
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
    _pivot = n, m = self.ypivot
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
  def __init__(
      self, depth, node_id, parent_id, parent_bound, result,
      bound: Bounds = None, cuts: Cuts = None,
      backend_func=None, result_args=None
  ):
    super().__init__(depth, node_id, parent_id, parent_bound, result, bound, cuts)
    self.bool_model_init = False
    self.backend_func = backend_func
    self.result_args = result_args
  
  def create_model(self, qp: QP):
    self.result = self.backend_func(qp, self.bound, **self.result_args)
    self.cuts.add_cuts(self.result, self.result_args['backend_name'])
    self.bool_model_init = True


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
  n, d = branch.zpivot
  # todo do not need m
  m = 0
  u_i, l_i = bounds.zub[n, d], bounds.zlb[n, d]
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


class BCInfo(object):
  #############
  # counters
  #############
  counter_primal_improv = 0
  counter_primal_invoke = 0
  counter_total_nodes = 1
  counter_solved_nodes = 1


def generate_child_items(
    total_nodes, qp: QP, parent: MscBBItem, branch: MscBranch,
    verbose=False, backend_name='msk',
    backend_func=None, sdp_solver="MOSEK",
    with_shor: Result = None
):
  # left <=
  _ = branch.imply_all_bounds(qp, parent.bound)
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
      _item = MscBBItem(
        parent.depth + 1, _current_node,
        parent.node_id, parent.result.relax_obj,
        _r,  # result set to null
      )
      _item.bool_model_init = True
    else:
      _result_args = dict(
        solver=sdp_solver,
        verbose=verbose,
        solve=False,
        backend_name=backend_name
      )
      # add cuts to cut off
      _cuts = parent.cuts.generate_cuts(branch, _bounds, scope=_scope)
      
      _item = MscBBItem(
        parent.depth + 1,
        _current_node,
        parent.node_id,
        parent.result.relax_obj,
        None,  # result set to null
        _bounds,
        _cuts,
        backend_func, _result_args  # func, arg to create result
      )
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
  interval_logging = params.interval_logging
  
  admmparams = ADMMParams()
  
  # choose backend
  if backend_func is None:
    if backend_name == 'msk':
      backend_func = bg_msk_msc.msc_diag
    else:
      raise ValueError("not implemented")
  # choose backend
  interval_primal = params.interval_primal
  if primal_func is None and bool_use_primal:
    if primal_name == 'admm':
      primal_func = bg_msk_msc_admm.msc_admm
      admmparams.max_iteration = 20
    elif primal_name == 'nadmm':
      # more expensive
      primal_func = bg_msk_norm_admm.msc_admm
      admmparams.max_iteration = 200
      interval_primal = params.interval_primal * 10
    else:
      interval_primal = 1
  
  # problems
  k = -1
  start_time = time.time()
  
  # create a copy of QP based on decomposition method
  qp = QP(*qp_init.unpack())
  qp.decompose(decompose_method=decompose_method)
  
  # root
  root_bound = MscBounds(**bounds.__dict__, qp=qp)
  
  print("Solving root node")
  
  # solution info
  bcinfo = BCInfo()
  bcinfo.best_r = root_r = backend_func(qp, bounds=root_bound, solver=params.dual_backend, verbose=False, solve=True)
  
  # global cuts
  glc = MscCuts()
  root = MscBBItem(0, 0, -1, 1e8, result=root_r, bound=root_bound, cuts=glc)
  root.bool_model_init = True
  bcinfo.ub = ub = root_r.relax_obj
  bcinfo.lb = -1e6
  
  feasible_via_primal = {}  # solution by heuristics
  
  ub_dict = {0: ub}
  queue = PriorityQueue()
  queue.put((-ub, root))
  
  def _process_item(item) -> int:
    """
    :rtype: int
    """
    
    parent_sdp_val = item.parent_bound
    ub = max(ub_dict.values())
    ub_dict.pop(item.node_id)
    
    r_primal = feasible_via_primal.pop(item.node_id) if item.node_id in feasible_via_primal else None
    if parent_sdp_val < bcinfo.lb:
      # prune this tree
      print(f"prune #{item.node_id} since parent pruned")
      return -1
    
    if not item.bool_model_init:
      item.create_model(qp)
    
    r = item.result
    
    if not r.solved:
      r.solve(verbose=verbose, qp=qp)
      r.solve_time = time.time() - start_time
      bcinfo.counter_solved_nodes += 1
    
    if r.relax_obj < bcinfo.lb:
      # prune this tree
      if k % interval_logging == 0:
        print(f"prune #{item.node_id} @{r.relax_obj :.4f} by bound")
      return -1
    
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
    bool_start_primal = k % interval_primal == 0
    if bool_has_primal and k > 0:
      if bool_start_primal:
        # check if parent primal solution is still feasible,
        # if so then the solution is valid
        #  there is no need for this.
        _zval = r_primal.zval
        if np.all(_zval <= item.bound.zub) and np.all(_zval >= item.bound.zub):
          bool_pass_primal = True
      else:
        bool_pass_primal = True
    
    if bool_start_primal and bool_has_primal and not bool_pass_primal:
      try:
        r_primal = primal_func(qp, item.bound, True, admmparams, r)
        bcinfo.counter_primal_invoke += 1
        if not bool_sol_feasible or r_primal.true_obj > r.true_obj:
          r.xval = r_primal.xval
          r.yval = r_primal.yval
          r.zval = r_primal.zval
          print(f"primal {primal_func.__name__} is better: {r_primal.true_obj, r.true_obj}")
          r.true_obj = r_primal.true_obj
          bcinfo.counter_primal_improv += 1
      except Exception as e:
        print(f"primal {primal_func.__name__} failed ")
        print(e.__traceback__.format_exec())
    
    # if bool_pass_primal:
    #   # do not need to start primal
    #   # still using parent's primal
    #   print("primal passed")
    
    if r.true_obj > bcinfo.lb:
      bcinfo.best_r = r
      bcinfo.lb = r.true_obj
    
    gap = (ub - bcinfo.lb) / (abs(ub) + 1e-3)
    gap_primal = abs(r.true_obj - r.relax_obj) / (abs(r.relax_obj) + 1e-3)
    
    if k % interval_logging == 0:
      print(
        f"time: {r.solve_time:.2e} #{item.node_id:.2e}, "
        f"depth: {item.depth:.2e}, "
        f"feas: {r.resc_feas:.3e} "
        f"feasC: {r.resc_feasC:.3e} "
        f"obj: {r.true_obj:.3e}, "
        f"sdp_obj: {r.relax_obj:.3e}, "
        f"gap_*: {gap_primal:.2%} ",
        f"gap:{gap:.2%} ([{bcinfo.lb:.2e},{r.relax_obj:.2e},{ub:.2e}]")
    
    if gap <= params.opt_eps or r.solve_time >= params.time_limit:
      print(f"terminate #{item.node_id} by gap or time_limit")
      return 1
    
    if bool_feasible or gap_primal < params.opt_eps:
      print(f"prune #{item.node_id} by feasible solution")
      # feasible[item.node_id] = r
      return -1
    
    _ = generate_child_items(
      bcinfo.counter_total_nodes, qp, item, br,
      sdp_solver=params.dual_backend,
      verbose=verbose,
      backend_name=backend_name,
      backend_func=backend_func,
    )
    for next_item in _:
      bcinfo.counter_total_nodes += 1
      next_priority = - r.relax_obj.round(PRECISION_OBJVAL)
      queue.put((next_priority, next_item))
      ub_dict[next_item.node_id] = r.relax_obj
      feasible_via_primal[next_item.node_id] = r_primal
    return 0
  
  while not queue.empty():
    
    k += 1
    
    priority, item = queue.get()
    info = _process_item(item)
    queue.task_done()
    if info == 1:
      break
  
  bcinfo.best_r.nodes = bcinfo.counter_solved_nodes
  max_left = max(*ub_dict.values()) if ub_dict.__len__() > 0 else 1e6
  bcinfo.best_r.bound = min(bcinfo.best_r.relax_obj, max_left)
  bcinfo.best_r.solve_time = time.time() - start_time
  
  print(f"//finished with {bcinfo.best_r.solve_time: .3f} s \n"
        f"//primal improvement {bcinfo.counter_primal_improv} \n"
        f"//primal invocation {bcinfo.counter_primal_invoke} \n"
        f"//total nodes: {bcinfo.counter_total_nodes} \n"
        f"//search nodes: {bcinfo.counter_solved_nodes} \n"
        f"//unexpr nodes: {len(queue.queue)}")
  
  if DEBUG_BB:
    ctr.create_snapshot()
    ctr.stats.print_summary()
  
  with queue.mutex:
    queue.unfinished_tasks = 0
    queue.all_tasks_done.notify_all()
    queue.queue.clear()
  
  feasible_via_primal.clear()
  
  return bcinfo.best_r
