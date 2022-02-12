"""
This file the branch-and-cut for Shor-like formulations
- use Shor in the backend
- trying to branch on x
"""
import json
from dataclasses import dataclass
from queue import PriorityQueue

import numpy as np
import time

from . import bg_msk, bg_cvx
from .classes import qp_obj_func, QP, BCParams, Result, Bounds, Branch, CuttingPlane
from .classes import PRECISION_OBJVAL, PRECISION_SOL


class SDPSmallCone(CuttingPlane):
  def __init__(self, data):
    self.data = data
  
  def serialize_to_msk(self, *args, **kwargs):
    pass
  
  @staticmethod
  def apply(branch, bounds):
    # todo, find a way to implement this
    # or to show it is not valid.
    raise ValueError("not finished!")
    i = branch.xpivot
    j = branch.xminor
    u_i, l_i = bounds.xub[i, 0], bounds.xlb[i, 0]
    u_j, l_j = bounds.xub[j, 0], bounds.xlb[j, 0]
    return SDPSmallCone((i, j, u_i, l_i, u_j, l_j))


class RLTCuttingPlane(CuttingPlane):
  def __init__(self, data):
    super().__init__(data)
  
  def serialize_to_cvx(self, xvar, yvar):
    i, j, u_i, l_i, u_j, l_j = self.data
    # (xi - li)(xj - uj) <= 0
    expr1 = yvar[i, j] - xvar[i, 0] * u_j - l_i * xvar[j, 0] + u_j * l_i <= 0
    # (xi - ui)(xj - lj) <= 0
    expr2 = yvar[i, j] - xvar[i, 0] * l_j - u_i * xvar[j, 0] + u_i * l_j <= 0
    # (xi - li)(xj - lj) >= 0
    expr3 = yvar[i, j] - xvar[i, 0] * l_j - l_i * xvar[j, 0] + l_j * l_i >= 0
    # (xi - ui)(xj - uj) >= 0
    expr4 = yvar[i, j] - xvar[i, 0] * u_j - u_i * xvar[j, 0] + u_i * u_j >= 0
    yield expr1
    yield expr2
    yield expr3
    yield expr4
  
  def serialize_to_msk(self, xvar, yvar, zvar):
    expr = bg_msk.expr
    exprs = expr.sub
    exprm = expr.mul
    n = yvar.getShape()[0]
    i, j, u_i, l_i, u_j, l_j = self.data
    # yij = zvar.index(i, j)
    # xi, xj = zvar.index(n, i), zvar.index(n, j)
    yij = yvar.index(i, j)
    xi, xj = xvar.index(i, 0), xvar.index(j, 0)
    # (xi - li)(xj - uj) <= 0
    expr1, dom1 = exprs(exprs(yij, exprm(u_j, xi)),
                        exprm(l_i, xj)), bg_msk.dom.lessThan(- u_j * l_i)
    # (xi - ui)(xj - lj) <= 0
    expr2, dom2 = exprs(exprs(yij, exprm(l_j, xi)),
                        exprm(u_i, xj)), bg_msk.dom.lessThan(- u_i * l_j)
    # (xi - li)(xj - lj) >= 0
    expr3, dom3 = exprs(exprs(yij, exprm(l_j, xi)),
                        exprm(l_i, xj)), bg_msk.dom.greaterThan(- l_j * l_i)
    # (xi - ui)(xj - uj) >= 0
    expr4, dom4 = exprs(exprs(yij, exprm(u_j, xi)),
                        exprm(u_i, xj)), bg_msk.dom.greaterThan(-u_i * u_j)
    yield expr1, dom1
    yield expr2, dom2
    # yield expr3, dom3
    # yield expr4, dom4


def add_rlt_cuts(branch, bounds):
  i = branch.xpivot
  j = branch.xminor
  u_i, l_i = bounds.xub[i, 0], bounds.xlb[i, 0]
  u_j, l_j = bounds.xub[j, 0], bounds.xlb[j, 0]
  return RLTCuttingPlane((i, j, u_i, l_i, u_j, l_j))


cutting_method = {
  'rlt': add_rlt_cuts
}


class Cuts(object):
  
  def __init__(self):
    self.cuts = {}
  
  def generate_cuts(self, branch: Branch, bounds: Bounds, scope=None):
    
    # cuts
    if scope is None:
      scope = cutting_method
    new_cuts = Cuts()
    for k, v in scope.items():
      val = v(branch, bounds)
      new_cuts.cuts[k] = self.cuts.get(k, []) + [val]
    
    return new_cuts
  
  def add_cuts(self, r: Result, backend_name):
    if backend_name == 'cvx':
      assert isinstance(r, bg_cvx.CVXResult)
      self.add_cuts_to_cvx(r)
    elif backend_name == 'msk':
      assert isinstance(r, bg_msk.MSKResult)
      self.add_cuts_to_msk(r)
    else:
      raise ValueError(f"not implemented backend {backend_name}")
  
  def add_cuts_to_cvx(self, r: bg_cvx.CVXResult):
    
    _problem = r.problem
    x, y = r.xvar, r.yvar
    
    for cut_type, cut_list in self.cuts.items():
      for ct in cut_list:
        for expr in ct.serialize_to_cvx(x, y):
          _problem._constraints.append(expr)
  
  def add_cuts_to_msk(self, r: bg_msk.MSKResult):
    
    _problem: bg_msk.mf.Model = r.problem
    x, y, z = r.xvar, r.yvar, None
    
    for cut_type, cut_list in self.cuts.items():
      for ct in cut_list:
        for expr, dom in ct.serialize_to_msk(x, y, z):
          _problem.constraint(expr, dom)


@dataclass(order=True)
class BBItem(object):
  def __init__(self, depth, node_id, parent_id, parent_bound, result, bound: Bounds, cuts: Cuts):
    self.priority = 0
    self.depth = depth
    self.node_id = node_id
    self.parent_id = parent_id
    self.parent_bound = parent_bound
    self.result = result
    self.cuts = cuts
    if bound is None:
      self.bound = Bounds()
    else:
      self.bound = bound


def generate_child_items(total_nodes, parent: BBItem, branch: Branch, verbose=False, backend_name='msk',
                         backend_func=None, sdp_solver="MOSEK"):
  Q, q, A, a, b, sign, *_ = parent.qp.unpack()
  # left <=
  left_bounds = Bounds(*parent.bound.unpack())
  left_succ = left_bounds.update_bounds_from_branch(branch, left=True)
  
  # left_r = backend_func(parent.qp, left_bounds, solver=sdp_solver, verbose=verbose, solve=False)
  left_r = backend_func(parent.qp, left_bounds, solver=sdp_solver, verbose=verbose, solve=False, r_parent=parent.result)
  if not left_succ:
    # problem is infeasible:
    left_r.solved = True
    left_r.relax_obj = -1e6
    left_cuts = Cuts()
  else:
    # add cuts to cut off
    left_cuts = parent.cuts.generate_cuts(branch, left_bounds)
    left_cuts.add_cuts(left_r, backend_name)
  
  left_item = BBItem(parent.qp, parent.depth + 1, total_nodes, parent.node_id, parent.result.relax_obj, left_r,
                     left_bounds, left_cuts)
  
  # right >=
  right_bounds = Bounds(*parent.bound.unpack())
  right_succ = right_bounds.update_bounds_from_branch(branch, left=False)
  
  # right_r = backend_func(parent.qp, right_bounds, solver=sdp_solver, verbose=verbose, solve=False)
  right_r = backend_func(parent.qp, right_bounds, solver=sdp_solver, verbose=verbose, solve=False,
                         r_parent=parent.result)
  if not right_succ:
    # problem is infeasible
    right_r.solved = True
    right_r.relax_obj = -1e6
    right_cuts = Cuts()
  else:
    # add cuts to cut off
    right_cuts = parent.cuts.generate_cuts(branch, right_bounds)
    right_cuts.add_cuts(right_r, backend_name)
  
  right_item = BBItem(parent.qp, parent.depth + 1, total_nodes + 1, parent.node_id, parent.result.relax_obj, right_r,
                      right_bounds, right_cuts)
  return left_item, right_item


def bb_box(qp: QP, bounds: Bounds, verbose=False, params=BCParams(), **kwargs):
  print(json.dumps(params.__dict__(), indent=2))
  backend_func = kwargs.get('func')
  backend_name = params.dual_backend
  if backend_func is None:
    if backend_name == 'msk':
      backend_func = bg_msk.shor
    else:
      raise ValueError("not implemented")
  print(f"primal func using {params.sdp_rank_redunction_solver}")
  # root
  root_bound = bounds
  
  # problems
  k = 0
  start_time = time.time()
  print("solving root node")
  root_r = backend_func(qp, root_bound, solver=params.dual_backend, verbose=True, solve=True)
  best_r = root_r
  
  # global cuts
  glc = Cuts()
  
  root = BBItem(qp, 0, 0, -1, 1e8, result=root_r, bound=root_bound, cuts=glc)
  total_nodes = 1
  ub = root_r.relax_obj
  lb = -1e6
  
  ub_dict = {0: ub}
  queue = PriorityQueue()
  queue.put((-ub, root))
  feasible = {}
  
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
      r.solve(primal=params.sdp_rank_redunction_solver, feas_eps=params.feas_eps)
      r.solve_time = time.time() - start_time
    
    if r.relax_obj < lb:
      # prune this tree
      print(f"prune #{item.node_id} by bound")
      continue
    
    if r.true_obj > lb:
      best_r = r
      lb = r.true_obj
    
    gap = (ub - lb) / (abs(lb) + 1e-2)
    
    x = r.xval
    y = r.yval
    res = r.res
    res_norm = r.res_norm
    
    print(
      f"time: {r.solve_time: .2f}/{r.unit_time: .4f} #{item.node_id}, "
      f"depth: {item.depth}, feas: {res.max():.3e}, obj: {r.true_obj:.4f}, "
      f"sdp_obj: {r.relax_obj:.4f}, gap:{gap:.4%} ([{lb: .2f},{ub: .2f}]")
    
    if gap <= params.opt_eps or r.solve_time >= params.time_limit:
      print(f"terminate #{item.node_id} by gap or time_limit")
      break
    
    if res_norm <= params.feas_eps:
      print(f"prune #{item.node_id} by feasible solution")
      feasible[item.node_id] = r
      continue
    
    ## branching
    
    br = Branch()
    br.simple_vio_branch(x, y, res)
    left_item, right_item = generate_child_items(
      total_nodes, item, br, sdp_solver=params.dual_backend, verbose=verbose, backend_name=backend_name,
      backend_func=backend_func)
    total_nodes += 2
    next_priority = - r.relax_obj.round(PRECISION_OBJVAL)
    # next_priority = - r.true_obj
    queue.put((next_priority, right_item))
    queue.put((next_priority, left_item))
    ub_dict[left_item.node_id] = r.relax_obj.round(PRECISION_OBJVAL)
    ub_dict[right_item.node_id] = r.relax_obj.round(PRECISION_OBJVAL)
    #
    
    k += 1
  
  best_r.nodes = total_nodes
  best_r.solve_time = time.time() - start_time
  return best_r
