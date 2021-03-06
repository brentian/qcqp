"""
This file the branch-and-cut for mixed cone formulation

- use N-MSC in the backend
- trying to branch on rho = x • xi
- then branch for x

"""
import json
from dataclasses import dataclass
from queue import PriorityQueue

import numpy as np
import time

from . import bg_msk_mix, bg_msk, bg_msk_norm
from .classes import qp_obj_func, QP, BCParams, Result, Bounds, Branch, CuttingPlane
from .classes import PRECISION_OBJVAL, PRECISION_SOL

cutting_method = {
  # 'rlt': add_rlt_cuts not needed in this case
  # "sdpcone": SDPSmallCone.apply
}


class MMSCBranch(Branch):
  PRECISION = PRECISION_SOL
  
  def __init__(self, res):
    self.xpivot = None
    self.xpivot_val = None
    self.rhopivot_val = None
    # keep residuals of xx^T
    self.res = res
    self.e = None
  
  def simple_vio_branch(self, x, rho, res, bounds):
    """
    simply create a disjunctive,
      x <= x* or x >= x*
    """
    x_index = res.argmax()
    self.xpivot = x_index
    self.xpivot_val = x[x_index, 0].round(self.PRECISION)
    self.rhopivot_val = rho[x_index, 0].round(self.PRECISION)
  
  def simple_bls_branch(self, x, rho, res, bounds):
    """
    simply create a disjunctive, by ``balancing''
      x <= (l+u)/2 or x >= (l+u)/2
    may be unable to cut off x*
    """
    x_index = res.argmax()
    self.xpivot = x_index
    self.xpivot_val = bounds.xlb[x_index, 0] / 2 + bounds.xub[x_index, 0] / 2
    self.rhopivot_val = rho[x_index, 0].round(self.PRECISION)

  def generate_edges(self, r):
    self.e = tuple(self.res.flatten().argsort()[-2:])


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
    if backend_name == 'msk':
      assert isinstance(r, bg_msk.MSKResult)
      self.add_cuts_to_msk(r)
    else:
      raise ValueError(f"not implemented backend {backend_name}")
  
  def add_cuts_to_msk(self, r: bg_msk.MSKResult):
    
    _problem: bg_msk.mf.Model = r.problem
    x, y, z = r.xvar, r.yvar, None
    
    for cut_type, cut_list in self.cuts.items():
      for ct in cut_list:
        for expr, dom in ct.serialize_to_msk(x, y, z):
          _problem.constraint(expr, dom)


@dataclass(order=True)
class BBItem(object):
  def __init__(self, qp, depth, node_id, parent_id, parent_bound, result, bound: Bounds, cuts: Cuts):
    self.priority = 0
    self.depth = depth
    self.node_id = node_id
    self.parent_id = parent_id
    self.parent_bound = parent_bound
    self.result = result
    self.qp = qp
    self.cuts = cuts
    if bound is None:
      self.bound = Bounds()
    else:
      self.bound = bound


def generate_child_items(total_nodes, parent: BBItem, branch: MMSCBranch, verbose=False, backend_name='msk',
                         backend_func=None, sdp_solver="MOSEK"):
  Q, q, A, a, b, sign, *_ = parent.qp.unpack()
  ##############
  # create new edges
  ##############
  if branch.e:
    new_edges = parent.result.edges.union({branch.e})
  else:
    new_edges = parent.result.edges
  # left <=
  left_bounds = Bounds(*parent.bound.unpack())
  left_succ = left_bounds.update_bounds_from_branch(branch, left=True)
  
  left_r = backend_func(parent.qp, left_bounds, edges=new_edges, solver=sdp_solver, verbose=verbose, solve=False)
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
  
  right_r = backend_func(parent.qp, right_bounds, edges=new_edges, solver=sdp_solver, verbose=verbose, solve=False)
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
      backend_func = bg_msk_mc.socp
    else:
      raise ValueError("not implemented")
  print(f"backend using {backend_func.__name__}")
  
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
    
    r: bg_msk_mc.MSKMixConeResult = item.result
    
    parent_sdp_val = item.parent_bound
    
    if parent_sdp_val < lb:
      # prune this tree
      print(f"prune #{item.node_id} since parent pruned")
      ub_dict.pop(item.node_id)
      continue
    
    if not r.solved:
      r.solve(qp=qp)
      r.solve_time = time.time() - start_time
    
    if r.relax_obj < lb:
      # prune this tree
      print(f"prune #{item.node_id} by bound")
      ub_dict.pop(item.node_id)
      continue
    
    ub = max(ub_dict.values())
    ub_dict.pop(item.node_id)
    if r.true_obj > lb:
      best_r = r
      lb = r.true_obj
    
    gap = (ub - lb) / (abs(lb) + 1e-2)
    
    x = r.xval
    rho = r.rhoval
    res = r.res
    res_norm = r.res_norm
    
    print(
      f"time: {r.solve_time: .2f}/{r.unit_time: .4f} #{item.node_id}, #sc: {len(r.edges)} "
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
    br = MMSCBranch(res)
    br.simple_vio_branch(x, rho, res, item.bound)
    br.generate_edges(r)
    left_item, right_item = generate_child_items(
      total_nodes, item, br, sdp_solver=params.dual_backend, verbose=verbose, backend_name=backend_name,
      backend_func=backend_func)
    total_nodes += 2
    next_priority = - r.relax_obj.round(PRECISION_OBJVAL)
    queue.put((next_priority, right_item))
    queue.put((next_priority, left_item))
    ub_dict[left_item.node_id] = r.relax_obj.round(PRECISION_OBJVAL)
    ub_dict[right_item.node_id] = r.relax_obj.round(PRECISION_OBJVAL)
    #
    
    k += 1
  
  best_r.nodes = total_nodes
  best_r.solve_time = time.time() - start_time
  return best_r


def bb_box_nsocp(
    qp: QP, bounds: Bounds, verbose=False, params=BCParams(), **kwargs
):
  return bb_box(qp, bounds, verbose, params, func=bg_msk_mc.socp, **kwargs)
