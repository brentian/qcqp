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

from . import bg_msk, bg_cvx, bg_msk_chordal
from .bb import BBItem, BCParams
from .classes import qp_obj_func, QP, Params, Result, Bounds, Branch, CuttingPlane


class ChordRLTCuttingPlane(CuttingPlane):
  def __init__(self, data):
    super().__init__(data)
  
  def serialize_to_msk(self, xvar, yrvar, yivar, qp):
    expr = bg_msk.expr
    exprs = expr.sub
    exprm = expr.mul
    dom = bg_msk.dom
    i, j, lb, ub = self.data
    
    arr_cc_index = qp.node_to_Er[i]
    arr_ic_index = qp.node_to_Eir[i]
    for idx in arr_cc_index:
      er = qp.Er[idx]
      cr = list(qp.cc[idx])
      _lb_i, _ub_i = lb[i], ub[i]
      _lb_cr, _ub_cr = lb[cr], ub[cr]
      _row = cr.index(i)
      _yr = yrvar[idx]
      _xr = exprm(er, xvar)
      ###############
      # only on rows will not work
      ###############
      # _yvar = yrvar[idx].slice([0, _row], [cr.__len__(), _row + 1])
      # _xvar = xvar.index([_row, 0])
      # expr1, dom1 = exprs(_yvar, exprm(_xvar, _lb_cr + _ub_cr)), bg_msk.dom.lessThan(- _lb_cr * _ub_cr)
      # yield expr1, dom1
      expr1 = exprs(
        exprs(_yr, exprm(_xr, _lb_cr.T)),
        exprm(_ub_cr, expr.transpose(_xr))
      )
      dom1 = dom.lessThan(-_ub_cr @ _lb_cr.T)
      yield expr1, dom1
    pass
    # yij = zvar.index(i, j)
    # xi, xj = zvar.index(n, i), zvar.index(n, j)
    # yij = yvar.index(i, j)
    # xi, xj = xvar.index(i, 0), xvar.index(j, 0)
    # # (xi - li)(xj - uj) <= 0
    # expr1, dom1 = exprs(exprs(yij, exprm(u_j, xi)),
    #                     exprm(l_i, xj)), bg_msk.dom.lessThan(- u_j * l_i)
    # # (xi - ui)(xj - lj) <= 0
    # expr2, dom2 = exprs(exprs(yij, exprm(l_j, xi)),
    #                     exprm(u_i, xj)), bg_msk.dom.lessThan(- u_i * l_j)
    # # (xi - li)(xj - lj) >= 0
    # expr3, dom3 = exprs(exprs(yij, exprm(l_j, xi)),
    #                     exprm(l_i, xj)), bg_msk.dom.greaterThan(- l_j * l_i)
    # # (xi - ui)(xj - uj) >= 0
    # expr4, dom4 = exprs(exprs(yij, exprm(u_j, xi)),
    #                     exprm(u_i, xj)), bg_msk.dom.greaterThan(-u_i * u_j)
    # yield expr1, dom1
    # yield expr2, dom2
    # yield expr3, dom3
    # yield expr4, dom4


def add_rlt_cuts(branch, bounds):
  i = branch.xpivot
  j = branch.xminor
  # u_i, l_i = bounds.xub[i, 0], bounds.xlb[i, 0]
  # u_j, l_j = bounds.xub[j, 0], bounds.xlb[j, 0]
  return ChordRLTCuttingPlane((i, j, bounds.xlb, bounds.xub))


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
    if backend_name == 'msk':
      assert isinstance(r, bg_msk_chordal.MSKChordalResult)
      self.add_cuts_to_msk(r)
    else:
      raise ValueError(f"not implemented backend {backend_name}")
  
  def add_cuts_to_msk(self, r: bg_msk_chordal.MSKChordalResult):
    
    _problem: bg_msk.mf.Model = r.problem
    x, y, z = r.xvar, r.Yrvar, r.Yivar
    for cut_type, cut_list in self.cuts.items():
      for ct in cut_list:
        for expr, dom in ct.serialize_to_msk(x, y, z, r.qp):
          _problem.constraint(expr, dom)


def generate_child_items(total_nodes, parent: BBItem, branch: Branch, verbose=False, backend_name='msk',
                         backend_func=None, sdp_solver="MOSEK"):
  Q, q, A, a, b, sign, *_ = parent.qp.unpack()
  # left <=
  left_bounds = Bounds(*parent.bound.unpack())
  left_succ = left_bounds.update_bounds_from_branch(branch, left=True)
  
  left_r = backend_func(parent.qp, left_bounds, solver=sdp_solver, verbose=verbose, solve=False)
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
  
  right_r = backend_func(parent.qp, right_bounds, solver=sdp_solver, verbose=verbose, solve=False)
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
  backend_name = params.backend_name
  if backend_func is None:
    if backend_name == 'msk':
      backend_func = bg_msk_chordal.ssdp
    else:
      raise ValueError("not implemented")
  
  # root
  root_bound = bounds
  
  # problems
  k = 0
  start_time = time.time()
  print("solving root node")
  root_r = backend_func(qp, root_bound, solver=params.sdp_solver, verbose=True, solve=True)
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
    
    if parent_sdp_val < lb:
      # prune this tree
      print(f"prune #{item.node_id} since parent pruned")
      continue
    
    if not r.solved:
      r.solve()
      r.true_obj = qp_obj_func(item.qp.Q, item.qp.q, r.xval)
      r.solve_time = time.time() - start_time
    
    if r.relax_obj < lb:
      # prune this tree
      print(f"prune #{item.node_id} by bound")
      continue
    
    ub = min(parent_sdp_val, ub)
    r.bound = ub
    if r.true_obj > lb:
      best_r = r
      lb = r.true_obj
    
    gap = (ub - lb) / lb
    
    x = r.xval
    y = r.yval
    res = np.abs(y - x @ x.T)
    
    print(
      f"time: {r.solve_time: .2f} #{item.node_id}, "
      f"depth: {item.depth}, feas: {res.max():.3e}, obj: {r.true_obj:.4f}, "
      f"sdp_obj: {r.relax_obj:.4f}, gap:{gap:.4%} ([{lb: .2f},{ub: .2f}]")
    
    if gap <= params.opt_eps or r.solve_time >= params.time_limit:
      print(f"terminate #{item.node_id} by gap or time_limit")
      break
    
    if res.max() <= params.feas_eps:
      print(f"prune #{item.node_id} by feasible solution")
      feasible[item.node_id] = r
      continue
    
    ## branching    
    br = Branch()
    br.simple_vio_branch(x, y, res)
    left_item, right_item = generate_child_items(
      total_nodes, item, br, sdp_solver=params.sdp_solver, verbose=verbose, backend_name=backend_name,
      backend_func=backend_func)
    total_nodes += 2
    next_priority = - r.relax_obj.round(3)
    queue.put((next_priority, right_item))
    queue.put((next_priority, left_item))
    #
    
    k += 1
  
  best_r.nodes = total_nodes
  best_r.solve_time = time.time() - start_time
  return best_r
