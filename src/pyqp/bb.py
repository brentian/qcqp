# branch and bound tree implementations
import collections
import time

from .classes import QP
from .bg_cvx import *
import pandas as pd
from queue import PriorityQueue
import sys
from dataclasses import dataclass, field
from typing import Any


class Branch(object):
    def __init__(self):
        self.xpivot = None
        self.xpivot_val = None
        self.xminor = None
        self.xminor_val = None
        self.ypivot = None
        self.ypivot_val = None

    def simple_vio_branch(self, x, y, res):
        res_sum = res.sum(0)
        x_index = res_sum.argmax()
        self.xpivot = x_index
        self.xpivot_val = x[self.xpivot, 0].round(6)
        x_minor = res[x_index].argmax()
        self.xminor = x_minor
        self.xminor_val = x[x_minor, 0].round(6)
        self.ypivot = x_index, x_minor
        self.ypivot_val = y[x_index, x_minor].round(6)


class Bounds(object):
    def __init__(self, xlb=None, xub=None, ylb=None, yub=None):
        # sparse implementation
        self.xlb = xlb.copy()
        self.xub = xub.copy()
        self.ylb = ylb.copy()
        self.yub = yub.copy()

    def unpack(self):
        return self.xlb, self.xub, self.ylb, self.yub

    def update_bounds_from_branch(self, branch: Branch, left=True):
        # todo, extend this
        _succeed = False
        _pivot = branch.xpivot
        _val = branch.xpivot_val
        _lb, _ub = self.xlb[_pivot, 0], self.xub[_pivot, 0]
        if left and _val < _ub:
            # <= and a valid upper bound
            self.xub[_pivot, 0] = _val
            _succeed = True
        if not left and _val > _lb:
            self.xlb[_pivot, 0] = _val
            # self.ylb = self.xlb @ self.xlb.T
            _succeed = True

        # after update, check bound feasibility:
        if self.xlb[_pivot, 0] > self.xub[_pivot, 0]:
            _succeed = False
        return _succeed


class CuttingPlane(object):
    def __init__(self, data):
        self.data = data

    def serialize_to_cvx(self, *args, **kwargs):
        pass


class RLTCuttingPlane(CuttingPlane):
    def __init__(self, data):
        super().__init__(data)

    def serialize_to_cvx(self, xvar, yvar):
        i, j, u_i, l_i, u_j, l_j = self.data
        # (xi - li)(xj - uj) <= 0
        expr1 = yvar[i, j] - xvar[i, 0] * u_j - l_i * xvar[j, 0] + u_j * l_i <= 0
        # (xi - ui)(xj - lj) <= 0
        expr2 = yvar[i, j] - xvar[i, 0] * l_j - u_i * xvar[j, 0] + u_i * l_j <= 0
        yield expr1
        yield expr2


class Cuts(object):
    def __init__(self):
        self.cuts = {}

    def add_global_cuts_to_cvx(self, r: CVXResult, branch: Branch, qp: QP, bounds: Bounds):
        _pivot, _val = branch.xpivot, branch.xpivot_val
        _problem = r.problem
        x, y = r.xvar, r.yvar

        # cuts
        _rlt = self.add_rlt_cuts_cvx(branch, bounds)
        new_cuts = Cuts()
        new_cuts.cuts['rlt'] = self.cuts.get('rlt', []) + [_rlt]

        for cut_type, cut_list in new_cuts.cuts.items():
            for ct in cut_list:
                for expr in ct.serialize_to_cvx(x, y):
                    _problem._constraints.append(expr)
        return new_cuts

    def add_rlt_cuts_cvx(self, branch, bounds):
        i = branch.xpivot
        j = branch.xminor
        u_i, l_i = bounds.xub[i, 0], bounds.xlb[i, 0]
        u_j, l_j = bounds.xub[j, 0], bounds.xlb[j, 0]
        return RLTCuttingPlane((i, j, u_i, l_i, u_j, l_j))

    # def update_ybounds_from_branch(self, branch: Branch, left=True):
    #     _pivot = branch.xpivot
    #     _val = branch.xpivot_val
    #
    #     if left:
    #         self.yub[_pivot, :] = (_val ** 2) * self.xub[:,0]
    #         self.yub[:, _pivot] = (_val ** 2) * self.xub[:,0]
    #     else:
    #         self.ylb[_pivot, :] = (_val ** 2) * self.xlb[:,0]
    #         self.ylb[:, _pivot] = (_val ** 2) * self.xlb[:,0]


@dataclass(order=True)
class BBItem(object):
    def __init__(self, qp, depth, node_id, parent_id, result, bound: Bounds, cuts: Cuts):
        self.priority = 0
        self.depth = depth
        self.node_id = node_id
        self.parent_id = parent_id
        self.result = result
        self.qp = qp
        self.cuts = cuts
        if bound is None:
            self.bound = Bounds()
        else:
            self.bound = bound


def generate_child_items(total_nodes, parent: BBItem, branch: Branch):
    Q, q, A, a, b, sign, lb, ub, ylb, yub = parent.qp.unpack()
    # left <=
    left_bounds = Bounds(*parent.bound.unpack())
    left_succ = left_bounds.update_bounds_from_branch(branch, left=True)
    left_qp = QP(
        Q, q, A, a, b, sign,
        *left_bounds.unpack()
    )
    left_r = shor_relaxation(*left_qp.unpack(), verbose=False, solve=False)
    if not left_succ:
        # problem is infeasible:
        left_r.solved = True
        left_r.relax_obj = -1e6
        left_cuts = Cuts()
    else:
        # add cuts to cut off
        left_cuts = parent.cuts.add_global_cuts_to_cvx(left_r, branch, left_qp, left_bounds)

    left_item = BBItem(left_qp, parent.depth + 1, total_nodes, parent.node_id, left_r, left_bounds, left_cuts)

    # right >=
    right_bounds = Bounds(*parent.bound.unpack())
    right_succ = right_bounds.update_bounds_from_branch(branch, left=False)
    right_qp = QP(
        Q, q, A, a, b, sign,
        *right_bounds.unpack()
    )
    right_r = shor_relaxation(*right_qp.unpack(), verbose=False, solve=False)
    if not right_succ:
        # problem is infeasible
        right_r.solved = True
        right_r.relax_obj = -1e6
        right_cuts = Cuts()
    else:
        # add cuts to cut off
        right_cuts = parent.cuts.add_global_cuts_to_cvx(right_r, branch, right_qp, right_bounds)
    right_item = BBItem(right_qp, parent.depth + 1, total_nodes + 1, parent.node_id, right_r, right_bounds, right_cuts)
    return left_item, right_item


def bb_box(qp: QP, verbose=False):
    # some alg hyper params
    feas_eps = 1e-6
    opt_eps = 1e-6

    # choose branching

    # problems
    k = 0
    lb = -1e6
    ub = -1e6
    root_r = shor_relaxation(*qp.unpack(), verbose=verbose, solve=False)
    best_r = root_r
    # root
    root_bound = Bounds(
        qp.lb, qp.ub,
        qp.ylb, qp.yub
    )
    # global cuts
    glc = Cuts()

    root = BBItem(qp, 0, 0, -1, result=root_r, bound=root_bound, cuts=glc)
    total_nodes = 1

    queue = PriorityQueue()
    queue.put((0, root))
    feasible = {}
    start_time = time.time()

    while not queue.empty():
        priority, item = queue.get()
        r = item.result

        if - priority < lb:
            # prune this tree
            print(f"prune #{item.node_id} since parent pruned")
            continue

        if not r.solved:
            r.problem.solve(verbose=verbose)
            r.solve_time = time.time() - start_time
            r.yval = r.yvar.value
            r.xval = xval = r.xvar.value
            r.relax_obj = r.problem.value
            r.true_obj = qp_obj_func(item.qp.Q, item.qp.q, xval)
            r.solved = True

        if r.relax_obj < lb:
            # prune this tree
            print(f"prune #{item.node_id} with prio {priority}")
            continue

        ub = max(r.relax_obj, ub)
        if r.true_obj > lb:
            best_r = r
            lb = r.true_obj
        # r.check(qp)
        x = r.xval
        y = r.yval
        res = np.abs(y - x @ x.T)
        gap = (ub - lb) / lb
        print(
            f"with prio {priority} #{item.node_id}, depth: {item.depth}, obj: {r.true_obj:.4f}, sdp_obj: {r.relax_obj:.4f}, gap:{gap:.4%} ([{lb: .2f},{ub: .2f}]")

        if res.max() <= feas_eps:
            print(f"#{item.node_id} is feasible with prio {priority}")
            feasible[item.node_id] = r
            continue

        ## branching

        br = Branch()
        br.simple_vio_branch(x, y, res)
        left_item, right_item = generate_child_items(total_nodes, item, br)
        total_nodes += 2
        next_priority = - r.relax_obj.round(3)
        queue.put((next_priority, right_item))
        queue.put((next_priority, left_item))
        #

        k += 1
    return best_r
