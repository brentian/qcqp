# branch and bound tree implementations
import json
import time
from dataclasses import dataclass
from queue import PriorityQueue

import numpy as np
from . import bg_msk, bg_cvx
from .classes import QP, Params, qp_obj_func, Result
from .bb import BCParams, BBItem, Cuts, RLTCuttingPlane
from .classes import MscBounds, Branch


class MscBranch(Branch):
    def __init__(self):

        self.ypivot = None
        self.ypivot_val = None
        self.zpivot = None
        self.zpivot_val = None

    def simple_vio_branch(self, y, z, yc, zc, res):
        y_index = np.unravel_index(np.argmax(res, axis=None), res.shape)
        self.ypivot = self.zpivot = y_index
        self.ypivot_val = yc.T[y_index]
        self.zpivot_val = zc.T[y_index]

    def imply_bounds(self, bounds: MscBounds, left=True):
        _succeed = False
        _pivot = self.zpivot
        _val = self.zpivot_val
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


class MscBBItem(BBItem):
    pass


class MscRLT(RLTCuttingPlane):
    def serialize_to_cvx(self, zvar, yvar):
        n, i, u_i, l_i = self.data
        # (xi - li)(xi - ui) <= 0
        expr1 = yvar[n][i, 0] - zvar[n][i, 0] * u_i - l_i * zvar[n][i, 0] + u_i * l_i <= 0
        # (xi - li)(xi - li) >= 0l_j
        expr3 = yvar[n][i, 0] - zvar[n][i, 0] * l_i - l_i * zvar[n][i, 0] + l_i * l_i >= 0
        # (xi - ui)(xi - ui) >= 0
        expr4 = yvar[n][i, 0] - zvar[n][i, 0] * u_i - u_i * zvar[n][i, 0] + u_i * u_i >= 0
        yield expr1
        yield expr3
        yield expr4


def add_rlt_cuts(branch, bounds):
    n, i = branch.zpivot
    u_i, l_i = bounds.zub[n, i, 0], bounds.zlb[n, i, 0]
    return MscRLT((n, i, u_i, l_i))


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



def generate_child_items(total_nodes, parent: MscBBItem, branch: MscBranch, verbose=False, backend_name='msk',
                         backend_func=None, sdp_solver="MOSEK"):
    Q, q, A, a, b, sign, lb, ub, ylb, yub, diagx = parent.qp.unpack()
    # left <=
    left_succ, left_bounds = branch.imply_bounds(parent.bound, left=True)
    left_r = backend_func(parent.qp, bounds=left_bounds, solver=sdp_solver, verbose=verbose, solve=False)
    if not left_succ:
        # problem is infeasible:
        left_r.solved = True
        left_r.relax_obj = -1e6
        left_cuts = MscCuts()
    else:
        # add cuts to cut off
        left_cuts = parent.cuts.generate_cuts(branch, left_bounds)
        left_cuts.add_cuts(left_r, backend_name)

    left_item = MscBBItem(parent.qp, parent.depth + 1, total_nodes, parent.node_id, left_r, left_bounds, left_cuts)

    # right >=
    right_succ, right_bounds = branch.imply_bounds(parent.bound, left=False)
    right_r = backend_func(parent.qp, bounds=right_bounds, solver=sdp_solver, verbose=verbose, solve=False)
    if not right_succ:
        # problem is infeasible
        right_r.solved = True
        right_r.relax_obj = -1e6
        right_cuts = MscCuts()
    else:
        # add cuts to cut off
        right_cuts = parent.cuts.generate_cuts(branch, right_bounds)
        right_cuts.add_cuts(right_r, backend_name)

    right_item = MscBBItem(parent.qp, parent.depth + 1, total_nodes + 1, parent.node_id,
                           right_r, right_bounds, right_cuts)
    return left_item, right_item


def bb_box(qp: QP, verbose=False, params=BCParams()):
    print(json.dumps(params.__dict__(), indent=2))
    backend_name = params.backend_name
    if backend_name == 'msk':
        backend_func = None
    elif backend_name == 'cvx':
        backend_func = bg_cvx.msc_relaxation
    else:
        raise ValueError("not implemented")
    # choose branching

    # problems
    k = 0
    start_time = time.time()
    print("solving root node")
    # root
    qp.decompose()
    root_bound = MscBounds.construct(qp)
    root_r = backend_func(qp, bounds=root_bound, solver=params.sdp_solver, verbose=True, solve=True)
    best_r = root_r

    # global cuts
    glc = MscCuts()
    root = MscBBItem(qp, 0, 0, -1, result=root_r, bound=root_bound, cuts=glc)
    total_nodes = 1
    ub = root_r.relax_obj
    lb = -1e6

    queue = PriorityQueue()
    queue.put((-ub, root))
    feasible = {}

    while not queue.empty():
        priority, item = queue.get()
        r = item.result

        parent_sdp_val = - priority

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

        r.true_obj = qp_obj_func(item.qp.Q, item.qp.q, r.xval)
        ub = min(parent_sdp_val, ub)
        r.bound = ub
        if r.true_obj > lb:
            best_r = r
            lb = r.true_obj

        gap = (ub - lb) / lb

        x = r.xval
        z = r.zval
        y = r.yval
        zc = r.Zval
        yc = r.Yval
        resc = np.abs(yc.T - zc.T ** 2)

        print(
            f"time: {r.solve_time: .2f} #{item.node_id}, "
            f"depth: {item.depth}, "
            f"feas: {resc.max():.3e} "
            f"obj: {r.true_obj:.4f}, "
            f"sdp_obj: {r.relax_obj:.4f}, gap:{gap:.4%} ([{lb: .2f},{ub: .2f}]")

        if gap <= params.opt_eps or r.solve_time >= params.time_limit:
            print(f"terminate #{item.node_id} by gap or time_limit")
            break

        if resc.max() <= params.feas_eps:
            print(f"prune #{item.node_id} by feasible solution")
            feasible[item.node_id] = r
            continue

        ## branching

        br = MscBranch()
        br.simple_vio_branch(y, z, yc, zc, resc)
        left_item, right_item = generate_child_items(
            total_nodes, item, br, sdp_solver=params.sdp_solver, verbose=verbose, backend_name=backend_name,
            backend_func=backend_func)
        total_nodes += 2
        next_priority = - r.relax_obj.round(3)
        queue.put((next_priority, right_item))
        queue.put((next_priority, left_item))
        #

        k += 1

    best_r.solve_time = time.time() - start_time
    return best_r
