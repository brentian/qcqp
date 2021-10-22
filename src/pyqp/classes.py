from argparse import ArgumentParser

try:
  import gurobipy as grb
except:
  print("no gurobipy found!")
  grb = None

import numpy as np
import numpy.linalg as nl
from .instances import QP, QPInstanceUtils as QPI


class Eval(object):
  def __init__(self, prob_num, solve_time, best_bound, best_obj, relax_obj=0.0, nodes=1):
    self.prob_num = prob_num
    self.solve_time = round(solve_time, 2)
    self.best_bound = best_bound if best_bound == "-" else round(
      best_bound, 2)
    self.best_obj = round(best_obj, 2)
    self.relax_obj = round(relax_obj, 2)
    self.nodes = nodes


class Result:
  def __init__(
      self,
      problem=None,
  ):
    self.problem = problem
    self.relax_obj = 0
    self.true_obj = 0
    self.bound = 0
    self.start_time = 0
    self.solve_time = 0
    self.total_time = 0
    self.build_time = 0
    self.nodes = 0
  
  def eval(self, problem_id=""):
    return Eval(problem_id, self.solve_time, self.bound, self.true_obj, relax_obj=self.relax_obj, nodes=self.nodes)
  
  def check(self, qp: QP):
    pass


keys = ['feas_eps', 'opt_eps', 'time_limit']


class QCQPParser(ArgumentParser):
  pass


class Params(object):
  feas_eps = 1e-4
  opt_eps = 1e-4
  time_limit = 200
  
  def __dict__(self):
    return {k: self.__getattribute__(k) for k in keys}


class BCParams(Params):
  def __init__(self):
    super().__init__()
  
  feas_eps = 1e-3
  opt_eps = 5e-4
  time_limit = 200
  logging_interval = 10
  relax = True # todo fix this
  sdp_solver_backend = 'msk'
  sdp_rank_redunction_solver = 1
  fpath = ''
  
  def produce_args(self, parser: QCQPParser, method_universe):
    parser.print_usage()
    args, _ = parser.parse_known_args()
    
    r = sorted(map(int, args.r.split(",")))
    selected_methods = {method_universe[k] for k in r}
    verbose = args.verbose
    self.time_limit = args.time_limit
    self.sdp_solver_backend = args.bg
    self.sdp_rank_redunction_solver = args.bg_rd
    self.fpath = args.fpath
    kwargs = dict(
      relax=True,
      sense="max",
      verbose=verbose,
      params=self,
      rlt=True
    )
    return kwargs, selected_methods


def qp_obj_func(Q, q, xval: np.ndarray):
  return xval.T.dot(Q).dot(xval).trace() + xval.T.dot(q).trace()


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
    if ylb is not None:
      self.ylb = ylb.copy()
    else:
      self.ylb = xlb @ xlb.T
    if yub is not None:
      self.yub = yub.copy()
    else:
      self.yub = xub @ xub.T
  
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
  
  def serialize_to_msk(self, *args, **kwargs):
    pass
  
  def serialize(self, backend_name, *args, **kwargs):
    if backend_name == 'cvx':
      self.serialize_to_cvx(*args, **kwargs)
    elif backend_name == 'msk':
      self.serialize_to_msk(*args, **kwargs)
    else:
      raise ValueError(f"not implemented backend {backend_name}")


class MscBounds(Bounds):
  def __init__(self, xlb=None, xub=None, zlb=None, zub=None, ylb=None, yub=None, dlb=None, dub=None):
    # sparse implementation
    self.xlb = xlb.copy()
    self.xub = xub.copy()
    self.zlb = zlb.copy()
    self.zub = zub.copy()
    if ylb is not None:
      self.ylb = ylb.copy()
    else:
      self.ylb = None
    if yub is not None:
      self.yub = yub.copy()
    else:
      self.yub = None
    if dub is not None:
      self.dlb = dlb.copy()
    else:
      self.dlb = None
    if dub is not None:
      self.dub = dub.copy()
    else:
      self.dub = None
  
  def unpack(self):
    return self.xlb.copy(), self.xub.copy(), \
           self.zlb.copy(), self.zub.copy(), \
           self.ylb.copy(), self.yub.copy(), \
           self.dlb.copy(), self.dub.copy()
  
  @classmethod
  def construct(cls, qp: QP, imply_y=True):
    # for x
    xlb = np.zeros((qp.n, qp.d))
    xub = np.ones((qp.n, qp.d))
    # for z and y's
    zlb = []
    zub = []
    qpos, qipos = qp.Qpos
    qneg, qineg = qp.Qneg
    
    zub.append(
      ((qpos.T * (qpos.T > 0)).sum(axis=1)
       + (qneg.T * (qneg.T > 0)).sum(axis=1)).reshape(qp.q.shape))
    zlb.append(
      ((qpos.T * (qpos.T < 0)).sum(axis=1)
       + (qneg.T * (qneg.T < 0)).sum(axis=1)).reshape(qp.q.shape))
    
    for i in range(qp.a.shape[0]):
      apos, ipos = qp.Apos[i]
      aneg, ineg = qp.Aneg[i]
      zub.append(
        ((apos.T * (apos.T > 0)).sum(axis=1)
         + (aneg.T * (aneg.T > 0)).sum(axis=1)).reshape(qp.q.shape))
      zlb.append(
        ((apos.T * (apos.T < 0)).sum(axis=1)
         + (aneg.T * (aneg.T < 0)).sum(axis=1)).reshape(qp.q.shape))
    newbl = cls(xlb, xub, np.array(zlb).round(4), np.array(zub).round(4))
    if imply_y:
      newbl.imply_y(qp)
    return newbl
  
  def imply_y(self, qp):
    yub = np.max([self.zlb ** 2, self.zub ** 2], axis=0)
    ylb = np.zeros(yub.shape)
    self.ylb = ylb
    self.yub = yub
    dub = [(qp.decom_map[i] @ yub[i]).reshape((-1, 2)) for i in range(len(yub))]
    self.dlb = np.zeros((len(dub), 2))
    self.dub = np.vstack(dub)
