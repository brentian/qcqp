try:
  import gurobipy as grb
except:
  print("no gurobipy found!")
  grb = None

import numpy as np
import numpy.linalg as nl


# todo use a C struct or CC class
class QP(object):
  def __init__(self, Q, q, A, a, b, sign, lb, ub, ylb, yub, diagx=None, shape=None):
    self.Q = Q / 2 + Q.T / 2
    self.q = q
    self.A = A / 2 + np.transpose(A, (0, 2, 1)) / 2
    self.a = a
    self.b = b
    self.sign = sign
    self.lb = lb
    self.ub = ub
    self.ylb = ylb
    self.yub = yub
    self.diagx = diagx
    if shape is None:
      # infer from Q
      self.n, self.d = q.shape
      self.m, *_ = a.shape
    self.description = self.__str__()
    self.Qpos, self.Qneg, self.Qmul = None, None, None
    self.Apos = None
    self.Aneg = None
    self.Amul = None
    self.zlb = None
    self.zub = None
    self.decom_map = None
  
  def __str__(self):
    # todo add a description
    return ""
  
  def __repr__(self):
    return self.description
  
  def unpack(self):
    return self.Q, self.q, self.A, self.a, self.b, self.sign, \
           self.lb, self.ub, self.ylb, self.yub, self.diagx
  
  @staticmethod
  def create_random_instance(n, m):
    Q = np.random.randint(-5, 5, (n, n))
    A = np.random.randint(-5, 5, (m, n, n))
    q = np.random.randint(0, 5, (n, 1))
    a = np.random.randint(0, 5, (m, n, 1))
    b = np.random.randint(- 2 * n, 2 * n, (m))
    sign = np.ones(shape=m)
    lb = np.zeros(shape=(n, 1))
    ub = np.ones(shape=(n, 1))
    return QP(Q, q, A, a, b, sign, lb, ub, lb @ lb.T, ub @ ub.T)
  
  def decompose(self, validate=False, decompose_method='eig-type1', **kwargs):
    """
    decompose into positive and negative part
    Returns
    -------
    """
    __unused = kwargs
    self.Apos = {}
    self.Aneg = {}
    self.Amul = {}
    if decompose_method == 'eig-type1':
      func = self._decompose_matrix
    elif decompose_method == 'eig-type2':
      func = self._decompose_matrix_eig
    else:
      raise ValueError("not such decomposition method")
    (upos, ipos), (uneg, ineg), mul = func(self.Q)
    self.Qpos, self.Qneg = (upos, ipos), (uneg, ineg)
    self.Qmul = mul
    decom_arr = []
    decom_map = np.zeros((self.m + 1, 2, self.n))
    decom_map[0, 0, ipos] = 1
    decom_map[0, 1, ineg] = 1
    decom_arr.append([ipos, ineg])
    
    for i in range(self.m):
      (ap, ip), (an, inn), mul = func(self.A[i])
      self.Apos[i] = (ap, ip)
      self.Aneg[i] = (an, inn)
      self.Amul[i] = mul
      decom_map[i + 1, 0, ip] = 1
      decom_map[i + 1, 1, inn] = 1
      decom_arr.append([ip, inn])
      if validate:
        d = np.abs(ap @ ap.T - an @ an.T - self.A[i])
        assert d.max() < 1e-3
    self.decom_map = decom_map
    self.decom_arr = decom_arr
  
  def _decompose_matrix(self, A):
    gamma, u = nl.eig(A)
    ipos = (gamma > 0).astype(int)
    ineg = (gamma < 0).astype(int)
    eig = np.diag(gamma)
    upos = u @ np.sqrt(np.diag(ipos) @ eig)
    uneg = u @ np.sqrt(- np.diag(ineg) @ eig)
    #
    ipos, *_ = np.nonzero(ipos)
    ineg, *_ = np.nonzero(ineg)
    mul = np.ones(shape=(self.n, 1))  # todo: fix this for matrix case
    mul[ineg] = -1
    
    return (upos, ipos), (uneg, ineg), mul
  
  def _decompose_matrix_eig(self, A):
    gamma, u = nl.eig(A)
    ipos = (gamma > 0).astype(int)
    ineg = (gamma < 0).astype(int)
    eig = np.diag(gamma)
    upos = u @ np.diag(ipos)
    uneg = u @ np.diag(ineg)
    #
    ipos, *_ = np.nonzero(ipos)
    ineg, *_ = np.nonzero(ineg)
    mul = gamma.reshape((self.n, 1))
    
    return (upos, ipos), (uneg, ineg), mul


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
  def __init__(self, problem=None, yval=0, xval=0, tval=0, relax_obj=0, true_obj=0, bound=0,
               solve_time=0):
    self.problem = problem
    self.yval = yval
    self.xval = xval
    self.tval = tval
    self.relax_obj = relax_obj
    self.true_obj = true_obj
    self.bound = bound
    self.solve_time = solve_time
    self.nodes = 0
  
  def eval(self, problem_id=""):
    return Eval(problem_id, self.solve_time, self.bound, self.true_obj, relax_obj=self.relax_obj, nodes=self.nodes)
  
  def check(self, qp: QP):
    x, y = self.xval, self.yval
    # res = (y - x @ x.T)
    # print(f"y - xx':{res.min(), res.max()}")
    for m in range(qp.A.shape[0]):
      print(f"A*Y + a * x - b:{(x.T @ qp.A[m] * x).trace() + (qp.a[m].T @ x).trace() - qp.b[m]}: {qp.sign[m]}")


keys = ['feas_eps', 'opt_eps', 'time_limit']


class Params(object):
  feas_eps = 1e-4
  opt_eps = 1e-4
  time_limit = 200
  
  def __dict__(self):
    return {k: self.__getattribute__(k) for k in keys}


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
  def __init__(self, zlb=None, zub=None, ylb=None, yub=None, dlb=None, dub=None):
    # sparse implementation
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
    return self.zlb.copy(), self.zub.copy(), \
           self.ylb.copy(), self.yub.copy(), \
           self.dlb.copy(), self.dub.copy()
  
  @classmethod
  def construct(cls, qp, imply_y=True):
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
    newbl = cls(np.array(zlb).round(4), np.array(zub).round(4))
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
