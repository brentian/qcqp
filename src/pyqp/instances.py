import json
from networkx.algorithms.cuts import volume

import numpy as np
import numpy.linalg as nl

import networkx as nx
import networkx.algorithms as nxa
import matplotlib.pyplot as plt
from collections import defaultdict


class QP(object):
  import copy
  cp = copy.deepcopy
  EPS_EIGEN = 1e-4
  
  def __init__(self, Q, q, A, a, b, sign, al=None, au=None):
    self.Q = Q / 2 + Q.T / 2
    self.q = q
    self.A = A / 2 + np.transpose(A, (0, 2, 1)) / 2
    self.a = a
    self.b = b
    self.sign = sign
    # LHS and RHS
    self.uni = True
    self.al = al
    self.au = au
    self.vl = None
    self.vu = None
    # basic finished
    self.description = self.__str__()
    self.Qpos, self.Qneg, self.Qmul = None, None, None
    self.Apos = None
    self.Aneg = None
    self.Amul = None
    self.Aeig = None
    self.zlb = None
    self.zub = None
    self.decom_map = None
    self.decom_method = ""
    self.cc = None
    self.ic = None
    self.Er = None
    self.name = None
    
    # infer from data
    self.n, self.d = q.shape
    self.m, *_ = a.shape
    self.eigen()
    
    # convexify terms
    self.V = 0
    self.R = {}
    self.l = {}
    self.At = {}
  
  def __str__(self):
    # todo add a description
    return ""
  
  def __repr__(self):
    return self.description
  
  def unpack(self):
    return self.Q, self.q, self.A, self.a, self.b, self.sign, self.al, self.au
  
  ########################
  # homogenization
  ########################
  def construct_homo(self):
    self.Qh = np.bmat([[self.Q, self.q / 2], [self.q.T / 2, np.zeros((1, 1))]]).__array__()
    self.Ah = []
    for i in range(self.m):
      _Ah = np.bmat(
        [
          [self.A[i], self.a[i] / 2],
          [self.a[i].T / 2, np.ones((1, 1)) * (-self.b[i])]
        ]
      )
      self.Ah.append(_Ah.__array__())
  
  ########################
  # store eigenvalues
  ########################
  def precondition(self):
    self.bool_zero_mat = {}
    self.bool_zero_mat[0] = self._bool_zero( - self.Q)
    for i in range(self.m):
      Ai = self.A[i]
      self.bool_zero_mat[i + 1] = self._bool_zero(Ai)
  
  def eigen(self):
    """
    now we simply construct it by eigenvalue decomposition.
    :return:
    """
    self.precondition()
    n, m = self.n, self.m
    self.indefU = {}
    self.indefn = {}
    self.U = np.empty((m + 1, n, n), dtype=np.float)
    self.gamma = np.empty((m + 1, n), dtype=np.float)
    gamma, u = nl.eigh(self.Q)
    indefn = (gamma > self.EPS_EIGEN).sum()
    self.indefn[0] = indefn
    self.indefU[0] = u[:, :indefn]  # only positive part
    self.gamma[0] = gamma
    self.U[0] = u
    for i in range(self.m):
      if self.bool_zero_mat[i]:
        continue
      Ai = self.A[i]
      gamma_i, u_i = nl.eigh(Ai)
      self.gamma[i + 1] = gamma_i
      self.U[i + 1] = u_i
      indefn = (gamma < - self.EPS_EIGEN).sum()
      self.indefn[i + 1] = indefn
      self.indefU[i + 1] = u_i[:, :indefn]
    
    self.bool_spectral = True
  
  @staticmethod
  def _bool_zero(_A):
    if _A is None:
      return True
    return (np.abs(_A) <= 1e-5).all()
  
  @staticmethod
  def _scaled_cholesky(n, A):
    """
    Scale an indefinite A to be PSD
     R = l*I + A
    :param n:
    :param A:
    :return:
    """
    gamma, u = nl.eigh(A)
    l = (0 if min(gamma) > 0 else - min(gamma)) + QP.EPS_EIGEN
    As = A + l * np.eye(n)
    try:
      R = np.linalg.cholesky(As)
    except Exception as e:
      print(np.linalg.eigvalsh(As).min())
      print(e.__traceback__.format_exc())
    return l, R, As
  
  ########################
  # decomposition
  # and orthogonal basis
  ########################
  def decompose(
      self,
      decompose_method='eig-type2',
      convexify_method=0,
      **kwargs
  ):
    """
    decompose into positive and negative part
    Returns
    -------
    """
    __unused = kwargs
    self.Apos = {}
    self.Aneg = {}
    self.Amul = {}
    self.Aeig = {}
    
    if decompose_method == 'eig-type1':
      func = self._decompose_matrix
    elif decompose_method == 'eig-type2':
      func = self._decompose_matrix_eig
    else:
      raise ValueError("not such decomposition method")
    self.decom_method = decompose_method
    decom_arr = []
    decom_map = np.zeros((self.m + 1, 2, self.n))
    if self.Q is not None:
      (upos, ipos), (uneg, ineg), mul, gamma = func(self.Q)
      self.Qpos, self.Qneg = (upos, ipos), (uneg, ineg)
      self.Qmul = mul
      self.Qeig = gamma
      decom_map[0, 0, ipos] = 1
      decom_map[0, 1, ineg] = 1
      decom_arr.append([ipos, ineg])
    
    for i in range(self.m):
      Ai = self.A[i]
      if Ai is not None:
        (ap, ip), (an, inn), mul, gamma = func(Ai)
        self.Apos[i] = (ap, ip)
        self.Aneg[i] = (an, inn)
        self.Amul[i] = mul
        self.Aeig[i] = gamma
        decom_map[i + 1, 0, ip] = 1
        decom_map[i + 1, 1, inn] = 1
        decom_arr.append([ip, inn])
      
      else:
        self.Apos[i] = None
        self.Aneg[i] = None
        self.Amul[i] = None
        self.Aeig[i] = None
    self.decom_map = decom_map
    self.decom_arr = decom_arr
    
    # convexify
    self.convexify(method=convexify_method)
  
  def convexify(self, method: int = 0):
    """
    :param method:
    - 0, use x'x \le s to convexify
    - 1, use Q = VLV', and then (V'x)
    - 2, use a predetermined
      orthonormal (partial) basis V
      this V is potentially low rank.
    
    :return:
    """
    if method == 1:
      # construct convex cones
      for i in range(self.m):
        Ai = self.A[i]
        if not self.bool_zero_mat[i + 1]:
          self.l[i + 1], self.R[i + 1], self.At[i + 1] = self._scaled_cholesky(self.n, Ai)
        else:
          self.l[i + 1], self.R[i + 1], self.At[i + 1] = 0, 0, 0
      if not self.bool_zero_mat[0]:
        self.l[0], self.R[0], self.At[0] = self._scaled_cholesky(self.n, - self.Q)
      else:
        self.l[0], self.R[0], self.At[0] = 0, 0, 0
      self.V = self.U[0]
    
    return 1
  
  def _decompose_matrix(self, A):
    """
    A cholesky like decomposition.
     Q = VLV^T,
     then Q+ = V√L+, Q- = V√L-
    :param A:
    :return:
    """
    gamma, u = nl.eigh(A)
    ipos = (gamma >= 0).astype(int)
    ineg = (gamma < 0).astype(int)
    eig = np.diag(gamma)
    upos = u @ np.sqrt(np.diag(ipos) @ eig)
    uneg = u @ np.sqrt(-np.diag(ineg) @ eig)
    #
    ipos, *_ = np.nonzero(ipos)
    ineg, *_ = np.nonzero(ineg)
    mul = np.ones(shape=(self.n, 1))  # todo: fix this for matrix case
    mul[ineg] = -1
    
    return (upos, ipos), (uneg, ineg), mul, gamma.reshape((self.n, 1))
  
  def _decompose_matrix_eig(self, A):
    gamma, u = nl.eigh(A)
    ipos = (gamma > 0).astype(int)
    ineg = (gamma < 0).astype(int)
    eig = np.diag(gamma)
    upos = u @ np.diag(ipos)
    uneg = u @ np.diag(ineg)
    #
    ipos, *_ = np.nonzero(ipos)
    ineg, *_ = np.nonzero(ineg)
    mul = gamma.reshape((self.n, 1))
    
    return (upos, ipos), (uneg, ineg), mul, gamma.reshape((self.n, 1))
  
  ########################
  # feasibility checking
  ########################
  @staticmethod
  def _check_one_un(x, _A, _a, _b):
    return max((x.T @ _A @ x).trace() + (x.T @ _a).trace() - _b, 0)
  
  @staticmethod
  def _check_one_bi(x, _A, _a, _l, _u):
    _expr = (x.T @ _A @ x).trace() + (x.T @ _a).trace()
    return max(max(_l - _expr, 0), max((_expr - _u), 0))
  
  def check(self, x):
    if self.uni:
      return np.fromiter(
        (self._check_one_un(x, self.A[i], self.a[i], self.b[i]) for i in range(self.m)), dtype=np.float64)
    return np.fromiter(
      (self._check_one_bi(x, self.A[i], self.a[i], self.al[i], self.au[i]) for i in range(self.m)), dtype=np.float64)
  
  ########################
  # cliques and chordal sparsity
  ########################
  def construct_chordal(self):
    g = nx.Graph()
    g.add_edges_from([(i, j) for i, j in zip(*self.Q.nonzero()) if i != j])
    g_chordal, alpha = nxa.complete_to_chordal_graph(g)
    # cc is a list of maximal cliques
    # e.g.:
    cc = list(nxa.chordal_graph_cliques(g_chordal))
    # @test, merge to two groups
    cc1 = {i for cl in cc[:len(cc) // 2] for i in cl}
    cc2 = {i for cl in cc[len(cc) // 2:] for i in cl}
    cc = [cc1, cc2]
    a = cc1.intersection(cc2)
    ic = []
    if len(a) > 0:
      ic = [a]
    # now we compute pairwise intersections
    self.cc = cc
    self.ic = ic
    self.Er = [QP.create_er_from_clique(cr, self.n) for k, cr in enumerate(cc)]
    self.Eir = [QP.create_er_from_clique(cr, self.n) for k, cr in enumerate(ic)]
    self.F = sum(er.T @ er for er in self.Er)
    self.node_to_Er = defaultdict(list)
    self.node_to_Eir = defaultdict(list)
    for idx_cr, cr in enumerate(cc):
      for l in cr:
        self.node_to_Er[l].append(idx_cr)
    for idx_cr, cr in enumerate(ic):
      for l in cr:
        self.node_to_Eir[l].append(idx_cr)
    self.g = g
    self.g_chordal = g_chordal
  
  @staticmethod
  def create_er_from_clique(cr, n):
    nr = len(cr)
    Er = np.zeros((nr, n))
    for row, col in enumerate(cr):
      Er[row, col] = 1
    return Er
  
  ####################
  # serialization
  ####################
  def serialize(self, wdir):
    import json
    import os
    import time
    stamp = time.time()
    
    fname = os.path.join(wdir, f"{self.n}_{self.m}.{stamp}.json")
    data = {}
    data['n'] = self.n
    data['m'] = self.m
    data['d'] = self.d
    data['Q'] = self.Q.flatten().astype(float).tolist()
    data['q'] = self.q.flatten().astype(float).tolist()
    data['A'] = self.A.flatten().astype(float).tolist()
    data['a'] = self.a.flatten().astype(float).tolist()
    data['b'] = self.b.flatten().astype(float).tolist()
    json.dump(data, open(fname, 'w'))
  
  @classmethod
  def read(cls, fpath):
    data = json.load(open(fpath, 'r'))
    n, m, d = data['n'], data['m'], data.get('d', 1)
    sense = data.get('sense', 'max')[:3].lower()
    mu = 1 if sense == 'max' else -1
    try:
      Q = np.array(data['Q']).reshape((n, n)) * mu
    except:
      print("no quad terms in the objective")
      Q = np.zeros(shape=(n, n))
    try:
      q = np.array(data['q']).reshape((n, d)) * mu
    except:
      print("no linear terms in the objective")
      q = np.zeros(shape=(n, d))
    try:
      A = np.array(data['A']).reshape((m, n, n))
    except:
      print("no quad terms in the constraints")
      A = np.zeros(shape=(m, n, n))
    try:
      a = np.array(data['a']).reshape((m, n, d))
    except:
      print("no linear terms in the constraints")
      a = np.zeros(shape=(m, n, d))
    try:
      b = np.array(data['b'])
      sign = np.ones(m)
      al = au = None
      uni = True
    except:
      b = sign = None
      print("do not provide unilateral RHS, this is a bilateral problem")
      try:
        al = np.array(data['al'])
        au = np.array(data['au'])
        uni = False
      except:
        raise ValueError(
          "do not provide bilateral LHS&RHS, no constraints are successfully parsed"
        )
    try:
      vl = np.array(data['vl']).reshape((n, d))
      vu = np.array(data['vu']).reshape((n, d))
    except:
      vl = np.zeros((n, d))
      vu = np.ones((n, d))
    instance = cls(Q, q, A, a, b, sign, al=al, au=au)
    instance.vl = vl
    instance.vu = np.ones((n, d))  # vu * (vu <= 1e6) + 1e6 * (vu > 1e6)  # avoid too big RHS
    instance.name = data.get("name")
    instance.uni = uni
    return instance


class QPInstanceUtils(object):
  """
  create special QP instances
  """
  
  @staticmethod
  def _wrapper(Q, q, A, a, b, sign):
    return QP(Q, q, A, a, b, sign)
  
  @staticmethod
  def cvx(n, m):
    """
    create convex instance
    :param n:
    :param m:
    :return:
    """
    Q = np.random.randint(-4, 4, (n, n))
    # Q = - Q.T @ Q
    A = np.random.randint(-5, 5, (m, n, n))
    # A = np.zeros(A.shape)
    q = np.random.randint(0, 5, (n, 1))
    a = np.random.randint(0, 5, (m, n, 1))
    b = np.ones(m) * 3 * n
    sign = np.ones(shape=m)
    Q = Q.T @ Q
    A = -A.transpose(0, 2, 1) @ A
    return QPInstanceUtils._wrapper(Q, q, A, a, b, sign)
  
  @staticmethod
  def normal(n, m, rho=0.5):
    """
    create blocks instance
    :param n:
    :param m:
    :param rho: density
    :return:
    """
    Q = np.random.randint(-4, 10, (n, n))
    # Q = - Q.T @ Q
    A = np.random.randint(-5, 5, (m, n, n))
    # A = np.zeros(A.shape)
    q = np.random.randint(-5, 5, (n, 1))
    a = np.random.randint(0, 5, (m, n, 1))
    b = np.ones(m) * n
    sign = np.ones(shape=m)
    Q = (np.random.random(Q.shape) <= rho) * (Q + Q.T)
    A = (np.random.random(A.shape) <= rho) @ (A + A.transpose(0, 2, 1))
    return QPInstanceUtils._wrapper(Q, q, A, a, b, sign)
  
  @staticmethod
  def block(n, m, r, eps=0):
    """
    create pure block
    :param n:
    :param m:
    :param r:
    :return:
    """
    q = np.random.randint(-5, 5, (n, 1))
    a = np.random.randint(0, 5, (m, n, 1))
    b = np.ones(m) * 3 * n
    sign = np.ones(shape=m)
    A = np.random.randint(-5, 5, (m, n, n))
    Q = np.random.randint(-4, 4, (n, n))
    qc = 0
    cc = []
    band = np.ceil(n / r).astype(int)
    sel = list(range(n))
    for i in range(n // band):
      cr = sel[i * band:(i + 1) * band]
      cc.append(cr)
      nr = len(cr)
      Er = np.zeros((nr, n))
      for row, col in enumerate(cr):
        Er[row, col] = 1
      qr = Er @ Q @ Er.T
      qc += Er.T @ qr @ Er
    
    qp = QPInstanceUtils._wrapper(qc, q, A, a, b, sign)
    return qp
