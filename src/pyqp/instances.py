import numpy as np
import numpy.linalg as nl

import networkx as nx
import networkx.algorithms as nxa
import matplotlib.pyplot as plt


class QP(object):
  import copy
  cp = copy.deepcopy
  
  def __init__(self, Q, q, A, a, b, sign, cc=None, shape=None):
    self.Q = Q / 2 + Q.T / 2
    self.q = q
    self.A = A / 2 + np.transpose(A, (0, 2, 1)) / 2
    self.a = a
    self.b = b
    self.sign = sign
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
    self.cc = cc
    self.ic = None
    self.Er = None
    if shape is None:
      # infer from data
      self.n, self.d = q.shape
      self.m, *_ = a.shape
    self.construct_chordal()
  
  def __str__(self):
    # todo add a description
    return ""
  
  def __repr__(self):
    return self.description
  
  def unpack(self):
    return self.Q, self.q, self.A, self.a, self.b, self.sign
    # \ self.lb, self.ub, self.ylb, self.yub, [i for i in self.cc] if self.cc is not None else None
  
  ########################
  # eigenvalue decomposition
  # and orthogonal basis
  ########################
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
    self.Aeig = {}
    if decompose_method == 'eig-type1':
      func = self._decompose_matrix
    elif decompose_method == 'eig-type2':
      func = self._decompose_matrix_eig
    else:
      raise ValueError("not such decomposition method")
    self.decom_method = decompose_method
    (upos, ipos), (uneg, ineg), mul, gamma = func(self.Q)
    self.Qpos, self.Qneg = (upos, ipos), (uneg, ineg)
    self.Qmul = mul
    self.Qeig = gamma
    decom_arr = []
    decom_map = np.zeros((self.m + 1, 2, self.n))
    decom_map[0, 0, ipos] = 1
    decom_map[0, 1, ineg] = 1
    decom_arr.append([ipos, ineg])
    
    for i in range(self.m):
      (ap, ip), (an, inn), mul, gamma = func(self.A[i])
      self.Apos[i] = (ap, ip)
      self.Aneg[i] = (an, inn)
      self.Amul[i] = mul
      self.Aeig[i] = gamma
      decom_map[i + 1, 0, ip] = 1
      decom_map[i + 1, 1, inn] = 1
      decom_arr.append([ip, inn])
      if validate:
        d = np.abs(ap @ ap.T - an @ an.T - self.A[i])
        assert d.max() < 1e-3
    self.decom_map = decom_map
    self.decom_arr = decom_arr
  
  def _decompose_matrix(self, A):
    """
    A cholesky like decomposition.
    :param A:
    :return:
    """
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
    
    return (upos, ipos), (uneg, ineg), mul, gamma.reshape((self.n, 1))
  
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
    
    return (upos, ipos), (uneg, ineg), mul, gamma.reshape((self.n, 1))
  
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
    ic = [cc1.intersection(cc2)]
    # now we compute pairwise intersections
    self.cc = cc
    self.ic = ic
    self.Er = [QP.create_er_from_clique(cr, self.n) for k, cr in enumerate(cc)]
  
  @staticmethod
  def create_er_from_clique(cr, n):
    nr = len(cr)
    Er = np.zeros((nr, n))
    for row, col in enumerate(cr):
      Er[row, col] = 1
    return Er


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
    A = - A.transpose(0, 2, 1) @ A
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
    Q = np.random.randint(-4, 4, (n, n))
    # Q = - Q.T @ Q
    A = np.random.randint(-5, 5, (m, n, n))
    # A = np.zeros(A.shape)
    q = np.random.randint(0, 5, (n, 1))
    a = np.random.randint(0, 5, (m, n, 1))
    b = np.ones(m) * 3 * n
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
    q = np.random.randint(0, 5, (n, 1))
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
      cr = sel[i * band: (i + 1) * band]
      cc.append(cr)
      nr = len(cr)
      Er = np.zeros((nr, n))
      for row, col in enumerate(cr):
        Er[row, col] = 1
      qr = Er @ Q @ Er.T
      qc += Er.T @ qr @ Er
    
    qp = QPInstanceUtils._wrapper(qc, q, A, a, b, sign)
    return qp
  
  @staticmethod
  def chordal(n, m):
    intervals = np.random.randint(-100, 100, (n, 2))
    intervals.sort(axis=1)
    G = nx.interval_graph(intervals.tolist())
    nx.draw(G)
    plt.savefig("/tmp/1.png")
    # todo ...
