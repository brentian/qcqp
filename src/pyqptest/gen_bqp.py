"""
Construct box-constrained nonconvex QP,
  following the same tradition in,
  - Le Thi, H.A., Pham, T.: Dinh. Solving a class of linearly constrained indeﬁnite quadratic problems by D.C. algorithms. J. Glob. Optim. 11, 253–285 (1997)
  and also in,
  - Luo, H., Bai, X., Lim, G., Peng, J.: New global algorithms for quadratic programming with a few negative eigenvalues based on alternative direction method and convex relaxation. Mathematical Programming Computation. 11, 119–171 (2019)
  Basically, the following procedure,
  Q = PTP',
  T = diag(T1, ..., Tn),
  P = W1W2W3, Wj is constructed from wj,
  w -> W:
    W = I − 2ww'/ |w|^2
  where
  wj ~ U(-1,1), q ~ U(-1,1)
  
  Note we use maximizing, then,
    indefinite rank-r matrix is:
    Tk ~ U(1,0), if k = 1, ..., r
    Tk ~ U(-1,0), if k = r+1, ..., n
"""
import numpy as np
from pyqp.bg_grb import *
from pyqp.bg_msk_mix import *
from pyqp.instances import QPInstanceUtils

ANNOTATION = lambda x: f"Box-QP-{x}"


def generate(n, problem_dtls: str):
  ####################
  # defining the rank
  ####################
  r, *_ = problem_dtls.split(",")
  r = int(r)
  ###########
  
  W = np.random.uniform(-1, 1, (3, n, 1))
  Ws = np.empty((3, n, n))
  for i in [0, 1, 2]:
    wi = W[i]
    Ws[i] = np.eye(n) - 2 * wi @ wi.T / (wi.T @ wi)
  P = Ws[0] @ Ws[1] @ Ws[2]
  Tp = np.random.uniform(0, 1, r)
  Tn = np.random.uniform(-1, 0, n - r)
  T = np.hstack([Tp, Tn])
  Q = P @ np.diag(T) @ P.T
  q = np.random.uniform(-1, 1, (n, 1))
  ###########
  
  ############
  # since m = 0, no constraints (except for the bound)
  # and thus the following does not really matter
  ############
  
  A = np.empty((0, n, n))
  
  a = np.empty((0, n, 1))
  b = np.empty((0, n, 1))
  sign = np.ones(shape=0)
  
  #############
  # wrap up
  #############
  qp: QP = QPInstanceUtils._wrapper(Q, q, A, a, b, sign)
  
  qp.note = ANNOTATION(r)
  return qp


def test(qp: QP):
  pass


if __name__ == '__main__':
  import sys
  
  n = int(sys.argv[1])
  seed = int(sys.argv[2])
  np.random.seed(seed)
  bd = Bounds(xlb=np.zeros(n), s=n)
  qp = generate(n, 0)
  rg = qp_gurobi(qp, bd, sense='max')
  print(rg.xval)
