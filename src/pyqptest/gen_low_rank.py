"""
Construct nonconvex Q that is
 convex except for a small Qr of rank r
  max xQx = max x[Qr]x - |Rx|^2
"""
import numpy as np
from pyqp.bg_grb import *
from pyqp.bg_msk_mix import *
from pyqp.instances import QPInstanceUtils

ANNOTATION = lambda x: f"Q {int(x)}-indefinite"


def generate(n, m, problem_dtls: str):
  r, *_ = problem_dtls.split(",")
  r = int(r)
  ###########
  ###########
  Rp = np.random.randint(-4, 4, (n, r))
  Rn = np.random.randint(-4, 4, (n, n))
  
  Q = Rp @ Rp.T - Rn @ Rn.T
  
  Ga = np.random.randint(0, 5, (m, r))
  Rr = np.zeros((m, r, r))
  for i in range(m):
    np.fill_diagonal(Rr[i], np.diag(Ga[i]))
  
  Arn = np.random.uniform(-1, 1, (m, n, r))
  
  Arp = np.random.randint(0, 5, (m, n, n))
  
  A = Arp @ Arp.transpose(0, 2, 1) - Arn @ Rr @ Arn.transpose(0, 2, 1)
  q = np.random.randint(0, 5, (n, 1))
  a = np.random.randint(0, 5, (m, n, 1))
  b = np.ones(m) * n
  sign = np.ones(shape=m)
  
  qp: QP = QPInstanceUtils._wrapper(Q, q, A, a, b, sign)
  qp.Qpos = Rp, None
  qp.Qneg = Rn, None
  qp.Apos = Arp, None
  qp.Aneg = Arn, None
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
