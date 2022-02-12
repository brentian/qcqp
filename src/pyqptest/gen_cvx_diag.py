"""
Construct D.C. matrices,
 By Q = D - Q-
  xQx = - xQ-x + D•X
 Q- is psd and D is diagonal
"""
import numpy as np
from pyqp.bg_grb import *
from pyqp.bg_msk_mix import *
from pyqp.instances import QPInstanceUtils


def generate(n, m):
  Rp = np.diag(np.random.randint(-5, 5, n))
  Rn = np.random.randint(-4, 4, (n, n))
  # Rn = np.zeros((n, n))
  Q = Rp - Rn @ Rn.T
  
  Arp = np.diag(np.random.randint(-5, 5, n))
  Arn = np.random.randint(0, 5, (m, n, n))
  
  A = Arp - Arn @ Arn.transpose(0, 2, 1)
  q = np.random.randint(0, 5, (n, 1))
  a = np.random.randint(0, 5, (m, n, 1))
  b = np.ones(m) * n
  sign = np.ones(shape=m)
  
  qp: QP = QPInstanceUtils._wrapper(Q, q, A, a, b, sign)
  qp.Qpos = Rp, None
  qp.Qneg = Rn, None
  qp.Apos = Arp, None
  qp.Aneg = Arn, None
  return qp


def sdp(qp):
  Q, q, A, a, b, sign, *_ = qp.unpack()
  m, n, d = A.shape
  xshape = (n, d)
  model = mf.Model('shor_msk')
  
  model.setLogHandler(sys.stdout)
  
  Z = model.variable("Z", dom.inPSDCone(n + 1))
  Y = Z.slice([0, 0], [n, n])
  x = Z.slice([0, n], [n, n + 1])
  
  
  # s = x^Tx
  s = model.variable('s', dom.lessThan(n/2))
  # y = x^TRR^Tx
  y = model.variable("y", [m + 1])
  
  Qpos = qp.Qpos[0]
  Qneg = qp.Qneg[0]
  Apos = qp.Apos[0]
  Aneg = qp.Aneg[0]
  # bounds
  model.constraint(expr.sub(expr.sum(Y.diag()), s), dom.lessThan(0))
  model.constraint(Z.index(n, n), dom.equalsTo(1.))
  model.constraint(
    expr.vstack(0.5, y.index(0), expr.flatten(expr.mul(Qneg.T, x))),
    dom.inRotatedQCone()
  )
  model.constraint(
    expr.vstack(0.5, s, expr.flatten(x)),
    dom.inRotatedQCone()
  )
  # model.constraint(x, dom.inRange(0, 1))
  
  # objectives
  obj_expr = expr.add(
    [expr.dot(Qpos, Y),
     expr.dot(x, q),
     expr.mul(-1.0, y.index(0))]
  )
  for i in range(m):
    if sign[i] == 0:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.equalsTo(b[i]))
    elif sign[i] == -1:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.greaterThan(b[i]))
    else:
      model.constraint(
        expr.add(expr.sum(expr.mulElm(Y, A[i])), expr.dot(x, a[i])),
        dom.lessThan(b[i]))
  #
  
  model.objective(mf.ObjectiveSense.Maximize, obj_expr)
  model.solve()
  xx = x.level()
  print(x.level().reshape((n, 1)).round(6))
  print(Z.level().reshape((n + 1, n + 1)).round(6))
  print(y.level()[0] - xx@Qneg @ Qneg.T @ xx)

if __name__ == '__main__':
  import sys
  
  n = int(sys.argv[1])
  seed = int(sys.argv[2])
  np.random.seed(seed)
  bd = Bounds(xlb=np.zeros(n), xub=100*np.ones(n), s=n/2)
  qp = generate(n, 0)
  sdp(qp)
  rg = qp_gurobi(qp, bd, sense='max')
  print(rg.xval)
