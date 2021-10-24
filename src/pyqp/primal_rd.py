"""
rank reduction primal methods
"""

import numpy as np

from pyqp.classes import Result

RANDOM_SAMPLE_SIZE = 20000


def so_rank1_normal(r: Result):
  v = np.linalg.cholesky(r.yval)
  obj_max = -1e6
  yb = 0
  xb = 0
  xib = 0
  mean = np.zeros(r.qp.n)
  cov = np.eye(r.qp.n)
  yy = 0
  for idx in range(RANDOM_SAMPLE_SIZE):
    xi = np.random.normal(0, 1, r.qp.n) \
      .reshape((r.qp.n, r.qp.d))
    xd = v @ xi
    Yd = xd @ xd.T
    if r.qp.check(xd):
      pass
    else:
      continue
    obj = (Yd @ r.qp.Q).trace() + (xd.T @ r.qp.q).trace()
    if obj > obj_max:
      obj_max = obj
      yb = Yd
      xb = xd
      xib = xi
    yy += Yd / RANDOM_SAMPLE_SIZE
  
  r.yb = yb
  r.xb = xb
  return 1


PRIMAL_METHOD_ID = {
  1: so_rank1_normal
}
