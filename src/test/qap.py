import sys
import json
import re
import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
estimated_coord_style = dict(
  marker='o',
  facecolor='none',
  edgecolor='brown'
)


def read_qap_data(path):
  try:
    f = open(path, 'r')
  except:
    raise ValueError("cannot open data file")
  
  dim = int(f.readline().split('\n')[0])
  
  data = (float(a) for line in f for a in re.findall('\d+', line))
  
  arr = np.fromiter(data, dtype=np.float).reshape((2, dim, dim))
  
  A, B = arr
  return A, B


def qap_vec_msc(ab, A, B, rlt=True):
  try:
    import mosek.fusion as mf
    expr = mf.Expr
    dom = mf.Domain
    mat = mf.Matrix
  except Exception as e:
    import logging
    logging.exception(e)
    raise ValueError("no MOSEK found")
  
  n, _ = A.shape
  model = mf.Model('snl_msc_msk')
  X = model.variable("X", [n, n], dom.inRange(0, 1))
  z = model.variable("z", [n ** 2, 1])
  y = model.variable("y", [n ** 2, 1])
  
  # Birkov
  model.constraint(expr.sum(X, 0), dom.equalsTo(1))
  model.constraint(expr.sum(X, 1), dom.equalsTo(1))
  
  x = expr.reshape(X, z.getShape())
  model.constraint(expr.sub(
    expr.mul(u, z),
    x
  ), dom.equalsTo(0))
  
  if rlt:
    # this means you can place on x directly.
    rlt_expr = expr.sub(expr.sum(y), expr.sum(x))
    model.constraint(rlt_expr, dom.lessThan(0))
  
  for i in range(n ** 2):
    conic = expr.vstack(0.5, y.index([i, 0]), z.index([i, 0]))
    model.constraint(conic, dom.inRotatedQCone())
  
  obj_expr = expr.dot(gamma.reshape(n ** 2, 1), y)
  model.objective(mf.ObjectiveSense.Minimize, obj_expr)
  model.setLogHandler(sys.stdout)
  model.solve()
  return X.level(), \
         y.level(), \
         z.level()


if __name__ == '__main__':
  np.random.seed(1)
  A = np.random.randint(0, 5, (3, 3))
  B = np.random.randint(0, 5, (3, 3))
  # not random
  fp = "data/qapdata/sko100e.dat"
  A, B = read_qap_data(fp)
  
  ab = np.kron(B, A)
  ab = (ab.T + ab) / 2
  gamma, u = np.linalg.eig(ab)
  gamma = gamma.astype(float)
  u = u.astype(float)
  Xv, yv, zv = qap_vec_msc(ab, A, B)
