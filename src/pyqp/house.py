import numpy as np
import numpy.linalg as la
from pprint import pprint

np.set_printoptions(linewidth=150)
np.random.seed(1)

A = np.random.random((6, 6))


def householder(x, n=0):
  sig = (x ** 2).sum()
  pprint(sig)
  v = x.copy()
  v[0] = 1
  _x = x[0]
  if sig == 0 and _x >= 0:
    beta = 0
  elif sig == 0 and _x < 0:
    beta = -2
  else:
    pprint(_x)
    mu = np.sqrt(_x ** 2 + sig)
    _v = _x - mu if _x <= 0 else - sig / (_x + mu)
    v[0] = _v
    beta = 2 * (_v ** 2) / (sig + _v ** 2)
    v = v / _v
  m = x.shape[0]
  if n > m:
    a = np.zeros(n)
    a[n - m:] = v
    v = a.reshape((n, 1))
  else:
    v = v.reshape((m, 1))
  return beta, v


def house(vec, n):
  vec = np.asarray(vec, dtype=float)
  if vec.ndim != 1:
    raise ValueError("vec.ndim = %s, expected 1" % vec.ndim)
  m = vec.shape[0]
  u = vec
  u[0] = -(np.sum(np.square(u[1:]))) / (vec[0] + np.linalg.norm(vec))
  u = u / np.linalg.norm(vec)
  if n <= m:
    n = m
    a = u
  else:
    a = np.zeros(n)
    a[n - m:] = u
  I = np.eye(n)
  H = I - 2 * (np.outer(a, a))
  return H


def block_diag_house(A, stp=2):
  A_ = A.copy()
  n = A.shape[0]
  H = np.eye(n)
  ncol = 0
  row = -1
  for st in np.arange(0, n, stp):
    row += stp
    for r in np.arange(ncol, ncol + stp):
      if r >= n:
        break
      col = A_[row:, r]
      h = house(col, n)
      H = H @ h
      A_ = h @ A_
      ncol += 1
  
  return H, A_


block_diag_house(A, stp=2)