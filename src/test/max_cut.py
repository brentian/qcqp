import numpy as np

from pyqp.classes import QP


def create_random_mc(n: int):
  e = np.ones((n, 1))
  w = np.random.randint(0, 5, (n, n))
  q = 1 / 2 * (w + w.T) @ e

  A = np.zeros((0, n, n))
  a = np.zeros((0, n, 1))
  b = np.zeros((0, n))
  sign = np.zeros((0, n))
  lb = np.zeros(shape=(n, 1))
  ub = np.ones(shape=(n, 1))

  return QP(-w, q, A, a, b, sign, lb, ub, lb @ lb.T, ub @ ub.T)
