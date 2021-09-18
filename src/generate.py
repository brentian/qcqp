import argparse

import numpy as np

from pyqp.classes import QPI, QP

np.random.seed(1)
parser = argparse.ArgumentParser("QCQP runner")
parser.add_argument("--n", type=int, help="dim of x", default=5)
parser.add_argument("--m", type=int, help="if randomly generated num of constraints", default=5)
parser.add_argument("--num", type=int, help="number of instances", default=1)
parser.add_argument("--rho", type=float, help="density of problem", default=.2)
parser.add_argument("--fpath", type=str, help="file directory", default="data/generated/")

# fix seed
np.random.seed(1)


def main():
  parser.print_usage()
  args = parser.parse_args()
  n, m, fpath, rho, num = args.n, args.m, args.fpath, args.rho, args.num
  for _ in range(num):
    qp: QP = QPI.normal(int(n), int(m), rho=args.rho)
    qp.serialize(wdir=fpath)


if __name__ == '__main__':
  main()
