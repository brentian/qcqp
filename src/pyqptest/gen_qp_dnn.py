"""
Construct D.C. matrices,
 By Q = RR^T - MM^T
 such that M is doubly nonnegative
"""
from pyqp.instances import QPInstanceUtils
from pyqptest.helpers import *


def generate(n, m):
  Rp = np.random.randint(-5, 5, (n, n))
  Rn = np.random.randint(0, 4, (n, n))
  Q = Rp @ Rp.T - Rn @ Rn.T
  Arp = np.random.randint(-2, 2, (m, n, n))
  Arn = np.random.randint(0, 5, (m, n, n))

  A = Arp @ Arp.transpose(0, 2, 1) - Arn @ Arn.transpose(0, 2, 1)
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


if __name__ == '__main__':
  parser.print_usage()
  args = parser.parse_args()
  params = BCParams()
  kwargs, r_methods = params.produce_args(parser, METHOD_CODES)
  n, m, pc = args.n, args.m, args.pc
  
  # problem
  problem_id = f"{n}:{m}:{0}"
  # start
  bd = Bounds(xlb=np.zeros(shape=(n, 1)), xub=np.ones(shape=(n, 1)))
  qp = generate(n, m)
  
  if params.fpath:
    qp.serialize(params.fpath)
