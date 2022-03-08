from .helpers import *
import pyqptest.gen_low_rank as gen_ncvx_fixed
import pyqptest.gen_bqp as gen_bqp

# fix seed
np.random.seed(1)

if __name__ == '__main__':
  parser.print_usage()
  args = parser.parse_args()
  
  np.random.seed(args.seed)
  n, m, ptype, btype = args.n, args.m, args.problem_type, args.bound_type
  # problem dtls
  pdtl_str = args.problem_dtls
  bdtl_str = args.bound_dtls
  # problem
  problem_id = f"{n}:{m}:{0}"
  # start
  if ptype == 0:
    qp = QPI.normal(int(n), int(m), rho=0.5)
  elif ptype == 1:
    qp = gen_bqp.generate(int(n), pdtl_str)
  elif ptype == 2:
    qp = gen_ncvx_fixed.generate(int(n), int(m), pdtl_str)
  else:
    raise ValueError("no such problem type defined")
  if btype == 0:
    bd = Bounds(xlb=np.zeros(shape=(n, 1)), xub=np.ones(shape=(n, 1)))
  else:
    bd = Bounds(shape=(n, 1), s=n / 10)
  qp.serialize(wdir=args.fpath)
