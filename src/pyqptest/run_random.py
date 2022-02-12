#  MIT License
#
#  Copyright (c) 2021 Cardinal Operations
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy of
#  this software and associated documentation files (the "Software"), to deal in
#  the Software without restriction, including without limitation the rights to
#  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
#  of the Software, and to permit persons to whom the Software is furnished to do
#  so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
import copy

from pyqptest.helpers import *
from pyqptest import run
import pyqptest.gen_low_rank as gen_ncvx_fixed
import pyqptest.gen_bqp as gen_bqp

from pympler import tracker

tr = tracker.SummaryTracker()

parser.add_argument(
  "--seed",
  type=int,
  default=1,
  help="random seed"
)

if __name__ == '__main__':
  
  parser.print_usage()
  args = parser.parse_args()
  params = BCParams()
  admmparams = ADMMParams()
  params.produce_args(parser, METHOD_CODES)
  
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
  
  qp.name = problem_id
  ########################
  # collect results
  ########################
  
  evals = []
  results = {}
  # run methods
  run.run_single_instance(qp, bd, evals, results, params, admmparams)
  
  df_eval = pd.DataFrame.from_records(evals)
  print(df_eval)
  print(
    df_eval[[
      'prob_num', 'solve_time', 'best_bound', 'best_obj', 'node_time', 'nodes',
      'method'
    ]].to_latex()
  )
  
  if args.dump_instance:
    pass
