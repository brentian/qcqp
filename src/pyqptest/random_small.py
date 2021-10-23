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
from pyqp.bg_msk_admm import ADMMParams
from pyqptest.helpers import *

np.random.seed(1)

parser.add_argument(
  "--sparsity",
  type=float,
  default=0.5,
  help="sparsity of the problem, 0-1, 1 means the most dense"
)
if __name__ == '__main__':

  parser.print_usage()
  args = parser.parse_args()
  params = BCParams()
  admmparams = ADMMParams()
  kwargs, r_methods = params.produce_args(parser, METHOD_CODES)
  _ = admmparams.produce_args(parser, METHOD_CODES)

  n, m, pc = args.n, args.m, args.pc
  # problem
  problem_id = f"{n}:{m}:{0}"
  # start
  # qp = QPI.block(n, m, r=2, eps=0.5)
  qp = QPI.normal(int(n), int(m), rho=args.sparsity)
  bd = Bounds(xlb=np.zeros(shape=(n, 1)), xub=np.ones(shape=(n, 1)))

  evals = []
  results = {}
  # run methods
  for k in r_methods:
    func = METHODS[k]
    qp1 = bb_msc.QP(*qp.unpack())
    qp1.decompose()
    r = func(qp1, bd, params=params, admmparams=admmparams)
    reval = r.eval(problem_id)
    evals.append({**reval.__dict__, "method": k})
    results[k] = r

  for k, r in results.items():
    print(f"{k} benchmark @{r.relax_obj}")
    r.check(qp)
    print(r.xval)

  df_eval = pd.DataFrame.from_records(evals)
  print(df_eval)
  print(df_eval[['prob_num', 'solve_time', 'relax_obj', 'method']].to_latex())

  if args.dump_instance:
    pass
