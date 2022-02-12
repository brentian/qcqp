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

from .helpers import *


def run_single_instance(
    qp, bd, evals, results, params, admmparams
):
  for k in params.selected_methods:
    func = METHODS[k]
    
    qp1 = copy.deepcopy(qp)
    special_params = QP_SPECIAL_PARAMS.get(k, {})
    if qp1.Qpos is None or special_params.get("force_decomp", True):
      qp1.decompose(**special_params)
      print(f"redecomposition with method {special_params}")
    try:
      r = func(qp1, bd, params=params, admmparams=admmparams, **special_params)
      reval = r.eval(qp.name)
      evals.append({**reval.__dict__, "method": k})
      results[k] = r
    except Exception as e:
      print(f"method {k} failed")
      import logging
      
      logging.exception(e)
  for k, r in results.items():
    print(f"{k} benchmark @{r.relax_obj}")
    r.check(qp)
    print(r.xval[r.xval > 0])


if __name__ == '__main__':
  
  parser.print_usage()
  args = parser.parse_args()
  params = BCParams()
  admmparams = ADMMParams()
  params.produce_args(parser, METHOD_CODES)
  
  qp = QP.read(params.fpath)
  
  n, m = qp.n, qp.m
  # problem
  problem_id = qp.name if qp.name else f"{n}:{m}:{0}"
  
  # start
  bd = Bounds(xlb=qp.vl, xub=qp.vu)
  
  evals = []
  results = {}
  
  # run method
  run_single_instance(qp, bd, evals, results, params, admmparams)
  df_eval = pd.DataFrame.from_records(evals)
  print(df_eval)
  print(
    df_eval[[
      'prob_num', 'solve_time', 'best_bound', 'best_obj', 'node_time', 'nodes',
      'method'
    ]].to_latex()
  )
  if DEBUG_BB:
    tr.print_diff()
