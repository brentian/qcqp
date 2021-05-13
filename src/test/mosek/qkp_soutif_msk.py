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

import sys
import json
import numpy as np
import itertools
import pandas as pd

from pyqp import bg_msk, bg_cvx
from pyqp.classes import QP
from ..qkp_soutif import read_qkp_soutif, qkp_gurobi

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    try:
        fp, n = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/qkp_soutif.py filepath n (number of variables)")
        raise e
    verbose = True

    # start
    Q, q, A, a, b, sign, lb, ub = read_qkp_soutif(filepath=fp, n=int(n))
    qp = QP(Q, q, A, a, b, sign, lb, ub, None, None)

    #
    r_grb_relax = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max", verbose=verbose)
    r_shor = bg_cvx.shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)
    r_shor_msk = bg_msk.shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)

    obj_values = {
        "gurobi_rel": r_grb_relax.true_obj,
        "sdp_qcqp1": r_shor.true_obj,
        "sdp_qcqp1_msk": r_shor_msk.true_obj,
    }
    r_shor_msk.check(qp)
    print(json.dumps(obj_values, indent=2))