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
from pyqp.bb import BCParams, bb_box
from pyqp.classes import QP
from ..qkp_soutif import *
import tqdm
import os

if __name__ == '__main__':

    try:
        fp = sys.argv[1]
    except Exception as e:
        print("usage:\n"
              "python tests/qkp_soutif_all.py soutif_dir")
        raise e
    files = os.listdir(fp)
    evals = []
    verbose = False
    params = BCParams()
    # start

    for fname in tqdm.tqdm(sorted(files)):
        print(f"running {fname}")
        n = int(fname.split("_")[1])
        prob_num = fname.split(".")[0]
        f_path = os.path.join(fp, fname)

        Q, q, A, a, b, sign, lb, ub = read_qkp_soutif(filepath=f_path, n=int(n))

        # r_grb = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, relax=False, sense="max")
        r_grb_relax = qkp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max")
        # r_shor = shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK')

        qp = QP(Q, q, A, a, b, sign, lb, ub, lb @ lb.T, ub @ ub.T)
        r_bb = bb_box(qp, verbose=True, params=params)

        obj_values = {
            "gurobi_relax": r_grb_relax.true_obj,
            "qcq_bb": r_bb.true_obj
        }

        print(json.dumps(obj_values, indent=2))

        # # validate
        # # grb relaxation
        # xrelg = np.array([i.x for i in x_grb_relax.values()]).reshape((3, 1))
        # print(xrelg.T.dot(A[0]).dot(xrelg).trace() + xrelg.T.dot(a[0]).trace())
        # print(xrelg.T.dot(Q).dot(xrelg).trace() + xrelg.T.dot(q).trace())
        #
        # # sdp by method 1
        # Y, x = _
        # xrelqc = x.value
        # yrelqc = Y.value
        # print(np.abs(yrelqc.diagonal() - xrelqc.flatten()).max())
        # print(yrelqc.T.dot(A[0]).trace() + xrelqc.T.dot(a[0]).trace())
        # print((yrelqc.T @ Q).trace() + q.T.dot(xrelqc).trace())

        # evaluations
        eval_grb_relax = r_grb_relax.eval(prob_num)
        eval_bb = r_bb.eval(prob_num)

        evals += [
            {**eval_grb_relax.__dict__, "method": "gurobi_relax"},
            {**eval_bb.__dict__, "method": "qcq_bb"},
        ]

    df_eval = pd.DataFrame.from_records(evals).set_index(["prob_num", "method"])
    print(df_eval)
    print(df_eval.to_latex())
    df_eval.to_excel("soutif.xlsx")
