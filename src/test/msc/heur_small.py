import pandas as pd
import sys
from pyqp.bb_msc import *
from pyqp import grb, heur_msc

np.random.seed(1)

if __name__ == '__main__':
    pd.set_option("display.max_columns", None)
    try:
        n, m, *_ = sys.argv[1:]
    except Exception as e:
        print("usage:\n"
              "python tests/random_bb.py n (number of variables) m (num of constraints)")
        raise e
    verbose = True
    evals = []
    params = BCParams()

    # problem
    problem_id = f"{n}:{m}:{0}"
    # start
    qp = QP.create_random_instance(int(n), int(m))
    qp.decompose()
    Q, q, A, a, b, sign, lb, ub, ylb, yub, diagx = qp.unpack()

    #
    r_grb_relax = grb.qp_gurobi(Q, q, A, a, b, sign, lb, ub, sense="max", verbose=verbose)
    r_cvx_shor = bg_cvx.shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)
    r_shor = bg_msk.shor_relaxation(Q, q, A, a, b, sign, lb, ub, solver='MOSEK', verbose=verbose)
    r_msc = bg_msk.msc_relaxation(qp, bounds=None, solver='MOSEK', verbose=verbose)
    r_msc_penalty = heur_msc.penalty_method(r_msc, qp)
    # r_msc_msk2 = bg_msk.msc_relaxation(qp, bounds=None, solver='MOSEK', verbose=verbose, constr_d=True)
    # r_msc_msk3 = bg_msk.msc_relaxation(qp, bounds=None, solver='MOSEK', verbose=verbose, rlt=True)

    obj_values = {
        "gurobi_rel": r_grb_relax.relax_obj,
        "cvx_shor": r_cvx_shor.relax_obj,
        "msk_shor": r_shor.relax_obj,
        "msk_msc": r_msc.relax_obj,
        "msk_msc_with_shor": r_msc_penalty.relax_obj,
        # "msk_msc_with_d": r_msc_msk2.relax_obj,
        # "msk_msc_with_rlt": r_msc_msk3.relax_obj,
    }

    r_msc_penalty.check(qp)
    print(json.dumps(obj_values, indent=2))